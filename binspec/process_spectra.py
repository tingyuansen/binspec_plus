'''
Code for reading in combined and individual visit spectra.
Any way that you can get your hands on the spectra should be fine, as long as you 

Here we adopt APOGEE DR14. Edit os.environs below for a later version of APOGEE data release.
Since our spectral model training set was normalized using the DR12 wavelength definition, 
even thought the spectra are from DR14, we will resample them into DR12 wavelength format.
'''

from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import sys
import os
import subprocess
import copy
import astropy.io.fits as pyfits

from . import utils
from . import spectral_model

os.environ["SDSS_LOCAL_SAS_MIRROR"] = "data.sdss3.org/sas/"
os.environ["RESULTS_VERS"] = "v603"
os.environ["APOGEE_APOKASC_REDUX"] = "v6.2a"
from apogee.spec import continuum

# dr12
master_path = "data.sdss3.org/sas/dr12/apogee/spectro/redux/r5/stars/"
apVisit_path = "data.sdss3.org/sas/dr12/apogee/spectro/redux/r5/apo25m/"

catalog_path = "l25_6d/v603/"
catalog_name = "allStar-v603.fits"
all_visit_name = "allVisit-v603.fits"

# dr14
#master_path = "data.sdss3.org/sas/dr14/apogee/spectro/redux/r8/stars/"
#apVisit_path = "data.sdss3.org/sas/dr14/apogee/spectro/redux/r8/apo25m/"

#catalog_path = "l31c/l31c.2/"
#catalog_name = "allStar-l31c.2.fits"
#all_visit_name = "allVisit-l31c.2.fits"

# download path
download_path = "apogee_download/"

# read in the default wavelength array and the list of pixels used for fitting
wavelength = utils.load_wavelength_array()
cont_pixels = utils.load_cannon_contpixels()

def read_apogee_catalog():
    '''
    read in the catalog of info for all stars in a data release. 
    '''
    filepath = os.path.join(master_path, catalog_path, catalog_name)  # dr14
    filename = os.path.join(download_path, catalog_name)
    try:
        os.makedirs(os.path.dirname(download_path))
    except OSError: pass
    if not os.path.exists(filename):
        print("Downloading : " + catalog_name)
        subprocess.check_call(["wget", filepath, "-O", "%s"%filename])
    all_star_catalog = pyfits.getdata(filename)
    catalog_id = all_star_catalog['APOGEE_ID'].astype("str")
    
    filepath = os.path.join(master_path, catalog_path, all_visit_name)  # dr14
    filename = os.path.join(download_path, all_visit_name)
    if not os.path.exists(filename):
        print("Downloading : " + catalog_name)
        subprocess.check_call(["wget", filepath, "-O", "%s"%filename])
    allvdata = pyfits.getdata(filename)

    fibers = np.zeros((len(all_star_catalog), 
        np.nanmax(all_star_catalog['NVISITS'])), dtype='int') - 1
    for ii in range(len(all_star_catalog)):
        for jj in range(all_star_catalog['NVISITS'][ii]):
            fibers[ii, jj] = allvdata[all_star_catalog['VISIT_PK'][ii][jj]]['FIBERID']

    return all_star_catalog, catalog_id, fibers


def get_combined_spectrum_single_object(apogee_id, catalog = None, save_local = False):
    '''
    apogee_id should be a byte-like object; i.e b'2M13012770+5754582'
    This downloads a single combined spectrum and the associated error array,
        and it normalizes both. 
    '''
    # read in the allStar catalog if you haven't already
    if catalog is None:
        catalog, catalog_id, fibers = read_apogee_catalog()
    
    _COMBINED_INDEX = 1
    
    msk = np.where(catalog_id == apogee_id)[0]
    if not len(msk):
        raise ValueError('the desired Apogee ID was not found in the allStar catalog.')

    field = catalog['FIELD'][msk[0]]
    loc_id = catalog['LOCATION_ID'][msk[0]]

    filename = 'apStar-r5-%s.fits' % apogee_id.strip() # dr12
    #filename = 'apStar-r8-%s.fits' % apogee_id.strip() # dr14
    if loc_id == 1:
        filepath = os.path.join(master_path,'apo1m', field.strip(), filename)
    else:
        filepath = os.path.join(master_path,'apo25m', '%i' % loc_id, filename)
    filename = os.path.join(download_path, filename)

    # download spectrum
    try:
        os.makedirs(os.path.dirname(download_path))
    except OSError: pass
    if not os.path.exists(filename):
        subprocess.check_call(["wget", filepath, '-O', '%s'%filename])

    # read spectrum
    temp1 = pyfits.getdata(filename, ext = 1, header = False)
    temp2 = pyfits.getdata(filename, ext = 2, header = False)
    temp3 = pyfits.getdata(filename, ext = 3, header = False)
    
    if temp1.shape[0] > 6000:
        spec = temp1
        specerr = temp2
        mask = temp3
    else:
        spec = temp1[_COMBINED_INDEX]
        specerr = temp2[_COMBINED_INDEX]
        mask = temp3[_COMBINED_INDEX]

    # convert ApStar grid to Aspcap grid
    spec = toAspcapGrid(spec) # dr12 wavelength format
    specerr = toAspcapGrid(specerr)
    
    # cull dead pixels
    choose = spec <= 0
    spec[choose] = 0.01
    specerr[choose] = np.max(np.abs(spec))*999.
        
    # continuum-normalize
    cont = utils.get_apogee_continuum(wavelength = wavelength, spec = spec, 
        spec_err = specerr, cont_pixels = cont_pixels)
    spec /= cont
    specerr /= cont
    
    if save_local:
        np.savez(download_path + 'spectrum_ap_id_' + str(apogee_id) + '_.npz',
                 spectrum = spec, spec_err = specerr)
    return spec, specerr

    
def get_visit_spectra_individual_object(apogee_id, allvisit_cat = None, save_local = False):
    '''
    Download the visit spectra for an individual object. 
    Get the v_helios from the allStar catalog, which are more accurate than 
        the values reported in the visit spectra fits files. 
    Use the barycentric correction to shift spectra to the heliocentric frame.
    Do a preliminary normalization similar to Bovy's routine
        It's critical that the spectra be renormalized prior to fitting using the 
        spectral_model.get_apogee_continuum() function for self-consistency. 
    apogee_id = byte-like object, i.e. '2M06133561+2433362'
    '''

    filepath = os.path.join(master_path, catalog_path, all_visit_name)  # dr14
    filename = os.path.join(download_path, all_visit_name)
    if not os.path.exists(filename):
        print("Downloading : " + catalog_name)
        subprocess.check_call(["wget", filepath, "-O", "%s"%filename])
    allvisit_cat = pyfits.getdata(filename)
    where_visits = np.where(allvisit_cat['APOGEE_ID'] == apogee_id)[0]
    
    plate_ids = np.array([int(i) for i in allvisit_cat[where_visits]['PLATE']])
    fiberids = allvisit_cat[where_visits]['FIBERID']
    mjds = allvisit_cat[where_visits]['MJD']
    JDs = allvisit_cat[where_visits]['JD']
    vhelios_accurate = allvisit_cat[where_visits]['VHELIO']
    vhelios_synth = allvisit_cat[where_visits]['SYNTHVHELIO']
    snrs = allvisit_cat[where_visits]['SNR']
    BCs = allvisit_cat[where_visits]['BC']

    all_spec, all_err, all_snr, all_hjd, all_vhelio = [], [], [], [], []
    
    for i, pid in enumerate(plate_ids):
        filepath = os.path.join(apVisit_path, '%s'%pid, '%s'%mjds[i])
        filename = 'apVisit-r5-%s-%s-%s.fits' % (pid, mjds[i], fiberids[i]) # dr12
        #filename = 'apVisit-r8-%s-%s-%s.fits' % (pid, mjds[i], fiberids[i]) # dr14
        filepath = os.path.join(filepath, filename)
        filename = os.path.join(download_path, filename)
        try:
            if not os.path.exists(filename):
                subprocess.check_call(["wget", filepath, '-O', '%s'%filename])

            # read spectrum
            spec = np.flipud(pyfits.getdata(filename, ext = 1, header = False).flatten())
            specerr = np.flipud(pyfits.getdata(filename, ext = 2, header = False).flatten())
            mask = np.flipud(pyfits.getdata(filename, ext = 3, header = False).flatten())
            wave = np.flipud(pyfits.getdata(filename, ext = 4, header = False).flatten())

            hdulist = pyfits.open(filename)
            masterheader = hdulist[0].header 
            hdulist.close()
    
            badpix = mask != 0
            if np.sum(badpix)/len(badpix) > 0.5:
                print('too many bad pixels!')
                continue # if 50% or more of the pixels are bad, don't bother.
            specerr[badpix] = 100*np.median(spec)

            # a small fraction of the visit spectra are on a different wavelength 
            # grid than normal (maybe commissioning?). In any case, interpolate them 
            # to the wavelength grid expected by Bovy's visit normalization routine. 
            if len(wave) != 12288:
                print('fixing wavelength...')
                standard_grid = utils.load_visit_wavelength()
                spec = np.interp(standard_grid, wave, spec)
                specerr = np.interp(standard_grid, wave, specerr)
                wave = np.copy(standard_grid)
            
            # preliminary normalization using Bovy's visit normalization routine.
            cont = continuum.fitApvisit(spec, specerr, wave)
            specnorm, errnorm = spec/cont, specerr/cont
        
            # correct for Earth's orbital motion. 
            spec_shift = utils.doppler_shift(wavelength = wave, flux = specnorm, dv = BCs[i])
            spec_err_shift = utils.doppler_shift(wavelength = wave, flux = errnorm, dv = BCs[i]) 
        
            # interpolate to the standard wavelength grid we use for combined spectra.
            interp_spec = np.interp(wavelength, wave, spec_shift)
            interp_err = np.interp(wavelength, wave, spec_err_shift)
            
            # truncate SNR at 200
            interp_err[interp_err < 0.005] = 0.005
            
            all_spec.append(interp_spec)
            all_err.append(interp_err)
            
            # Ideally, get the Julian date of the observations in the heliocentric frame. 
            # Sometimes this isn't available; in that case, get the Julian date in Earth's
            # frame. These differ by at most 8 minutes, so not a big deal. 
            try:
                all_hjd.append(masterheader['HJD'])
            except KeyError:
                all_hjd.append(JDs[i])
            
            # There are a few cases where the v_helios from the allvisit catalog are clearly wrong. 
            if np.abs(vhelios_accurate[i] > 1000) and np.abs(vhelios_synth[i] < 1000):
                vhel = vhelios_synth[i]
            else: vhel = vhelios_accurate[i]
    
            all_snr.append(snrs[i])
            all_vhelio.append(vhel)
        except pyfits.verify.VerifyError:
            print('there was a verification error')
            continue
    all_spec, all_err, all_snr, all_hjd, all_vhelio = np.array(all_spec), \
        np.array(all_err), np.array(all_snr), np.array(all_hjd), np.array(all_vhelio)
    msk = np.argsort(all_hjd)
    all_spec, all_err, all_snr, all_hjd, all_vhelio = all_spec[msk], all_err[msk], \
        all_snr[msk], all_hjd[msk], all_vhelio[msk]
    
    if save_local:
        np.savez('spectra/visit/visit_spectra_ap_id_' + str(apogee_id.decode()) + '_.npz',
                 spectra = all_spec, spec_errs = all_err, snrs = all_snr, 
                 hjds = all_hjd, vhelios = all_vhelio)
                 
    return all_spec, all_err, all_snr, all_hjd, all_vhelio

def renormalize_visit_spectrum(norm_spec, spec_err, label_guess, NN_coeffs_norm,
    NN_coeffs_flux, v_helio):
    '''
    Because visit spectra are initially normalized using a different routine than 
        is implemented in the main spectral modle, then need to be normalized again.
    
    This first obtains the continuum for a synthetic single-star model with parameters
        given by label_guess, multiplies the spectrum by this continuum, and then 
        normalizes that "unnormalized" spectrum using the default normalization routine. 
        It isn't critical that label_guess be vary accurate, since it only supplies a 
        smooth continuum that is divided out again anyway, but it can help a bit. Normally,
        label_guess is obtained by fitting a single-star model to the combined spectrum.
    '''
    star_labels = label_guess[:5]
    labels = np.concatenate([star_labels, [v_helio]]) 
    flux_spec_synth = spectral_model.get_surface_flux_spectrum_single_star(labels = labels, 
        NN_coeffs_norm = NN_coeffs_norm, NN_coeffs_flux = NN_coeffs_flux)
    cont_synth = utils.get_apogee_continuum(wavelength = wavelength, spec = flux_spec_synth, 
        spec_err = None, cont_pixels = cont_pixels)
    flux_spec_data = cont_synth*norm_spec
    cont_data = utils.get_apogee_continuum(wavelength = wavelength, spec = flux_spec_data, 
        spec_err = spec_err, cont_pixels = cont_pixels)
    renormalized_spec = flux_spec_data/cont_data
    return renormalized_spec

def download_visit_spectra_single_object_and_renormalize(apogee_id, p0_single_combined, 
    NN_coeffs_norm, NN_coeffs_flux, allvisit_cat = None, snr_min = 30):
    '''
    Download the visit spectra for one object. Keep the visits with sufficiently high
    SNR. Normalize them in a way consistent with our model.
    '''
    all_spec, all_err, all_snr, all_hjd, all_vhelio = get_visit_spectra_individual_object(
        apogee_id = apogee_id, allvisit_cat = allvisit_cat, save_local = False)
    msk = all_snr > snr_min
    all_spec, all_err, all_snr, all_hjd, all_vhelio = all_spec[msk], all_err[msk], \
        all_snr[msk], all_hjd[msk], all_vhelio[msk]
        
    renorm_specs = []
    for i, spec in enumerate(all_spec):
        renorm_spec = renormalize_visit_spectrum(norm_spec = spec, spec_err = all_err[i],
            label_guess = p0_single_combined, NN_coeffs_norm = NN_coeffs_norm,
            NN_coeffs_flux = NN_coeffs_flux, v_helio = all_vhelio[i])
        renorm_specs.append(renorm_spec)
    renorm_specs = np.array(renorm_specs)
    return renorm_specs, all_err, all_snr, all_hjd, all_vhelio 


def toAspcapGrid(spec):
    """
    Convert a spectrum from apStar grid to the ASPCAP grid (w/o the detector gaps)
    Adapted from Jo Bovy's APOGEE package
    """
    
    apStarBlu_lo,apStarBlu_hi,apStarGre_lo,apStarGre_hi,apStarRed_lo,apStarRed_hi \
        = 322, 3242, 3648, 6048, 6412, 8306 # dr12
    aspcapBlu_start = 0
    aspcapGre_start = apStarBlu_hi-apStarBlu_lo+aspcapBlu_start
    aspcapRed_start = apStarGre_hi-apStarGre_lo+aspcapGre_start
    aspcapTotal = apStarRed_hi-apStarRed_lo+aspcapRed_start

    out= np.zeros(aspcapTotal,dtype=spec.dtype)
    
    out[:aspcapGre_start]= spec[apStarBlu_lo:apStarBlu_hi]
    out[aspcapGre_start:aspcapRed_start]= spec[apStarGre_lo:apStarGre_hi]
    out[aspcapRed_start:]= spec[apStarRed_lo:apStarRed_hi]

    return out


