from setuptools import setup

setup(name='binspec',
      version='0.1',
      description='Tools for modeling and fitting the spectra of multiple-star systems.',
      author='Yuan-Sen Ting',
      author_email='ting@ias.edu',
      license='MIT',
      url='https://github.com/tingyuansen/binspec_plus',
      package_dir = {},
      packages=['binspec'],
      package_data={'binspec':['other_data/*.npz','neural_nets/*.npz']},
      dependency_links = [],
      install_requires=['torch', 'torchvision'])
