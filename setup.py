import os
from setuptools import find_packages, setup

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

__version__ = None
exec(open(os.path.join(THIS_DIR, "muffnn", "version.py")).read())

setup(
    name='muffnn',
    version=__version__,
    author='Civis Analytics, Inc.',
    author_email='opensource@civisanalytics.com',
    packages=find_packages(),
    url='https://github.com/civisanalytics/muffnn',
    description=('Multilayer Feed-Forward Neural Network (MuFFNN) models with '
                 'TensorFlow and scikit-learn'),
    long_description=open(os.path.join(THIS_DIR, 'README.md')).read(),
    include_package_data=True,
    license="BSD-3",
    install_requires=['numpy',
                      'scipy',
                      'scikit-learn~=0.19',
                      'tensorflow~=1.3']
)
