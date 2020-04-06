import os
from setuptools import find_packages, setup

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

_VERSION = '2.3.1'

CLASSIFIERS = [
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3 :: Only',
]

setup(
    name='muffnn',
    version=_VERSION,
    author='Civis Analytics, Inc.',
    author_email='opensource@civisanalytics.com',
    packages=find_packages(),
    url='https://github.com/civisanalytics/muffnn',
    description=('Multilayer Feed-Forward Neural Network (MuFFNN) models with '
                 'TensorFlow and scikit-learn'),
    long_description=open(os.path.join(THIS_DIR, 'README.rst')).read(),
    include_package_data=True,
    license="BSD-3",
    install_requires=['numpy',
                      'scipy',
                      'scikit-learn>=0.19',
                      'tensorflow>=1.15.2,<2'],
    classifiers=CLASSIFIERS
)
