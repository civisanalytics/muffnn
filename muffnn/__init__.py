import pkg_resources

from .mlp.mlp_classifier import MLPClassifier
from .mlp.mlp_regressor import MLPRegressor
from .autoencoder.autoencoder import Autoencoder
from .fm.fm_classifier import FMClassifier
from .fm.fm_regressor import FMRegressor
from .core import TFPicklingBase


__version__ = pkg_resources.get_distribution('muffnn').version

__all__ = ['MLPClassifier',
           'MLPRegressor',
           'Autoencoder',
           'FMClassifier',
           'FMRegressor',
           'TFPicklingBase',
           '__version__']
