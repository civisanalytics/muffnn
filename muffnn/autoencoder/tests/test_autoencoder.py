"""
Tests for the Autoencoder.
"""

import logging
import pprint
from io import BytesIO
import pickle
from unittest.mock import MagicMock

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.testing import (assert_array_equal,
                                   assert_array_almost_equal,
                                   assert_equal,
                                   assert_almost_equal)
from tensorflow import nn

from muffnn import Autoencoder


_LOGGER = logging.getLogger(__name__)
iris = load_iris()


def test_check_estimator():
    """Check adherence to Estimator API."""
    check_estimator(Autoencoder)


def _mse_check(hidden_units=(1,), dropout=0.0, learning_rate=1e-1):
    X = iris.data  # Use the iris features.
    X = MinMaxScaler().fit_transform(X)

    ae = Autoencoder(hidden_units=hidden_units,
                     n_epochs=4000,
                     random_state=4556,
                     learning_rate=learning_rate,
                     dropout=dropout)
    Xenc = ae.fit_transform(X)
    Xdec = ae.inverse_transform(Xenc)

    assert Xenc.shape == (X.shape[0], hidden_units[-1]), ("Encoded iris data "
                                                          "is not the right"
                                                          " shape!")

    assert Xdec.shape == X.shape, ("Decoded iris data is not the right "
                                   "shape!")

    score = np.mean((Xdec - X) ** 2)
    ae_score = ae.score(X)
    assert_almost_equal(score, ae_score)

    _LOGGER.info("ae hidden_units, dropout, MSE: %s, %g, %g",
                 pprint.pformat(hidden_units), dropout, ae_score)

    assert ae_score < 0.02, ("Autoencoder should have a MSE less than 0.02 "
                             "for the iris features.")

    return ae_score


def test_mse_single_hidden_unit():
    """Test the MSE metric w/ a single hidden unit."""
    _mse_check(hidden_units=(1,))


def test_mse_multiple_hidden_units():
    """Test the MSE metric w/ 2 hidden units."""
    _mse_check(hidden_units=(2,))


def test_mse_multiple_layers():
    """Test the MSE metric w/ layers (3, 2)."""
    _mse_check(hidden_units=(3, 2,))


def test_mse_dropout():
    """Test the MSE metric w/ dropout."""
    ae_score_dropout = _mse_check(hidden_units=(20, 20, 10, 10, 2),
                                  dropout=0.05,
                                  learning_rate=1e-3)

    ae_score_nodropout = _mse_check(hidden_units=(20, 20, 10, 10, 2),
                                    dropout=0.0,
                                    learning_rate=1e-3)

    assert ae_score_nodropout < ae_score_dropout, ("MSE with dropout should "
                                                   "be more than MSE with no "
                                                   " dropout!")


# def test_smoke_cross_entropy():
#     """Make sure it works with the cross-entropy metric."""
#     pass
#
# def test_smoke_mixed_metric():
#     """Make sure it works with a mixed metric."""
#     pass
#
# def test_persistence():
#     """Make sure we can pickle it."""
#     pass
#
# def test_replicability():
#     """Make sure it can be seeded properly."""
#     pass
#
# def test_partial_fit():
#     """"Test partial fit interface"""
#     pass
#
# def test_refitting():
#     """Make sure that refitting resets internals."""
#     pass
