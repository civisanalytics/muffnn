"""
Tests for the Autoencoder.
"""

import logging
import pprint
from io import BytesIO
import pickle

import numpy as np
import scipy.special
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.testing import (assert_array_almost_equal,
                                   assert_almost_equal)

from muffnn import Autoencoder


_LOGGER = logging.getLogger(__name__)
iris = load_iris()


def cross_entropy(ytrue, ypred):
    """Compute the cross-entropy for a binary classifier.

    This function uses the numerically stable version from TensorFlow

        np.max(x, 0) - x * z + np.log(1 + np.exp(-np.abs(x)))

    where `x` is the logit of `ypred` and `z` is `ytrue`.

    It also handles 0's and 1's specially since they can produce NaNs with
    the TensorFlow version.
    """

    ce = np.zeros_like(ytrue)
    x = scipy.special.logit(ypred)
    z = ytrue
    q = np.isfinite(x)
    ce[q] = (np.clip(x[q], 0.0, np.inf) - x[q] * z[q] +
             np.log(1.0 + np.exp(-1.0 * np.abs(x[q]))))

    # handle 1's and 0's specially
    q = (((ypred == 1.0) & (ytrue == 1.0)) |
         ((ypred == 0.0) & (ytrue == 0.0)))
    ce[q] = 0.0

    q = (((ypred == 1.0) & (ytrue == 0.0)) |
         ((ypred == 0.0) & (ytrue == 1.0)))
    ce[q] = np.inf

    return np.sum(ce, axis=1)


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

    score = np.mean(np.sum((Xdec - X) ** 2, axis=1))
    ae_score = ae.score(X)
    assert_almost_equal(score, ae_score)

    _LOGGER.warning("ae hidden_units, dropout, MSE: %s, %g, %g",
                    pprint.pformat(hidden_units), dropout, ae_score)

    assert ae_score < 0.10, ("Autoencoder should have a MSE less than 0.10 "
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


def test_persistence():
    """Make sure we can pickle it."""
    X = iris.data  # Use the iris features.
    X = MinMaxScaler().fit_transform(X)

    ae = Autoencoder(hidden_units=(1,),
                     n_epochs=1000,
                     random_state=4556,
                     learning_rate=1e-2,
                     dropout=0.0)
    Xenc = ae.fit_transform(X)

    b = BytesIO()
    pickle.dump(ae, b)
    ae_pickled = pickle.loads(b.getvalue())
    Xenc_pickled = ae_pickled.transform(X)
    assert_array_almost_equal(Xenc, Xenc_pickled)


def test_replicability():
    """Make sure it can be seeded properly."""
    X = iris.data  # Use the iris features.
    X = MinMaxScaler().fit_transform(X)

    ae1 = Autoencoder(hidden_units=(1,),
                      n_epochs=1000,
                      random_state=4556,
                      learning_rate=1e-2,
                      dropout=0.0)
    Xenc1 = ae1.fit_transform(X)

    ae2 = Autoencoder(hidden_units=(1,),
                      n_epochs=1000,
                      random_state=4556,
                      learning_rate=1e-2,
                      dropout=0.0)
    Xenc2 = ae2.fit_transform(X)

    assert_array_almost_equal(Xenc1, Xenc2)


def test_refitting():
    """Make sure that refitting resets internals."""
    X = iris.data  # Use the iris features.
    X = MinMaxScaler().fit_transform(X)

    # Use digitize to make a discrete problem.
    for i in range(X.shape[1]):
        bins = [0.0, np.median(X[:, i]), 1.1]
        X[:, i] = np.digitize(X[:, i], bins) - 1.0

    ae = Autoencoder(hidden_units=(1,),
                     n_epochs=1000,
                     random_state=4556,
                     learning_rate=1e-2,
                     dropout=0.0,
                     metric='cross-entropy')
    ae.fit(X)
    assert ae.input_layer_sz_ == 4, ("Input layer is the wrong size for the "
                                     "Autoencoder!")

    X_small = X[:, 0:-1]
    assert X_small.shape != X.shape, "Test data for refitting does not work!"
    ae.fit(X_small)
    assert ae.input_layer_sz_ == 3, ("Input layer is the wrong size for the "
                                     "Autoencoder!")


def _cross_entropy_check(hidden_units=(1,), dropout=0.0, learning_rate=1e-1):
    X = iris.data  # Use the iris features.
    X = MinMaxScaler().fit_transform(X)

    # Use digitize to make a discrete problem.
    for i in range(X.shape[1]):
        bins = [0.0, np.median(X[:, i]), 1.1]
        X[:, i] = np.digitize(X[:, i], bins) - 1.0

    ae = Autoencoder(hidden_units=hidden_units,
                     n_epochs=5000,
                     random_state=4556,
                     learning_rate=learning_rate,
                     dropout=dropout,
                     metric='cross-entropy')
    Xenc = ae.fit_transform(X)
    Xdec = ae.inverse_transform(Xenc)

    assert Xenc.shape == (X.shape[0], hidden_units[-1]), ("Encoded iris data "
                                                          "is not the right"
                                                          " shape!")

    assert Xdec.shape == X.shape, ("Decoded iris data is not the right "
                                   "shape!")

    score = np.mean(cross_entropy(X, Xdec))
    ae_score = ae.score(X)
    assert_almost_equal(score, ae_score, decimal=5)

    _LOGGER.warning("ae hidden_units, dropout, cross-entropy: %s, %g, %g",
                    pprint.pformat(hidden_units), dropout, ae_score)

    assert ae_score < 2.80, ("Autoencoder should have a cross-entropy less "
                             "than 2.80 for the iris features.")

    return ae_score


def test_cross_entropy_single_hidden_unit():
    """Test the cross-entropy metric w/ a single hidden unit."""
    _cross_entropy_check(hidden_units=(1,))


def test_cross_entropy_multiple_hidden_units():
    """Test the cross-entropy metric w/ 2 hidden units."""
    _cross_entropy_check(hidden_units=(2,))


def test_cross_entropy_multiple_layers():
    """Test the cross-entropy metric w/ layers (3, 2)."""
    _cross_entropy_check(hidden_units=(3, 2,))


def test_cross_entropy_dropout():
    """Test the cross-entropy metric w/ dropout."""
    ae_score_dropout \
        = _cross_entropy_check(hidden_units=(20, 20, 10, 10, 2),
                               dropout=0.05,
                               learning_rate=1e-4)

    ae_score_nodropout \
        = _cross_entropy_check(hidden_units=(20, 20, 10, 10, 2),
                               dropout=0.0,
                               learning_rate=1e-4)

    assert ae_score_nodropout < ae_score_dropout, (
        "Cross-entropy with dropout should be more than cross-entropy "
        "with no dropout!")


def _mixed_check(hidden_units=(1,), dropout=0.0, learning_rate=1e-1):
    X = iris.data  # Use the iris features.
    X = MinMaxScaler().fit_transform(X)

    # Use digitize to make a discrete problem for two of four columns.
    for i in range(2, X.shape[1]):
        bins = [0.0, np.median(X[:, i]), 1.1]
        X[:, i] = np.digitize(X[:, i], bins) - 1.0

    ae = Autoencoder(hidden_units=hidden_units,
                     n_epochs=5000,
                     random_state=4556,
                     learning_rate=learning_rate,
                     dropout=dropout,
                     metric='mse',
                     discrete_indices=[2, 3])
    Xenc = ae.fit_transform(X)
    Xdec = ae.inverse_transform(Xenc)

    assert Xenc.shape == (X.shape[0], hidden_units[-1]), ("Encoded iris data "
                                                          "is not the right"
                                                          " shape!")

    assert Xdec.shape == X.shape, ("Decoded iris data is not the right "
                                   "shape!")

    score = np.mean(cross_entropy(X[:, 2:], Xdec[:, 2:]) +
                    np.sum((X[:, 0:2] - Xdec[:, 0:2]) ** 2, axis=1))
    ae_score = ae.score(X)
    assert_almost_equal(score, ae_score)

    _LOGGER.warning("ae hidden_units, dropout, mixed: %s, %g, %g",
                    pprint.pformat(hidden_units), dropout, ae_score)

    assert ae_score < 0.2, ("Autoencoder should have a mixed metric less "
                            "than 0.2 for the iris features.")

    return ae_score


def test_mixed_single_hidden_unit():
    """Test the mixed metric w/ a single hidden unit."""
    _mixed_check(hidden_units=(1,))


def test_mixed_multiple_hidden_units():
    """Test the mixed metric w/ 2 hidden units."""
    _mixed_check(hidden_units=(2,))


def test_mixed_multiple_layers():
    """Test the mixed metric w/ layers (3, 2)."""
    _mixed_check(hidden_units=(3, 2,))


def test_mixed_dropout():
    """Test the mixed metric w/ dropout."""
    ae_score_dropout \
        = _mixed_check(hidden_units=(20, 20, 10, 10, 2),
                       dropout=0.01,
                       learning_rate=1e-3)

    ae_score_nodropout \
        = _mixed_check(hidden_units=(20, 20, 10, 10, 2),
                       dropout=0.0,
                       learning_rate=1e-3)

    assert ae_score_nodropout < ae_score_dropout, (
        "Mixed metric with dropout should be more than mixed "
        "metric with no dropout!")


def _check_cross_entropy_mixed_single_hidden_unit(discrete_indices=None):
    X = iris.data  # Use the iris features.
    X = MinMaxScaler().fit_transform(X)

    # Use digitize to make a discrete problem.
    for i in range(X.shape[1]):
        bins = [0.0, np.median(X[:, i]), 1.1]
        X[:, i] = np.digitize(X[:, i], bins) - 1.0

    ae = Autoencoder(hidden_units=(1,),
                     n_epochs=1000,
                     random_state=4556,
                     learning_rate=1e-1,
                     dropout=0.0,
                     metric='cross-entropy',
                     discrete_indices=discrete_indices)
    Xenc = ae.fit_transform(X)
    Xdec = ae.inverse_transform(Xenc)

    assert Xenc.shape == (X.shape[0], 1), ("Encoded iris data "
                                           "is not the right"
                                           " shape!")

    assert Xdec.shape == X.shape, ("Decoded iris data is not the right "
                                   "shape!")

    score = np.mean(cross_entropy(X, Xdec))
    ae_score = ae.score(X)
    assert_almost_equal(score, ae_score, decimal=5)

    _LOGGER.warning("ae discrete inds, dropout, cross-entropy: %s, %g, %g",
                    pprint.pformat(discrete_indices), 0.0, ae_score)

    assert ae_score < 1.50, ("Autoencoder should have a cross-entropy less "
                             "than 1.50 for the iris features.")

    return ae_score


def test_mixed_cross_entropy_single_hidden_unit():
    """Test to make sure discrete indices works in all cases."""
    # Use a variety of cases here.
    for discrete_indices in [None, [0], [1], [2], [2, 0], [2, 3],
                             [0, 2], [0, 1, 2, 3]]:
        _check_cross_entropy_mixed_single_hidden_unit(
            discrete_indices=discrete_indices)
