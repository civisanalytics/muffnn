"""
Tests for the Autoencoder.
"""

import logging
import pprint
from io import BytesIO
import pickle

import numpy as np
import scipy.special
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.testing import (assert_array_almost_equal,
                                   assert_almost_equal)

from muffnn import Autoencoder


_LOGGER = logging.getLogger(__name__)
iris = load_iris()


def sigmoid_cross_entropy(ytrue, ypred):
    """Compute the cross-entropy for a set of [0, 1] labels and output
    probabilities for each label.

    This function computes

        - ytrue * np.log(ypred) - (1 - ytrue) * np.log(1 - ypred)

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

    q = (((ypred == 1.0) & (ytrue != 1.0)) |
         ((ypred == 0.0) & (ytrue != 0.0)))
    ce[q] = np.inf

    return np.sum(ce, axis=1)


def softmax_cross_entropy(ytrue, ypred):
    """Compute the cross-entropy for a multi-class classifier.

    This function computes

        np.sum(-ytrue * np.log(ypred), axis=1)

    It also handles 0's since they can produce NaNs.

    See scikit.metrics.log_loss for a version with clipping of the
    probabilities.
    """

    ce = np.zeros_like(ytrue)

    q = ypred > 0.0
    ce[q] = -1.0 * ytrue[q] * np.log(ypred[q])

    # handle 0's specially
    q = (ypred == 0.0) & (ytrue == 0.0)
    ce[q] = 0.0

    q = (ypred == 0.0) & (ytrue != 0.0)
    ce[q] = np.inf

    return np.sum(ce, axis=1)


def test_check_estimator():
    """Check adherence to Estimator API."""
    check_estimator(Autoencoder)


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

    score = np.mean(sigmoid_cross_entropy(X, Xdec))
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


def _cross_entropy_mse_check(hidden_units=(1,),
                             dropout=0.0,
                             learning_rate=1e-1):
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

    score = np.mean(sigmoid_cross_entropy(X[:, 2:], Xdec[:, 2:]) +
                    np.sum((X[:, 0:2] - Xdec[:, 0:2]) ** 2, axis=1))
    ae_score = ae.score(X)
    assert_almost_equal(score, ae_score)

    _LOGGER.warning("ae hidden_units, dropout, mixed: %s, %g, %g",
                    pprint.pformat(hidden_units), dropout, ae_score)

    assert ae_score < 0.2, ("Autoencoder should have a cross-entropy + MSE "
                            "metric less than 0.2 for the iris features.")

    return ae_score


def test_cross_entropy_mse_single_hidden_unit():
    """Test cross-entropy + MSE metric w/ a single hidden unit."""
    _cross_entropy_mse_check(hidden_units=(1,))


def test_cross_entropy_mse_multiple_hidden_units():
    """Test the cross-entropy + MSE metric w/ 2 hidden units."""
    _cross_entropy_mse_check(hidden_units=(2,))


def test_cross_entropy_mse_multiple_layers():
    """Test cross-entropy + MSE metric w/ layers (3, 2)."""
    _cross_entropy_mse_check(hidden_units=(3, 2,))


def test_cross_entropy_mse_dropout():
    """Test cross-entropy + MSE metric w/ dropout."""
    ae_score_dropout \
        = _cross_entropy_mse_check(hidden_units=(20, 20, 10, 10, 2),
                                   dropout=0.01,
                                   learning_rate=1e-3)

    ae_score_nodropout \
        = _cross_entropy_mse_check(hidden_units=(20, 20, 10, 10, 2),
                                   dropout=0.0,
                                   learning_rate=1e-3)

    assert ae_score_nodropout < ae_score_dropout, (
        "Cross-entropy + MSE metric with dropout should be more than "
        "cross-entropy + MSE metric with no dropout!")


def _cross_entropy_or_discrete_check(discrete_indices=None):
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

    score = np.mean(sigmoid_cross_entropy(X, Xdec))
    ae_score = ae.score(X)
    assert_almost_equal(score, ae_score, decimal=5)

    _LOGGER.warning("ae discrete inds, dropout, cross-entropy: %s, %g, %g",
                    pprint.pformat(discrete_indices), 0.0, ae_score)

    assert ae_score < 1.50, ("Autoencoder should have a cross-entropy less "
                             "than 1.50 for the iris features.")

    return ae_score


def test_cross_entropy_or_discrete():
    """Test to make sure discrete indices or cross-entropy metric
    works in all cases."""
    # Use a variety of cases here.
    scores = []
    for discrete_indices in [None, [0], [1], [2], [2, 0], [2, 3],
                             [0, 2], [0, 1, 2, 3]]:
        scores.append(_cross_entropy_or_discrete_check(
            discrete_indices=discrete_indices))

    # All scores should be equal.
    for score in scores:
        assert_almost_equal(score, scores[0])


def _cat_mse_check(hidden_units=(1,), dropout=0.0, learning_rate=1e-1):
    X = iris.data  # Use the iris features.
    X = MinMaxScaler().fit_transform(X)

    # Use digitize to make a discrete problem for two of four columns.
    cat_cols = []
    cat_size = []
    for i in range(2, X.shape[1]):
        # Vary the number of categories to shake out bugs.
        nmid = 50.0 / i * np.arange(i+1) + 25.0
        bins = ([0.0] +
                list(np.percentile(X[:, i], nmid)) +
                [1.1])
        oe = OneHotEncoder(sparse=False)
        col = oe.fit_transform(
            (np.digitize(X[:, i], bins) - 1.0)[:, np.newaxis])
        cat_size.append(col.shape[1])
        cat_cols.append(col)

    X = np.hstack([X[:, 0:2]] + cat_cols)

    cat_begin = list(2 + np.array([0] + list(np.cumsum(cat_size)[:-1])))

    ae = Autoencoder(hidden_units=hidden_units,
                     n_epochs=5000,
                     random_state=4556,
                     learning_rate=learning_rate,
                     dropout=dropout,
                     metric='mse',
                     categorical_begin=cat_begin,
                     categorical_size=cat_size)
    Xenc = ae.fit_transform(X)
    Xdec = ae.inverse_transform(Xenc)

    assert Xenc.shape == (X.shape[0], hidden_units[-1]), ("Encoded iris data "
                                                          "is not the right"
                                                          " shape!")

    assert Xdec.shape == X.shape, ("Decoded iris data is not the right "
                                   "shape!")

    for begin, size in zip(cat_begin, cat_size):
        assert_array_almost_equal(
            np.sum(Xdec[:, begin: begin + size], axis=1), 1.0)

    score = np.mean(softmax_cross_entropy(X[:, 2:], Xdec[:, 2:]) +
                    np.sum((X[:, 0:2] - Xdec[:, 0:2]) ** 2, axis=1))
    ae_score = ae.score(X)
    assert_almost_equal(score, ae_score, decimal=5)

    _LOGGER.warning("ae hidden_units, dropout, mixed: %s, %g, %g",
                    pprint.pformat(hidden_units), dropout, ae_score)

    assert ae_score < 3.5, ("Autoencoder should have a categorical + MSE "
                            "metric less than 3.5 for the iris features.")

    return ae_score


def test_cat_mse_single_hidden_unit():
    """Test categorical + MSE metric w/ a single hidden unit."""
    _cat_mse_check(hidden_units=(1,))


def test_cat_mse_multiple_hidden_units():
    """Test categorical + MSE metric w/ 2 hidden units."""
    _cat_mse_check(hidden_units=(2,))


def test_cat_mse_multiple_layers():
    """Test categorical + MSE metric w/ layers (3, 2)."""
    _cat_mse_check(hidden_units=(3, 2,))


def test_cat_mse_dropout():
    """Test categorical + MSE metric w/ dropout."""
    ae_score_dropout \
        = _cat_mse_check(hidden_units=(20, 20, 10, 10, 2),
                         dropout=0.01,
                         learning_rate=1e-4)

    ae_score_nodropout \
        = _cat_mse_check(hidden_units=(20, 20, 10, 10, 2),
                         dropout=0.0,
                         learning_rate=1e-4)

    assert ae_score_nodropout < ae_score_dropout, (
        "Categorical + MSE metric with dropout should be more than "
        "categorical + MSE metric with no dropout!")


def _cat_cross_entropy_check(hidden_units=(1,),
                             dropout=0.0,
                             learning_rate=1e-1,
                             use_discrete_indices=False):
    X = iris.data  # Use the iris features.
    X = MinMaxScaler().fit_transform(X)

    # Use digitize to make a discrete problem for the columns.
    # Use both 0,1 columns and one-hot encoded columns.
    for i in range(0, 2):
        bins = [0.0, np.median(X[:, i]), 1.1]
        X[:, i] = np.digitize(X[:, i], bins) - 1.0

    cat_cols = []
    cat_size = []
    for i in range(2, X.shape[1]):
        # Vary the number of categories to shake out bugs.
        nmid = 50.0 / i * np.arange(i+1) + 25.0
        bins = ([0.0] +
                list(np.percentile(X[:, i], nmid)) +
                [1.1])
        oe = OneHotEncoder(sparse=False)
        col = oe.fit_transform(
            (np.digitize(X[:, i], bins) - 1.0)[:, np.newaxis])
        cat_size.append(col.shape[1])
        cat_cols.append(col)

    X = np.hstack([X[:, 0:2]] + cat_cols)

    cat_begin = list(2 + np.array([0] + list(np.cumsum(cat_size)[:-1])))

    if use_discrete_indices:
        ae = Autoencoder(hidden_units=hidden_units,
                         n_epochs=5000,
                         random_state=4556,
                         learning_rate=learning_rate,
                         dropout=dropout,
                         metric='mse',
                         discrete_indices=[0, 1],
                         categorical_begin=cat_begin,
                         categorical_size=cat_size)
    else:
        ae = Autoencoder(hidden_units=hidden_units,
                         n_epochs=5000,
                         random_state=4556,
                         learning_rate=learning_rate,
                         dropout=dropout,
                         metric='cross-entropy',
                         categorical_begin=cat_begin,
                         categorical_size=cat_size)
    Xenc = ae.fit_transform(X)
    Xdec = ae.inverse_transform(Xenc)

    assert Xenc.shape == (X.shape[0], hidden_units[-1]), ("Encoded iris data "
                                                          "is not the right"
                                                          " shape!")

    assert Xdec.shape == X.shape, ("Decoded iris data is not the right "
                                   "shape!")

    for begin, size in zip(cat_begin, cat_size):
        assert_array_almost_equal(
            np.sum(Xdec[:, begin: begin + size], axis=1), 1.0)

    score = np.mean(softmax_cross_entropy(X[:, 2:], Xdec[:, 2:]) +
                    sigmoid_cross_entropy(X[:, :2], Xdec[:, :2]))
    ae_score = ae.score(X)
    assert_almost_equal(score, ae_score, decimal=5)

    _LOGGER.warning("ae hidden_units, dropout, mixed: %s, %g, %g",
                    pprint.pformat(hidden_units), dropout, ae_score)

    assert ae_score < 4.5, ("Autoencoder should have a categorical + "
                            "cross-entropy metric less than 4.5 for the "
                            "iris features.")

    return ae_score


def test_cat_cross_entropy_single_hidden_unit():
    """Test categorical + cross-entropy metric w/ a single hidden unit."""
    score_false = _cat_cross_entropy_check(
        hidden_units=(1,), use_discrete_indices=False)
    score_true = _cat_cross_entropy_check(
        hidden_units=(1,), use_discrete_indices=True)
    assert_almost_equal(score_false, score_true)


def test_cat_cross_entropy_multiple_hidden_units():
    """Test categorical + cross-entropy metric w/ 2 hidden units."""
    score_false = _cat_cross_entropy_check(
        hidden_units=(2,), use_discrete_indices=False)
    score_true = _cat_cross_entropy_check(
        hidden_units=(2,), use_discrete_indices=True)
    assert_almost_equal(score_false, score_true)


def test_cat_cross_entropy_multiple_layers():
    """Test categorical + cross-entropy metric w/ layers (3, 2)."""
    score_false = _cat_cross_entropy_check(
        hidden_units=(3, 2,), use_discrete_indices=False)
    score_true = _cat_cross_entropy_check(
        hidden_units=(3, 2,), use_discrete_indices=True)
    assert_almost_equal(score_false, score_true)


def test_cat_cross_entropy_dropout():
    """Test categorical + cross-entropy metric w/ dropout."""
    dropout_scores = []
    nodropout_scores = []
    for udi in [False, True]:
        ae_score_dropout = _cat_cross_entropy_check(
            hidden_units=(20, 20, 10, 10, 2),
            dropout=0.01,
            learning_rate=1e-4,
            use_discrete_indices=udi)
        dropout_scores.append(ae_score_dropout)

        ae_score_nodropout = _cat_cross_entropy_check(
            hidden_units=(20, 20, 10, 10, 2),
            dropout=0.0,
            learning_rate=1e-4,
            use_discrete_indices=udi)
        nodropout_scores.append(ae_score_nodropout)

        assert ae_score_nodropout < ae_score_dropout, (
            "Categorical + cross-entropy metric with dropout should be more "
            "than categorical + cross-entropy metric with no dropout!")

    assert_almost_equal(dropout_scores[0], dropout_scores[1])
    assert_almost_equal(nodropout_scores[0], nodropout_scores[1])


def _cat_cross_entropy_mse_check(hidden_units=(1,),
                                 dropout=0.0,
                                 learning_rate=1e-1):
    X = iris.data  # Use the iris features.
    X = MinMaxScaler().fit_transform(X)

    # Use digitize to make a discrete problem for the columns.
    for i in range(0, 1):
        bins = [0.0, np.median(X[:, i]), 1.1]
        X[:, i] = np.digitize(X[:, i], bins) - 1.0

    cat_cols = []
    cat_size = []
    for i in range(2, X.shape[1]):
        # Vary the number of categories to shake out bugs.
        nmid = 50.0 / i * np.arange(i+1) + 25.0
        bins = ([0.0] +
                list(np.percentile(X[:, i], nmid)) +
                [1.1])
        oe = OneHotEncoder(sparse=False)
        col = oe.fit_transform(
            (np.digitize(X[:, i], bins) - 1.0)[:, np.newaxis])
        cat_size.append(col.shape[1])
        cat_cols.append(col)

    X = np.hstack([X[:, 0:2]] + cat_cols)

    cat_begin = list(2 + np.array([0] + list(np.cumsum(cat_size)[:-1])))

    ae = Autoencoder(hidden_units=hidden_units,
                     n_epochs=5000,
                     random_state=4556,
                     learning_rate=learning_rate,
                     dropout=dropout,
                     metric='mse',
                     discrete_indices=[0],
                     categorical_begin=cat_begin,
                     categorical_size=cat_size)
    Xenc = ae.fit_transform(X)
    Xdec = ae.inverse_transform(Xenc)

    assert Xenc.shape == (X.shape[0], hidden_units[-1]), ("Encoded iris data "
                                                          "is not the right"
                                                          " shape!")

    assert Xdec.shape == X.shape, ("Decoded iris data is not the right "
                                   "shape!")

    for begin, size in zip(cat_begin, cat_size):
        assert_array_almost_equal(
            np.sum(Xdec[:, begin: begin + size], axis=1), 1.0)

    score = np.mean(softmax_cross_entropy(X[:, 2:], Xdec[:, 2:]) +
                    sigmoid_cross_entropy(X[:, 0:1], Xdec[:, 0:1]) +
                    np.sum((X[:, 1:2] - Xdec[:, 1:2]) ** 2, axis=1))
    ae_score = ae.score(X)
    assert_almost_equal(score, ae_score, decimal=5)

    _LOGGER.warning("ae hidden_units, dropout, mixed: %s, %g, %g",
                    pprint.pformat(hidden_units), dropout, ae_score)

    assert ae_score < 4.5, ("Autoencoder should have a categorical + "
                            "cross-entropy + MSE metric less than 4.5 for "
                            "the iris features.")

    return ae_score


def test_cat_cross_entropy_mse_single_hidden_unit():
    """Test categorical + cross-entropy + MSE metric w/ a single
    hidden unit."""
    _cat_cross_entropy_mse_check(hidden_units=(1,))


def test_cat_cross_entropy_mse_multiple_hidden_units():
    """Test categorical + cross-entropy + MSE metric w/ 2 hidden units."""
    _cat_cross_entropy_mse_check(hidden_units=(2,))


def test_cat_cross_entropy_mse_multiple_layers():
    """Test categorical + cross-entropy + MSE metric w/ layers (3, 2)."""
    _cat_cross_entropy_mse_check(hidden_units=(3, 2,))


def test_cat_cross_entropy_mse_dropout():
    """Test categorical + cross-entropy + MSE metric w/ dropout."""

    ae_score_dropout = _cat_cross_entropy_mse_check(
        hidden_units=(20, 20, 10, 10, 2),
        dropout=0.01,
        learning_rate=1e-4)

    ae_score_nodropout = _cat_cross_entropy_mse_check(
        hidden_units=(20, 20, 10, 10, 2),
        dropout=0.0,
        learning_rate=1e-4)

    assert ae_score_nodropout < ae_score_dropout, (
        "Categorical + cross-entropy + MSE metric with dropout should be "
        "more than categorical + cross-entropy + MSE metric with no dropout!")
