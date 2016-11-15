"""
Tests for the Autoencoder.
"""

import logging
import pprint
from io import BytesIO
import pickle

import pytest
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.testing import (assert_array_almost_equal,
                                   assert_almost_equal)

from muffnn import Autoencoder


_LOGGER = logging.getLogger(__name__)
iris = load_iris()


def _cross_entropy(ytrue, ypred):
    """Compute the cross-entropy for a classifier.

    This function is used instead of the sklearn version since
    we want the cross-entropy for each example, not the total.

    This function computes

        np.sum(-ytrue * np.log(ypred), axis=1)

    It also handles 0's since they can produce NaNs.

    If the input arrays both have one-dimension, then binary classes are
    assumed with the `ypred` being probabilities for the positive class
    and (0, 1) marking negative and positive examples in `ytrue`.
    """

    if ytrue.ndim == 1 and ypred.ndim == 1:
        # Convert 1d inputs to 2d assuming binary classes.
        _ytrue = np.zeros((ytrue.shape[0], 2))
        q = ytrue == 1
        _ytrue[q, 1] = 1
        _ytrue[~q, 0] = 1

        _ypred = np.zeros((ytrue.shape[0], 2))
        _ypred[:, 0] = 1.0 - ypred
        _ypred[:, 1] = ypred
    else:
        _ytrue = ytrue
        _ypred = ypred

    ce = np.zeros_like(_ytrue)

    q = _ypred > 0.0
    ce[q] = -1.0 * _ytrue[q] * np.log(_ypred[q])

    # handle 0's specially
    q = (_ypred == 0.0) & (_ytrue == 0.0)
    ce[q] = 0.0

    q = (_ypred == 0.0) & (_ytrue != 0.0)
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

    # Use digitize to make a binary features.
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
    assert ae.input_layer_size_ == 4, ("Input layer is the wrong size for "
                                       "the Autoencoder!")

    X_small = X[:, 0:-1]
    assert X_small.shape != X.shape, "Test data for refitting does not work!"
    ae.fit(X_small)
    assert ae.input_layer_size_ == 3, ("Input layer is the wrong size for "
                                       "the Autoencoder!")


def test_errors_unallowed_metric():
    """Make sure unallowed metrics cause an error."""

    # This data will not actually be fit.
    # I am just using it to call the `fit` method.
    X = np.ones((1000, 4))

    # There are two code paths for this test. One for all features
    # using the default metric and one for a mix of metrics.

    # All features use the default metric.
    ae = Autoencoder(metric='blah')
    with pytest.raises(ValueError) as e:
        ae.fit(X)
        assert "Metric 'blah'" in str(e), (
            "Wrong error raised for testing unallowed metrics!")

    # Not all features use the default metric.
    ae = Autoencoder(metric='blah', binary_indices=[0])
    with pytest.raises(ValueError) as e:
        ae.fit(X)
        assert "Metric 'blah'" in str(e), (
            "Wrong error raised for testing unallowed metrics!")


def test_errors_metric_output_activation():
    """Make sure cross-entropy metric with activations not equal to
    `tensorflow.nn.sigmoid` or `tensorflow.nn.softmax` fails."""

    # This data will not actually be fit.
    # I am just using it to call the `fit` method.
    X = np.ones((1000, 4))

    # There are two code paths for this test. One for all features
    # using the default metric and one for a mix of metrics.

    # All features use the default metric.
    ae = Autoencoder(metric='cross-entropy', output_activation=tf.exp)
    with pytest.raises(ValueError) as e:
        ae.fit(X)
        assert "'cross-entropy' metric!" in str(e), (
            "Wrong error raised for testing 'cross-entropy' metric with "
            "output activation that is not allowed for all features!")

    # Not all features use the default metric.
    ae = Autoencoder(metric='cross-entropy',
                     output_activation=tf.exp,
                     binary_indices=[0])
    with pytest.raises(ValueError) as e:
        ae.fit(X)
        assert "'cross-entropy' metric!" in str(e), (
            "Wrong error raised for testing 'cross-entropy' metric with "
            "output activation that is not allowed for a subset of features!")


def _check_ae(max_score,
              hidden_units=(1,),
              dropout=None,
              learning_rate=1e-1,
              sparse_type=None,
              bin_inds=None,
              bin_inds_to_use=None,
              cat_inds=None,
              n_epochs=5000,
              metric='mse'):
    """Helper function for testing the Autoencoder.

    This function does in order:

    1. Loads the Iris data.
    2. Converts columns to either binary (bin_inds) or categorical (cat_inds).
    3. Converts the data to a sparse type (sparse_type).
    4. Builds and trains the autoencoder (learning_rate, n_epochs, dropout,
        hidden_units, metric, bin_inds_to_use)
    5. Tests the outputs (max_score).

    The `bin_inds_to_use` parameter in particular specifies which columns
    the autoencoder is explicitly told are binary values.
    """
    # Use the iris features.
    X = iris.data
    X = MinMaxScaler().fit_transform(X)

    # Make some columns binary or one-hot encoded.
    cat_size = []
    cat_begin = []
    binary_inds = []
    keep_cols = []
    def_inds = []
    num_cols = 0
    for i in range(X.shape[1]):
        if bin_inds is not None and i in bin_inds:
            bins = [0.0, np.median(X[:, i]), 1.1]
            keep_cols.append((np.digitize(X[:, i], bins) - 1.0)[:, np.newaxis])
            binary_inds.append(num_cols)
            num_cols += 1
        elif cat_inds is not None and i in cat_inds:
            # Vary the number of categories to shake out bugs.
            bins = np.percentile(X[:, i], 100.0 / (i + 3) * np.arange(i + 4))
            bins[0] = 0.0
            bins[-1] = 1.1
            oe = OneHotEncoder(sparse=False)
            col = oe.fit_transform(
                (np.digitize(X[:, i], bins) - 1.0)[:, np.newaxis])
            keep_cols.append(col)
            cat_begin.append(num_cols)
            cat_size.append(col.shape[1])
            num_cols += col.shape[1]
        else:
            keep_cols.append(X[:, i:i+1])
            def_inds.append(num_cols)
            num_cols += 1

    X = np.hstack(keep_cols)

    if len(cat_size) == 0:
        cat_indices = None
    else:
        cat_indices = np.hstack([np.array(cat_begin)[:, np.newaxis],
                                 np.array(cat_size)[:, np.newaxis]])
        assert cat_indices.shape[1] == 2, ("Categorical indices are the "
                                           "wrong shape!")

    if len(binary_inds) == 0:
        binary_inds = None

    if sparse_type is not None:
        X = getattr(sp, sparse_type + '_matrix')(X)

    # For sigmoid metric runs, we can set the metric or use binary_indices.
    # This if handles those cases.
    if bin_inds_to_use is None and binary_inds is not None:
        bin_inds_to_use = binary_inds
    elif bin_inds_to_use == -1:
        bin_inds_to_use = None

    ae = Autoencoder(hidden_units=hidden_units,
                     n_epochs=n_epochs,
                     random_state=4556,
                     learning_rate=learning_rate,
                     dropout=dropout,
                     metric=metric,
                     binary_indices=bin_inds_to_use,
                     categorical_indices=cat_indices)

    Xenc = ae.fit_transform(X)
    Xdec = ae.inverse_transform(Xenc)

    if sparse_type is not None:
        X = X.todense().A

    assert Xenc.shape == (X.shape[0], hidden_units[-1]), ("Encoded iris data "
                                                          "is not the right"
                                                          " shape!")

    assert Xdec.shape == X.shape, ("Decoded iris data is not the right "
                                   "shape!")

    # One-hot encoded stuff should come back out normalized.
    if cat_size is not None:
        for begin, size in zip(cat_begin, cat_size):
            assert_array_almost_equal(
                np.sum(Xdec[:, begin: begin + size], axis=1), 1.0, decimal=5)

    # Compute and test the scores.
    scores = 0.0
    for i in range(X.shape[1]):
        if binary_inds is not None and i in binary_inds:
            scores += _cross_entropy(X[:, i], Xdec[:, i])
        elif cat_size is not None and i in cat_begin:
            ind = cat_begin.index(i)
            b = cat_begin[ind]
            s = cat_size[ind]
            scores += _cross_entropy(X[:, b:b+s], Xdec[:, b:b+s])
        elif i in def_inds:
            if metric == 'mse':
                scores += np.sum((X[:, i:i+1] - Xdec[:, i:i+1]) ** 2, axis=1)
            else:
                scores += _cross_entropy(X[:, i], Xdec[:, i])

    ae_scores = ae.score_samples(X)
    assert_array_almost_equal(scores, ae_scores, decimal=5)

    score = np.mean(scores)
    ae_score = ae.score(X)
    assert_almost_equal(score, ae_score, decimal=5)

    _LOGGER.warning("\ntest info:\n    ae: %s\n    sparse format: %s\n"
                    "    score: %g\n    X[10]: %s\n    Xdec[10]: %s",
                    str(ae), sparse_type, ae_score,
                    pprint.pformat(list(X[10]), compact=True),
                    pprint.pformat(list(Xdec[10]), compact=True))

    assert ae_score < max_score, ("Autoencoder should have a score "
                                  "less than %f for the iris features." %
                                  max_score)

    return ae_score


MSE_MAX_SCORE = 0.1


def test_mse_single_hidden_unit():
    """Test the MSE metric w/ a single hidden unit."""
    _check_ae(MSE_MAX_SCORE, hidden_units=(1,))


def test_mse_multiple_hidden_units():
    """Test the MSE metric w/ 2 hidden units."""
    _check_ae(MSE_MAX_SCORE, hidden_units=(2,))


def test_mse_multiple_layers():
    """Test the MSE metric w/ layers (3, 2)."""
    _check_ae(MSE_MAX_SCORE, hidden_units=(3, 2))


def test_mse_dropout():
    """Test the MSE metric w/ dropout."""
    ae_score_dropout = _check_ae(MSE_MAX_SCORE,
                                 hidden_units=(20, 20, 10, 10, 2),
                                 dropout=0.05,
                                 learning_rate=1e-3)

    ae_score_nodropout = _check_ae(MSE_MAX_SCORE,
                                   hidden_units=(20, 20, 10, 10, 2),
                                   dropout=0.0,
                                   learning_rate=1e-3)

    assert ae_score_nodropout < ae_score_dropout, ("MSE with dropout should "
                                                   "be more than MSE with no "
                                                   " dropout!")


def test_cross_entropy_softmax_single_unit():
    """Test the cross-entropy metric w/ softmax and a single hidden unit."""

    # Use the iris features.
    X = iris.data
    X = MinMaxScaler().fit_transform(X)

    # Make some columns normalized to unity.
    X = X / np.sum(X, axis=1)[:, np.newaxis]

    for i in range(2):
        if i == 1:
            bins = [0.0, np.median(X[:, 0]), 1.1]
            X[:, 0] = np.digitize(X[:, 0], bins) - 1.0
            X[:, 1:] = X[:, 1:] / np.sum(X[:, 1:], axis=1)[:, np.newaxis]
            binary_indices = [0]
        else:
            binary_indices = None

        ae = Autoencoder(hidden_units=(1,),
                         n_epochs=5000,
                         random_state=4556,
                         learning_rate=1e-1,
                         dropout=0.0,
                         metric='cross-entropy',
                         output_activation=tf.nn.softmax,
                         binary_indices=binary_indices)

        Xenc = ae.fit_transform(X)
        Xdec = ae.inverse_transform(Xenc)

        assert Xenc.shape == (X.shape[0], 1), ("Encoded iris data is not the "
                                               "right shape!")

        assert Xdec.shape == X.shape, ("Decoded iris data is not the right "
                                       "shape!")

        if i == 1:
            # Softmax stuff should come back out normalized.
            assert_array_almost_equal(np.sum(Xdec[:, 1:], axis=1), 1.0)

            # Compute and test the scores.
            scores = _cross_entropy(X[:, 1:], Xdec[:, 1:])
            scores += _cross_entropy(X[:, 0], Xdec[:, 0])

            ae_scores = ae.score_samples(X)
            assert_array_almost_equal(scores, ae_scores, decimal=5)
        else:
            # Softmax stuff should come back out normalized.
            assert_array_almost_equal(np.sum(Xdec, axis=1), 1.0)

            # Compute and test the scores.
            scores = _cross_entropy(X, Xdec)
            ae_scores = ae.score_samples(X)
            assert_array_almost_equal(scores, ae_scores, decimal=5)

        score = np.mean(scores)
        ae_score = ae.score(X)
        assert_almost_equal(score, ae_score, decimal=5)

        _LOGGER.warning("\ntest info:\n    ae: %s\n"
                        "    score: %g\n    X[10]: %s\n    Xdec[10]: %s",
                        str(ae), ae_score,
                        pprint.pformat(list(X[10]), compact=True),
                        pprint.pformat(list(Xdec[10]), compact=True))

        assert ae_score < 2.5, ("Autoencoder should have a score "
                                "less than 2.5 for the iris features.")


def test_cross_entropy_single_hidden_unit():
    """Test the cross-entropy metric w/ a single hidden unit."""
    _check_ae(1.2,
              hidden_units=(1,),
              bin_inds=range(4),
              metric='cross-entropy')


def test_cross_entropy_or_binary():
    """Test to make sure binary indices or cross-entropy metric
    works in all cases."""
    # Use a variety of cases here.
    scores = []
    for binary_indices in [-1, [0], [1], [2], [2, 0], [2, 3],
                           [0, 2], [0, 1, 2, 3]]:
        scores.append(_check_ae(
            1.5,
            n_epochs=1000,
            bin_inds=range(4),
            metric='cross-entropy',
            bin_inds_to_use=binary_indices))

    # All scores should be equal.
    for score in scores:
        assert_almost_equal(score, scores[0], decimal=5)


def test_cat_single_hidden_unit():
    """Test categorical metric w/ a single hidden unit."""
    _check_ae(3.0, hidden_units=(1,), cat_inds=range(4))


def test_sparse_inputs():
    """Make sure sparse inputs work properly."""
    scores = []
    for sparse_type in [None, 'csr', 'bsr', 'coo', 'csc', 'dok', 'lil']:
        scores.append(_check_ae(
            5.0,
            sparse_type=sparse_type,
            cat_inds=range(4),
            n_epochs=1000))

    # All scores should be equal.
    for score in scores:
        assert_almost_equal(score, scores[0])


def test_cross_entropy_mse_single_hidden_unit():
    """Test cross-entropy + MSE metric w/ a single hidden unit."""
    _check_ae(0.2, hidden_units=(1,), bin_inds=[2, 3])


def test_cat_mse_single_hidden_unit():
    """Test categorical + MSE metric w/ a single hidden unit."""
    _check_ae(1.1, hidden_units=(1,), cat_inds=[2, 3])


def test_cat_cross_entropy_single_hidden_unit():
    """Test categorical + cross-entropy metric w/ a single hidden unit."""
    CAT_CE_MAX_SCORE = 1.7
    score_noinds = _check_ae(
        CAT_CE_MAX_SCORE,
        hidden_units=(1,),
        cat_inds=[2, 3],
        bin_inds=[0, 1],
        bin_inds_to_use=-1,
        metric='cross-entropy')
    score_inds = _check_ae(
        CAT_CE_MAX_SCORE,
        hidden_units=(1,),
        cat_inds=[2, 3],
        bin_inds=[0, 1],
        metric='mse')
    assert_almost_equal(score_noinds, score_inds)


def test_cat_cross_entropy_mse_single_hidden_unit():
    """Test categorical + cross-entropy + MSE metric w/ a single
    hidden unit."""
    _check_ae(1.0,
              hidden_units=(1,),
              cat_inds=[2, 3],
              bin_inds=[0])
