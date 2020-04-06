"""
Tests for the Autoencoder.
"""

import logging
import pprint
from io import BytesIO
import pickle
import sys
from unittest import mock

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

# Learning rate for most tests, changed in some places for convergence.
DEFAULT_LEARNING_RATE = 2e-3


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


class AutoencoderManyEpochs(Autoencoder):
    """Increase default model capacity and number of epochs so that
    the default Estimator will pass scikit-learn's sanity checks.
    """
    def __init__(self, hidden_units=(256,), batch_size=128, n_epochs=300,
                 keep_prob=1.0, hidden_activation=tf.nn.relu,
                 encoding_activation=None,
                 output_activation=None, random_state=None,
                 learning_rate=1e-3, loss='mse', sigmoid_indices=None,
                 softmax_indices=None):
        super(AutoencoderManyEpochs, self).__init__(
            hidden_units=hidden_units, batch_size=batch_size,
            n_epochs=n_epochs, keep_prob=keep_prob,
            hidden_activation=hidden_activation,
            encoding_activation=encoding_activation,
            output_activation=output_activation, random_state=random_state,
            learning_rate=learning_rate, loss=loss,
            sigmoid_indices=sigmoid_indices, softmax_indices=softmax_indices,
        )


# The subset invariance part of check_estimator seems to fail due to
# small numerical differences (e.g., 1e-6). The test failure is difficult to
# replicate outside of travis, but I was able to get the test to fail locally
# by changing atol in sklearn.utils.check_methods_subset_invariance from 1e-7
# to 1e-10. This simply skips that part of check_estimator.
@mock.patch('sklearn.utils.estimator_checks.check_methods_subset_invariance')
def test_check_estimator(mock_check_methods_subset_invariance):
    """Check adherence to Estimator API."""
    if sys.version_info.major == 3 and sys.version_info.minor == 7:
        # Starting in Tensorflow 1.14 and Python 3.7, there's one module
        # with a `0` in the __warningregistry__. Scikit-learn tries to clear
        # this dictionary in its tests.
        name = 'tensorboard.compat.tensorflow_stub.pywrap_tensorflow'
        with mock.patch.object(sys.modules[name], '__warningregistry__', {}):
            check_estimator(AutoencoderManyEpochs)
    else:
        check_estimator(AutoencoderManyEpochs)


def test_persistence():
    """Make sure we can pickle it."""
    X = iris.data  # Use the iris features.
    X = MinMaxScaler().fit_transform(X)

    ae = Autoencoder(hidden_units=(1,),
                     n_epochs=1000,
                     random_state=4556,
                     learning_rate=1e-2,
                     keep_prob=1.0)
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
                      keep_prob=1.0)
    Xenc1 = ae1.fit_transform(X)

    ae2 = Autoencoder(hidden_units=(1,),
                      n_epochs=1000,
                      random_state=4556,
                      learning_rate=1e-2,
                      keep_prob=1.0)
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
                     keep_prob=1.0,
                     loss='cross-entropy',
                     output_activation=tf.nn.sigmoid)
    ae.fit(X)
    assert ae.input_layer_size_ == 4, ("Input layer is the wrong size for "
                                       "the Autoencoder!")

    X_small = X[:, 0:-1]
    assert X_small.shape != X.shape, "Test data for refitting does not work!"
    ae.fit(X_small)
    assert ae.input_layer_size_ == 3, ("Input layer is the wrong size for "
                                       "the Autoencoder!")


def test_errors_overlapping_sigmoid_softmax_indixes():
    """Make overlapping sigmoid and softmax indices raises an error."""

    # This data will not actually be fit.
    # I am just using it to call the `fit` method.
    X = np.ones((1000, 4))

    ae = Autoencoder(loss='blah',
                     sigmoid_indices=[0],
                     softmax_indices=[[0, 2]])
    with pytest.raises(ValueError) as e:
        ae.fit(X)
        assert "Sigmoid indices and softmax indices" in str(e), (
            "Wrong error raised for overlapping sigmoid and softmax indices")


def test_errors_unallowed_loss():
    """Make sure unallowed losses cause an error."""

    # This data will not actually be fit.
    # I am just using it to call the `fit` method.
    X = np.ones((1000, 4))

    # There are two code paths for this test. One for all features
    # using the default loss and one for a mix of losses.

    # All features use the default loss.
    ae = Autoencoder(loss='blah')
    with pytest.raises(ValueError) as e:
        ae.fit(X)
        assert "Loss 'blah'" in str(e), (
            "Wrong error raised for testing unallowed losses!")

    # Not all features use the default loss.
    ae = Autoencoder(loss='blah', sigmoid_indices=[0])
    with pytest.raises(ValueError) as e:
        ae.fit(X)
        assert "Loss 'blah'" in str(e), (
            "Wrong error raised for testing unallowed losses!")


def test_monitor_ae():
    """Test the monitor keyword."""
    # Use the iris features.
    X = iris.data
    X = MinMaxScaler().fit_transform(X)

    ae = Autoencoder(hidden_units=(3, 2,),
                     n_epochs=7500,
                     random_state=4556,
                     learning_rate=DEFAULT_LEARNING_RATE,
                     keep_prob=1.0,
                     hidden_activation=tf.nn.sigmoid,
                     encoding_activation=tf.nn.sigmoid,
                     output_activation=tf.nn.sigmoid)

    def _monitor(epoch, est, stats):
        assert epoch <= 1000, "The autoencoder has been running too long!"
        if stats['loss'] < 0.2:
            assert epoch > 10, "The autoencoder returned too soon!"
            return True
        else:
            return False
    ae.fit(X, monitor=_monitor)


def test_errors_loss_output_activation():
    """Make sure cross-entropy loss with activations not equal to
    `tensorflow.nn.sigmoid` or `tensorflow.nn.softmax` fails."""

    # This data will not actually be fit.
    # I am just using it to call the `fit` method.
    X = np.ones((1000, 4))

    # There are two code paths for this test. One for all features
    # using the default loss and one for a mix of losses.

    # All features use the default loss.
    ae = Autoencoder(loss='cross-entropy', output_activation=tf.exp)
    with pytest.raises(ValueError) as e:
        ae.fit(X)
        assert "'cross-entropy' loss!" in str(e), (
            "Wrong error raised for testing 'cross-entropy' loss with "
            "output activation that is not allowed for all features!")

    # Not all features use the default loss.
    ae = Autoencoder(loss='cross-entropy',
                     output_activation=tf.exp,
                     sigmoid_indices=[0])
    with pytest.raises(ValueError) as e:
        ae.fit(X)
        assert "'cross-entropy' loss!" in str(e), (
            "Wrong error raised for testing 'cross-entropy' loss with "
            "output activation that is not allowed for a subset of features!")


def test_mse_sigmoid_activations():
    """Test the MSE loss w/ sigmoid activation."""
    # Use the iris features.
    X = iris.data
    X = MinMaxScaler().fit_transform(X)

    ae = Autoencoder(hidden_units=(3, 2,),
                     n_epochs=7500,
                     random_state=4556,
                     learning_rate=DEFAULT_LEARNING_RATE,
                     keep_prob=1.0,
                     hidden_activation=tf.nn.sigmoid,
                     encoding_activation=tf.nn.sigmoid,
                     output_activation=tf.nn.sigmoid)
    Xenc = ae.fit_transform(X)
    Xdec = ae.inverse_transform(Xenc)

    assert Xenc.shape == (X.shape[0], 2), ("Encoded iris data "
                                           "is not the right"
                                           " shape!")

    assert Xdec.shape == X.shape, ("Decoded iris data is not the right "
                                   "shape!")

    # Compute and test the scores.
    scores = 0.0
    for i in range(X.shape[1]):
        scores += np.sum((X[:, i:i+1] - Xdec[:, i:i+1]) ** 2, axis=1)

    ae_scores = ae.score_samples(X)
    assert_array_almost_equal(scores, ae_scores, decimal=5)

    score = np.mean(scores)
    ae_score = ae.score(X)
    assert_almost_equal(score, ae_score, decimal=5)

    max_score = 0.1
    _LOGGER.warning("\ntest info:\n    ae: %s\n"
                    "    score: %g\n    X[10]: %s\n    Xdec[10]: %s",
                    str(ae), ae_score,
                    pprint.pformat(list(X[10])),
                    pprint.pformat(list(Xdec[10])))

    assert ae_score < max_score, ("Autoencoder should have a score "
                                  "less than %f for the iris features." %
                                  max_score)


def test_sigmoid_softmax_cross_entropy_loss():
    """Test the cross-entropy loss w/ softmax and sigmoid."""

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

        ae = Autoencoder(hidden_units=(2,),
                         n_epochs=7500,
                         random_state=4556,
                         learning_rate=DEFAULT_LEARNING_RATE,
                         keep_prob=1.0,
                         loss='cross-entropy',
                         output_activation=tf.nn.softmax,
                         sigmoid_indices=binary_indices,
                         hidden_activation=tf.nn.relu,
                         encoding_activation=None)

        Xenc = ae.fit_transform(X)
        Xdec = ae.inverse_transform(Xenc)

        assert Xenc.shape == (X.shape[0], 2), ("Encoded iris data is not the "
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
                        pprint.pformat(list(X[10])),
                        pprint.pformat(list(Xdec[10])))

        assert ae_score < 2.5, ("Autoencoder should have a score "
                                "less than 2.5 for the iris features.")


def _check_ae(max_score,
              hidden_units=(1,),
              keep_prob=1.0,
              learning_rate=None,
              sparse_type=None,
              bin_inds=None,
              bin_inds_to_use=None,
              cat_inds=None,
              n_epochs=7500,
              loss='mse'):
    """Helper function for testing the Autoencoder.

    This function does in order:

    1. Loads the Iris data.
    2. Converts columns to either binary (bin_inds) or categorical (cat_inds).
        Binary stuff is sent as sigmoid indices and categorical stuff is sent
        as softmax indices.
    3. Converts the data to a sparse type (sparse_type).
    4. Builds and trains the autoencoder (learning_rate, n_epochs, dropout,
        hidden_units, loss, bin_inds_to_use)
    5. Tests the outputs (max_score).

    The `bin_inds_to_use` parameter in particular specifies which columns
    the autoencoder is explicitly told are sigmoid values.
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

    # For sigmoid runs, we can set the loss or use sigmoid_indices.
    # This if handles those cases.
    if bin_inds_to_use is None and binary_inds is not None:
        bin_inds_to_use = binary_inds
    elif bin_inds_to_use == -1:
        bin_inds_to_use = None

    if loss == 'cross-entropy':
        output_activation = tf.nn.sigmoid
    else:
        output_activation = None

    if learning_rate is None:
        learning_rate = DEFAULT_LEARNING_RATE

    ae = Autoencoder(hidden_units=hidden_units,
                     n_epochs=n_epochs,
                     random_state=4556,
                     learning_rate=learning_rate,
                     keep_prob=keep_prob,
                     loss=loss,
                     sigmoid_indices=bin_inds_to_use,
                     softmax_indices=cat_indices,
                     hidden_activation=tf.nn.relu,
                     encoding_activation=None,
                     output_activation=output_activation)

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
            if loss == 'mse':
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
                    pprint.pformat(list(X[10])),
                    pprint.pformat(list(Xdec[10])))

    assert ae_score < max_score, ("Autoencoder should have a score "
                                  "less than %f for the iris features." %
                                  max_score)

    return ae_score


MSE_MAX_SCORE = 0.1


def test_mse_single_hidden_unit():
    """Test the MSE loss w/ a single hidden unit."""
    _check_ae(MSE_MAX_SCORE, hidden_units=(1,))


def test_mse_multiple_hidden_units():
    """Test the MSE loss w/ 2 hidden units."""
    _check_ae(MSE_MAX_SCORE, hidden_units=(2,))


def test_mse_multiple_layers():
    """Test the MSE loss w/ layers (3, 2)."""
    _check_ae(MSE_MAX_SCORE, hidden_units=(3, 2))


def test_mse_dropout():
    """Test the MSE loss w/ dropout."""
    ae_score_dropout = _check_ae(MSE_MAX_SCORE,
                                 hidden_units=(20, 20, 10, 10, 2),
                                 keep_prob=0.95,
                                 learning_rate=1e-3)

    ae_score_nodropout = _check_ae(MSE_MAX_SCORE,
                                   hidden_units=(20, 20, 10, 10, 2),
                                   keep_prob=1.0,
                                   learning_rate=1e-3)

    assert ae_score_nodropout < ae_score_dropout, ("MSE with dropout should "
                                                   "be more than MSE with no "
                                                   " dropout!")


def test_sigmoid():
    """Test sigmoid indices."""
    _check_ae(1.2,
              hidden_units=(2,),
              bin_inds=range(4),
              loss='cross-entropy')


def test_cross_entropy_or_sigmoid():
    """Test to make sure sigmoid indices or cross-entropy loss
    works in all cases."""
    # Use a variety of cases here.
    scores = []
    for binary_indices in [-1, [0], [1], [2], [2, 0], [2, 3],
                           [0, 2], [0, 1, 2, 3]]:
        scores.append(_check_ae(
            1.5,
            n_epochs=1000,
            bin_inds=range(4),
            loss='cross-entropy',
            bin_inds_to_use=binary_indices))

    # All scores should be equal.
    for score in scores:
        assert_almost_equal(score, scores[0], decimal=5)


def test_softmax():
    """Test softmax indices."""
    _check_ae(3.0, hidden_units=(2,), cat_inds=range(4))


def test_sparse_inputs():
    """Make sure sparse inputs work properly."""
    scores = []
    for sparse_type in [None, 'csr', 'bsr', 'coo', 'csc', 'dok', 'lil']:
        scores.append(_check_ae(
            5.0,
            hidden_units=(2,),
            sparse_type=sparse_type,
            cat_inds=range(4),
            n_epochs=1000,
            learning_rate=5e-3  # Using a higher learning rate
                                # here for convergence.
            ))

    # All scores should be equal.
    for score in scores:
        assert_almost_equal(score, scores[0])


def test_sigmoid_mse():
    """Test sigmoid indices + MSE loss."""
    _check_ae(0.2, hidden_units=(2,), bin_inds=[2, 3])


def test_softmax_mse():
    """Test softmax indices + MSE loss."""
    _check_ae(1.1, hidden_units=(2,), cat_inds=[2, 3])


def test_softmax_sigmoid():
    """Test softmax + sigmoid indices."""
    CAT_CE_MAX_SCORE = 1.7
    score_noinds = _check_ae(
        CAT_CE_MAX_SCORE,
        hidden_units=(2,),
        cat_inds=[2, 3],
        bin_inds=[0, 1],
        bin_inds_to_use=-1,
        loss='cross-entropy')
    score_inds = _check_ae(
        CAT_CE_MAX_SCORE,
        hidden_units=(2,),
        cat_inds=[2, 3],
        bin_inds=[0, 1],
        loss='mse')
    assert_almost_equal(score_noinds, score_inds)


def test_softmax_sigmoid_mse():
    """Test softmax + sigmoid indices + MSE loss."""
    _check_ae(1.0,
              hidden_units=(2,),
              cat_inds=[2, 3],
              bin_inds=[0])
