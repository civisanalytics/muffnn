"""
Tests for the Autoencoder.
"""

import logging
import pprint
from io import BytesIO
import pickle

import numpy as np
import scipy.special
import scipy.sparse as sp
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


def _check_ae(max_score,
              hidden_units=(1,),
              dropout=0.0,
              learning_rate=1e-1,
              sparse_type=None,
              disc_inds=None,
              disc_inds_to_use=None,
              cat_inds=None,
              n_epochs=5000,
              metric='mse'):
    # Use the iris features.
    X = iris.data
    X = MinMaxScaler().fit_transform(X)

    # Make some columns discrete or one-hot encoded.
    cat_size = []
    cat_begin = []
    dis_inds = []
    keep_cols = []
    def_inds = []
    num_cols = 0
    for i in range(X.shape[1]):
        if disc_inds is not None and i in disc_inds:
            bins = [0.0, np.median(X[:, i]), 1.1]
            keep_cols.append((np.digitize(X[:, i], bins) - 1.0)[:, np.newaxis])
            dis_inds.append(num_cols)
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
        cat_size = None
        cat_begin = None

    if len(dis_inds) == 0:
        dis_inds = None

    if sparse_type is not None:
        X = getattr(sp, sparse_type + '_matrix')(X)

    # For sigmoid metric runs, we can set the metric or use discrete_indices.
    # This if handles those cases.
    if disc_inds_to_use is None and dis_inds is not None:
        disc_inds_to_use = dis_inds
    elif disc_inds_to_use == -1:
        disc_inds_to_use = None

    ae = Autoencoder(hidden_units=hidden_units,
                     n_epochs=n_epochs,
                     random_state=4556,
                     learning_rate=learning_rate,
                     dropout=dropout,
                     metric=metric,
                     discrete_indices=disc_inds_to_use,
                     categorical_begin=cat_begin,
                     categorical_size=cat_size)

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
                np.sum(Xdec[:, begin: begin + size], axis=1), 1.0)

    # Compute and test the scores.
    scores = 0.0
    for i in range(X.shape[1]):
        if dis_inds is not None and i in dis_inds:
            scores += sigmoid_cross_entropy(X[:, i:i+1], Xdec[:, i:i+1])
        elif cat_size is not None and i in cat_begin:
            ind = cat_begin.index(i)
            b = cat_begin[ind]
            s = cat_size[ind]
            scores += softmax_cross_entropy(X[:, b:b+s], Xdec[:, b:b+s])
        elif i in def_inds:
            if metric == 'mse':
                scores += np.sum((X[:, i:i+1] - Xdec[:, i:i+1]) ** 2, axis=1)
            else:
                scores += sigmoid_cross_entropy(X[:, i:i+1], Xdec[:, i:i+1])

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


def test_cross_entropy_single_hidden_unit():
    """Test the cross-entropy metric w/ a single hidden unit."""
    _check_ae(1.2,
              hidden_units=(1,),
              disc_inds=range(4),
              metric='cross-entropy')


def test_cross_entropy_or_discrete():
    """Test to make sure discrete indices or cross-entropy metric
    works in all cases."""
    # Use a variety of cases here.
    scores = []
    for discrete_indices in [-1, [0], [1], [2], [2, 0], [2, 3],
                             [0, 2], [0, 1, 2, 3]]:
        scores.append(_check_ae(
            1.5,
            n_epochs=1000,
            disc_inds=range(4),
            metric='cross-entropy',
            disc_inds_to_use=discrete_indices))

    # All scores should be equal.
    for score in scores:
        assert_almost_equal(score, scores[0])


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
    _check_ae(0.2, hidden_units=(1,), disc_inds=[2, 3])


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
        disc_inds=[0, 1],
        disc_inds_to_use=-1,
        metric='cross-entropy')
    score_inds = _check_ae(
        CAT_CE_MAX_SCORE,
        hidden_units=(1,),
        cat_inds=[2, 3],
        disc_inds=[0, 1],
        metric='mse')
    assert_almost_equal(score_noinds, score_inds)


def test_cat_cross_entropy_mse_single_hidden_unit():
    """Test categorical + cross-entropy + MSE metric w/ a single
    hidden unit."""
    _check_ae(1.0,
              hidden_units=(1,),
              cat_inds=[2, 3],
              disc_inds=[0])
