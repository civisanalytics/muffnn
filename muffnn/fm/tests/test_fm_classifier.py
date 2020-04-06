"""
Tests for the FM classifier.

based in part on sklearn's logistic tests:
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/tests/test_logistic.py
"""
from io import BytesIO
import pickle
import sys
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp
try:
    from scipy.misc import logsumexp
except ImportError:
    from scipy.special import logsumexp
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model.tests.test_logistic import check_predictions
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold

from muffnn import FMClassifier
from muffnn.fm.tests.util import assert_sample_weights_work

iris = load_iris()
X = [[-1, 0], [0, 1], [1, 1]]
Xsp = sp.csr_matrix(X)
Y1 = [0, 1, 1]
Y2 = [2, 1, 0]

# The defaults kwargs don't work for tiny datasets like those in these tests.
KWARGS = {"random_state": 2, "solver": 'L-BFGS-B', "rank": 1}
SGD_KWARGS = {"random_state": 2,
              "rank": 1,
              "solver_kwargs": {"learning_rate": 1.0},
              "n_epochs": 1000}


class FMClassifierLBFGSB(FMClassifier):
    def __init__(self):
        super(FMClassifierLBFGSB, self).__init__(
            rank=1, solver='L-BFGS-B', random_state=2)

    def predict_proba(self, *args, **kwargs):
        res = super(FMClassifierLBFGSB, self).predict_proba(*args, **kwargs)
        res = res.astype(np.float64)
        res /= np.sum(res, axis=1).reshape(-1, 1)
        return res


def test_make_feed_dict():
    """Test that the feed dictionary works ok."""
    clf = FMClassifier()
    clf.is_sparse_ = False
    clf._y = 0
    clf._x = 1
    clf._sample_weight = "sample_weight"
    for output_size in [1, 2, 3]:
        clf._output_size = output_size
        fd = clf._make_feed_dict(np.array(X), np.array(Y1))
        assert_array_almost_equal(fd[clf._y], Y1)
        if output_size == 1:
            assert fd[clf._y].dtype == np.float32, (
                "Wrong data type for y w/ output_size = 1 in feed dict!")
        else:
            assert fd[clf._y].dtype == np.int32, (
                "Wrong data type for y w/ output_size > 1 in feed dict!")
        assert_array_almost_equal(fd[clf._x], X)
        assert fd[clf._x].dtype == np.float32, (
            "Wrong data dtype for X in feed dict!")


def test_make_feed_dict_sparse():
    """Test that the feed dictionary works ok for sparse inputs."""
    clf = FMClassifier()
    clf.is_sparse_ = True
    clf._y = 0
    clf._x_inds = 1
    clf._x_vals = 2
    clf._x_shape = 3
    clf._sample_weight = "sample_weight"

    # changing this so test catches indexing errors
    X = [[-1, 0], [0, 1], [2, 3]]

    for output_size in [1, 2, 3]:
        clf._output_size = output_size
        fd = clf._make_feed_dict(np.array(X), np.array(Y1))
        assert_array_almost_equal(fd[clf._y], Y1)
        if output_size == 1:
            assert fd[clf._y].dtype == np.float32, (
                "Wrong data type for y w/ output_size = 1 in feed dict!")
        else:
            assert fd[clf._y].dtype == np.int32, (
                "Wrong data type for y w/ output_size > 1 in feed dict!")

        # Sparse arrays for TF are in row-major sorted order.
        assert_array_almost_equal(
            fd[clf._x_inds], [[0, 0], [1, 1], [2, 0], [2, 1]])
        assert fd[clf._x_inds].dtype == np.int64, (
            "Wrong data type for sparse inds in feed dict!")
        assert_array_almost_equal(fd[clf._x_vals], [-1, 1, 2, 3])
        assert fd[clf._x_vals].dtype == np.float32, (
            "Wrong data type for sparse vals in feed dict!")
        assert_array_almost_equal(fd[clf._x_shape], [3, 2])
        assert fd[clf._x_shape].dtype == np.int64, (
            "Wrong data type for sparse shape in feed dict!")


def test_check_estimator():
    """Check adherence to Estimator API."""
    if sys.version_info.major == 3 and sys.version_info.minor == 7:
        # Starting in Tensorflow 1.14 and Python 3.7, there's one module
        # with a `0` in the __warningregistry__. Scikit-learn tries to clear
        # this dictionary in its tests.
        name = 'tensorboard.compat.tensorflow_stub.pywrap_tensorflow'
        with mock.patch.object(sys.modules[name], '__warningregistry__', {}):
            check_estimator(FMClassifierLBFGSB)
    else:
        check_estimator(FMClassifierLBFGSB)


def test_predict_lbfgsb():
    """Test classification w/ L-BFGS-B."""
    check_predictions(FMClassifier(**KWARGS), X, Y1)
    check_predictions(FMClassifier(**KWARGS), X, Y2)


def test_predict_sgd():
    """Test classification w/ SGD."""
    check_predictions(FMClassifier(**SGD_KWARGS), X, Y1)
    check_predictions(FMClassifier(**SGD_KWARGS), X, Y2)


def test_sparse_lbfgsb():
    """Test sparse matrix handling w/ L-BFGS-B."""
    check_predictions(FMClassifier(**KWARGS), Xsp, Y1)
    check_predictions(FMClassifier(**KWARGS), Xsp, Y2)


def test_sparse_sgd():
    """Test sparse matrix handling w/ SGD."""
    check_predictions(FMClassifier(**SGD_KWARGS), Xsp, Y1)
    check_predictions(FMClassifier(**SGD_KWARGS), Xsp, Y2)


def test_persistence():
    """Test that models can be pickled and reloaded."""
    clf = FMClassifier(random_state=42)
    target = iris.target_names[iris.target]
    clf.fit(iris.data, target)
    probs1 = clf.predict_proba(iris.data)
    b = BytesIO()
    pickle.dump(clf, b)
    clf2 = pickle.loads(b.getvalue())
    probs2 = clf2.predict_proba(iris.data)
    assert_array_almost_equal(probs1, probs2)


def test_replicability():
    clf = FMClassifier(random_state=42)
    target = iris.target_names[iris.target]
    probs1 = clf.fit(iris.data, target).predict_proba(iris.data)
    probs2 = clf.fit(iris.data, target).predict_proba(iris.data)
    assert_array_almost_equal(probs1, probs2)


def test_partial_fit():
    X, y = iris.data, iris.target

    # Predict on the training set and check that it (over)fit as expected.
    clf = FMClassifier(n_epochs=1)
    for _ in range(30):
        clf.partial_fit(X, y)
    y_pred = clf.predict(X)
    assert ((y_pred - y) ** 2).mean() < 10

    # Check that the classes argument works.
    clf = FMClassifier(n_epochs=1)
    clf.partial_fit(X[:10], y[:10], classes=np.unique(y))
    for _ in range(30):
        clf.partial_fit(X, y)

    # Check that using the classes argument wrong will fail.
    with pytest.raises(ValueError):
        clf = FMClassifier(n_epochs=1)
        clf.partial_fit(X, y, classes=np.array([0, 1]))


def test_predict_one():
    """Check that right sized array is return when doing one prediction."""

    X, y = iris.data, iris.target

    clf = FMClassifier(n_epochs=1)
    clf.fit(X, y)
    p = clf.predict_proba(X[0:1, :])
    assert p.shape == (1, 3)

    y_binary = (y == y[0]).astype(float)
    clf.fit(X, y_binary)
    p = clf.predict_proba(X[0:1, :])
    assert p.shape == (1, 2)


def test_refitting():
    """Check that fitting twice works (e.g., to make sure that fit-related
    variables are cleared appropriately when refitting)."""

    X, y = iris.data, iris.target

    clf = FMClassifier(n_epochs=1)
    clf.fit(X, y)
    assert np.array_equal(clf.classes_, np.unique(y))
    y_binary = (y == y[0]).astype(float)
    clf.fit(X, y_binary)
    assert np.array_equal(clf.classes_, np.unique(y_binary))


def test_cross_val_predict():
    """Make sure it works in cross_val_predict."""

    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)

    clf = FMClassifier(rank=2, solver='L-BFGS-B', random_state=4567).fit(X, y)

    cv = KFold(n_splits=4, random_state=457, shuffle=True)
    y_oos = cross_val_predict(clf, X, y, cv=cv, method='predict')
    acc = accuracy_score(y, y_oos)

    assert acc >= 0.90, "accuracy is too low for iris in cross_val_predict!"


def test_model_computations_sigmoid():
    """Make sure we did the FM comps right sigmoid."""
    clf = FMClassifier(random_state=42)
    clf.fit(X, Y1)
    y_proba = clf.predict_proba(X)

    v = clf._v.eval(session=clf._session)
    beta = clf._beta.eval(session=clf._session)
    beta0 = clf._beta0.eval(session=clf._session)

    logit_y_proba = beta0 + np.dot(X, beta)
    _X = np.array(X)
    for i in range(2):
        for j in range(i+1, 2):
            _vals = []
            for k in range(v.shape[2]):
                _vals.append(np.dot(v[:, i, k], v[:, j, k]))
            logit_y_proba += ((_X[:, i] * _X[:, j])[:, np.newaxis] *
                              np.column_stack(_vals))

    y_proba_test = 1.0 / (1.0 + np.exp(-logit_y_proba))
    if y_proba_test.ndim == 1:
        y_proba_test = y_proba_test[:, np.newaxis]

    if y_proba_test.shape[1] == 1:
        y_proba_test = np.column_stack([1.0 - y_proba_test[:, 0],
                                        y_proba_test[:, 0]])

    assert_array_almost_equal(y_proba, y_proba_test)


def test_model_computations_softmax():
    """Make sure we did the FM comps right for softmax."""
    clf = FMClassifier(random_state=42)
    clf.fit(X, Y2)
    y_proba = clf.predict_proba(X)

    v = clf._v.eval(session=clf._session)
    beta = clf._beta.eval(session=clf._session)
    beta0 = clf._beta0.eval(session=clf._session)

    logit_y_proba = beta0 + np.dot(X, beta)
    _X = np.array(X)
    for i in range(2):
        for j in range(i+1, 2):
            _vals = []
            for k in range(v.shape[2]):
                _vals.append(np.dot(v[:, i, k], v[:, j, k]))
            logit_y_proba += ((_X[:, i] * _X[:, j])[:, np.newaxis] *
                              np.column_stack(_vals))

    y_proba_test = np.exp(logit_y_proba -
                          logsumexp(logit_y_proba, axis=-1)[:, np.newaxis])
    assert_array_almost_equal(y_proba, y_proba_test)


def test_sample_weight():
    assert_sample_weights_work(
        make_classification,
        {'n_samples': 3000},
        # TF SGD does not work so well....
        lambda: FMClassifier(rank=2, solver='L-BFGS-B', random_state=4567)
    )
