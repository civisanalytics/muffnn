"""
Tests for the FM Regressor

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
from scipy.stats import pearsonr
from sklearn.datasets import load_diabetes, make_regression
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.testing import assert_array_almost_equal, assert_equal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold

from muffnn import FMRegressor
from muffnn.fm.tests.util import assert_sample_weights_work

# toy dataset where Y = x[0] -2 * x[1] + 2 + err
diabetes = load_diabetes()
X = np.array([[-1, 0], [-2, 1], [1, 1], [2, 0], [-2, 0], [0, 2]],
             dtype=np.float32)
Xsp = sp.csr_matrix(X)
Y = X[:, 0] - 2 * X[:, 1] + 2 + \
    np.random.RandomState(42).randn(X.shape[0]) * 0.01

# The defaults kwargs don't work for tiny datasets like those in these tests.
KWARGS = {"random_state": 2, "solver": 'L-BFGS-B', "rank": 1}
SGD_KWARGS = {"random_state": 2,
              "rank": 1,
              "solver": 'L-BFGS-B',
              "n_epochs": 1000}


def check_predictions(est, X, y):
    """Check that the model is able to fit the regression training data.

    based on
    https://github.com/scikit-learn/scikit-learn/blob/af171b84bd3fb82eed4569aa0d1f976264ffae84/sklearn/linear_model/tests/test_logistic.py#L38
    """
    n_samples = len(y)
    preds = est.fit(X, y).predict(X)
    assert_equal(preds.shape, (n_samples,))
    assert_array_almost_equal(preds, y, decimal=1)


class FMRegressorLBFGSB(FMRegressor):
    def __init__(self):
        super(FMRegressorLBFGSB, self).__init__(
            rank=1, solver='L-BFGS-B', random_state=2)


def test_make_feed_dict():
    """Test that the feed dictionary works ok."""
    reg = FMRegressor()
    reg.is_sparse_ = False
    reg._y = 0
    reg._x = 1
    reg._sample_weight = 'sample_weight'

    output_size = 1

    reg._output_size = output_size
    fd = reg._make_feed_dict(np.array(X), np.array(Y))
    expected_keys = {0, 1, 'sample_weight'}
    assert set(fd.keys()) == expected_keys
    assert_array_almost_equal(fd[reg._y], Y)
    assert fd[reg._y].dtype == np.float32, (
        "Wrong data type for y w/ output_size = 1 in feed dict!")
    assert_array_almost_equal(fd[reg._x], X)
    assert fd[reg._x].dtype == np.float32, (
        "Wrong data dtype for X in feed dict!")


def test_make_feed_dict_sparse():
    """Test that the feed dictionary works ok for sparse inputs."""
    reg = FMRegressor()
    reg.is_sparse_ = True
    reg._y = 0
    reg._x_inds = 1
    reg._x_vals = 2
    reg._x_shape = 3
    reg._sample_weight = 'sample_weight'

    # changing this so test catches indexing errors
    X = [[-1, 0], [0, 1], [2, 3]]

    output_size = 1

    reg._output_size = output_size
    fd = reg._make_feed_dict(np.array(X), np.array(Y))
    assert_array_almost_equal(fd[reg._y], Y)
    if output_size == 1:
        assert fd[reg._y].dtype == np.float32, (
            "Wrong data type for y w/ output_size = 1 in feed dict!")
    else:
        assert fd[reg._y].dtype == np.int32, (
            "Wrong data type for y w/ output_size > 1 in feed dict!")

    # Sparse arrays for TF are in row-major sorted order.
    assert_array_almost_equal(
        fd[reg._x_inds], [[0, 0], [1, 1], [2, 0], [2, 1]])
    assert fd[reg._x_inds].dtype == np.int64, (
        "Wrong data type for sparse inds in feed dict!")
    assert_array_almost_equal(fd[reg._x_vals], [-1, 1, 2, 3])
    assert fd[reg._x_vals].dtype == np.float32, (
        "Wrong data type for sparse vals in feed dict!")
    assert_array_almost_equal(fd[reg._x_shape], [3, 2])
    assert fd[reg._x_shape].dtype == np.int64, (
        "Wrong data type for sparse shape in feed dict!")


def test_check_estimator():
    """Check adherence to Estimator API."""
    if sys.version_info.major == 3 and sys.version_info.minor == 7:
        # Starting in Tensorflow 1.14 and Python 3.7, there's one module
        # with a `0` in the __warningregistry__. Scikit-learn tries to clear
        # this dictionary in its tests.
        name = 'tensorboard.compat.tensorflow_stub.pywrap_tensorflow'
        with mock.patch.object(sys.modules[name], '__warningregistry__', {}):
            check_estimator(FMRegressorLBFGSB)
    else:
        check_estimator(FMRegressorLBFGSB)


def test_predict_lbfgsb():
    """Test regression w/ L-BFGS-B."""
    check_predictions(FMRegressor(**KWARGS), X, Y)


def test_predict_sgd():
    """Test regression w/ SGD."""
    check_predictions(FMRegressor(**SGD_KWARGS), X, Y)


def test_sparse_lbfgsb():
    """Test sparse matrix handling w/ L-BFGS-B."""
    check_predictions(FMRegressor(**KWARGS), Xsp, Y)


def test_sparse_sgd():
    """Test sparse matrix handling w/ SGD."""
    check_predictions(FMRegressor(**SGD_KWARGS), Xsp, Y)


def test_persistence():
    reg = FMRegressor(random_state=42)
    X_diabetes, y_diabetes = diabetes.data, diabetes.target
    ind = np.arange(X_diabetes.shape[0])
    rng = np.random.RandomState(0)
    rng.shuffle(ind)
    X_diabetes, y_diabetes = X_diabetes[ind], y_diabetes[ind]
    pred1 = reg.fit(X_diabetes, y_diabetes).predict(X_diabetes)
    b = BytesIO()
    pickle.dump(reg, b)
    reg2 = pickle.loads(b.getvalue())
    pred2 = reg2.predict(X_diabetes)
    assert_array_almost_equal(pred1, pred2)


def test_replicability():
    """Test that models can be pickled and reloaded."""
    reg = FMRegressor(random_state=42)
    X_diabetes, y_diabetes = diabetes.data, diabetes.target
    ind = np.arange(X_diabetes.shape[0])
    rng = np.random.RandomState(0)
    rng.shuffle(ind)
    X_diabetes, y_diabetes = X_diabetes[ind], y_diabetes[ind]

    # Just predict on the training set, for simplicity.
    pred1 = reg.fit(X_diabetes, y_diabetes).predict(X_diabetes)
    pred2 = reg.fit(X_diabetes, y_diabetes).predict(X_diabetes)
    assert_array_almost_equal(pred1, pred2)


def test_partial_fit():
    # FMClassifier tests don't need **KWARGS but otherwise we get garbage
    reg = FMRegressor(**KWARGS)
    X, y = diabetes.data, diabetes.target

    for _ in range(30):
        reg.partial_fit(X, y)

    y_pred = reg.predict(X)
    assert pearsonr(y_pred, y)[0] > 0.5


def test_multioutput():
    """Check that right sized array is return when doing one prediction."""
    reg = FMRegressor(random_state=2)

    X, y = diabetes.data, diabetes.target
    y = np.concatenate([y.reshape(-1, 1), y.reshape(-1, 1)], axis=1)

    with pytest.raises(ValueError):
        # not implemented!
        reg.fit(X, y)


def test_predict_one():
    """Check that right sized array is return when doing one prediction."""
    reg = FMRegressor(random_state=2)

    X, y = diabetes.data, diabetes.target

    reg.fit(X, y)
    p = reg.predict(X[0:1, :])
    assert p.shape == (1, )


def test_cross_val_predict():
    """Make sure it works in cross_val_predict."""

    Xt = StandardScaler().fit_transform(X)
    reg = FMRegressor(rank=2, solver='L-BFGS-B', random_state=4567).fit(Xt, Y)

    cv = KFold(n_splits=2, random_state=457, shuffle=True)
    y_oos = cross_val_predict(reg, Xt, Y, cv=cv, method='predict')
    p_r = pearsonr(Y, y_oos)[0]

    assert p_r >= 0.90, "Pearson R too low for fake data in cross_val_predict!"


def test_sample_weight():
    assert_sample_weights_work(
        make_regression,
        {'n_samples': 3000},
        # TF SGD does not work so well....
        lambda: FMRegressor(rank=2, solver='L-BFGS-B', random_state=4567)
    )
