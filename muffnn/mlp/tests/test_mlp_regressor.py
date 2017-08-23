"""
Tests for MLP Regressor
"""
from __future__ import print_function
from __future__ import division

import numpy as np
from sklearn.utils.testing import \
    assert_equal, assert_array_almost_equal
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.datasets import load_diabetes
from sklearn.utils.estimator_checks import check_estimator
from tensorflow import nn

from muffnn import MLPRegressor


# The defaults kwargs don't work for tiny datasets like those in these tests.
KWARGS = {"random_state": 0, "n_epochs": 1000, "batch_size": 1,
          "hidden_units": ()}


# toy dataset where Y = x[0] -2 * x[1] + 2 + err
X = np.array([[-1, 0], [-2, 1], [1, 1], [2, 0], [-2, 0], [0, 2]],
             dtype=np.float32)
X_sp = sp.csr_matrix(X)
Y = X[:, 0] - 2 * X[:, 1] + 2 + \
    np.random.RandomState(42).randn(X.shape[0]) * 0.01


def check_predictions(est, X, y):
    """Check that the model is able to fit the regression training data.

    based on
    https://github.com/scikit-learn/scikit-learn/blob/af171b84bd3fb82eed4569aa0d1f976264ffae84/sklearn/linear_model/tests/test_logistic.py#L38
    """
    n_samples = len(y)
    preds = est.fit(X, y).predict(X)
    assert_equal(preds.shape, (n_samples,))
    assert_array_almost_equal(preds, y, decimal=1)


# Make a subclass that has its default number of epochs high enough not to fail
# the toy example tests that have only a handful of examples.
class MLPRegressorManyEpochs(MLPRegressor):
    def __init__(self, hidden_units=(256,), batch_size=64,
                 keep_prob=1.0, activation=nn.relu, init_scale=0.1):
        super(MLPRegressorManyEpochs, self).__init__(
            hidden_units=hidden_units, batch_size=batch_size,
            n_epochs=100, keep_prob=keep_prob,
            activation=activation, init_scale=init_scale,
            random_state=42)


def test_check_estimator():
    """Check adherence to Estimator API."""
    check_estimator(MLPRegressorManyEpochs)


def test_predict():
    """Test binary classification."""
    check_predictions(MLPRegressor(**KWARGS), X, Y)
    check_predictions(MLPRegressor(**KWARGS), X_sp, Y)


def test_replicability():
    """Make sure running fit twice in a row finds the same parameters."""
    diabetes = load_diabetes()
    X_diabetes, y_diabetes = diabetes.data, diabetes.target
    ind = np.arange(X_diabetes.shape[0])
    rng = np.random.RandomState(0)
    rng.shuffle(ind)
    X_diabetes, y_diabetes = X_diabetes[ind], y_diabetes[ind]

    clf = MLPRegressor(keep_prob=0.9, random_state=42, n_epochs=100)
    target = y_diabetes
    # Just predict on the training set, for simplicity.
    pred1 = clf.fit(X_diabetes, target).predict(X_diabetes)
    pred2 = clf.fit(X_diabetes, target).predict(X_diabetes)
    assert_array_almost_equal(pred1, pred2)


def test_partial_fit():
    data = load_diabetes()
    clf = MLPRegressor(n_epochs=1)

    X, y = data['data'], data['target']

    for _ in range(30):
        clf.partial_fit(X, y)

    y_pred = clf.predict(X)
    assert pearsonr(y_pred, y)[0] > 0.5
