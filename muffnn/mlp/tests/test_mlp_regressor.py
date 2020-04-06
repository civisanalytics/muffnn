"""
Tests for MLP Regressor
"""

import sys
from unittest import mock

import numpy as np
import pytest
from sklearn.utils.testing import \
    assert_equal, assert_array_almost_equal
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.datasets import load_diabetes, make_regression
from sklearn.utils.estimator_checks import check_estimator
from tensorflow import nn

from muffnn import MLPRegressor
from muffnn.mlp.tests.util import assert_sample_weights_work


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


def test_sample_weight():
    """Ensure we handle sample weights for regression problems."""
    assert_sample_weights_work(
        make_regression,
        {'n_samples': 3000},
        lambda: MLPRegressor(n_epochs=30, random_state=42,
                             keep_prob=0.8, hidden_units=(128,))
    )


# Make a subclass that has no `solver` parameter. The scikit-learn
# `check_estimator` has a check which fails with a class as a default.
class MLPRegressorFewerParams(MLPRegressor):
    def __init__(self, hidden_units=(256,), batch_size=64, n_epochs=5,
                 keep_prob=1.0, activation=nn.relu,
                 random_state=None):
        super(MLPRegressorFewerParams, self).__init__(
            hidden_units=hidden_units, batch_size=batch_size,
            n_epochs=n_epochs, keep_prob=keep_prob,
            activation=activation,
            random_state=random_state)


def test_check_estimator():
    """Check adherence to Estimator API."""
    if sys.version_info.major == 3 and sys.version_info.minor == 7:
        # Starting in Tensorflow 1.14 and Python 3.7, there's one module
        # with a `0` in the __warningregistry__. Scikit-learn tries to clear
        # this dictionary in its tests.
        name = 'tensorboard.compat.tensorflow_stub.pywrap_tensorflow'
        with mock.patch.object(sys.modules[name], '__warningregistry__', {}):
            check_estimator(MLPRegressorFewerParams)
    else:
        check_estimator(MLPRegressorFewerParams)


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


def test_embedding_default():
    # Make sure the embedding works by default.
    data = load_diabetes()
    X, y = data['data'], data['target']

    clf = MLPRegressor(n_epochs=1)
    clf.fit(X, y)

    assert clf.transform(X).shape[1] == 256


def test_embedding_no_layers():
    # Make sure the embedding works with no layers.
    data = load_diabetes()
    X, y = data['data'], data['target']

    clf = MLPRegressor(n_epochs=1, hidden_units=[])
    clf.fit(X, y)

    assert clf.transform(X).shape[1] == 1


def test_embedding_specific_layer():
    # Make sure the embedding works with no layers.
    data = load_diabetes()
    X, y = data['data'], data['target']

    clf = MLPRegressor(
        n_epochs=1,
        hidden_units=(256, 8, 256),
        transform_layer_index=1)
    clf.fit(X, y)

    assert clf.transform(X).shape[1] == 8


def test_prediction_gradient():
    """Test computation of prediction gradients."""
    mlp = MLPRegressor(n_epochs=100, random_state=42, hidden_units=(5,))
    X, y = make_regression(
        n_samples=1000, n_features=10, n_informative=1, shuffle=False)
    mlp.fit(X, y)
    grad = mlp.prediction_gradient(X)
    grad_means = grad.mean(axis=0)
    assert grad.shape == X.shape
    # Check that only the informative feature has a large gradient.
    assert np.abs(grad_means[0]) > 0.5
    for m in grad_means[1:]:
        assert np.abs(m) < 0.1

    # Raise an exception for sparse inputs, which are not yet supported.
    X_sp = sp.csr_matrix(X)
    mlp.fit(X_sp, y)
    with pytest.raises(NotImplementedError):
        mlp.prediction_gradient(X_sp)
