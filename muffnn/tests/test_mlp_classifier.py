"""
Tests for MLP classifier

based in part on sklearn's logistic tests:
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/tests/test_logistic.py
"""

from io import BytesIO
import pickle
from unittest.mock import MagicMock

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import load_iris
from sklearn.linear_model.tests.test_logistic import check_predictions
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal
from tensorflow import nn

from muffnn import MLPClassifier


iris = load_iris()
X = [[-1, 0], [0, 1], [1, 1]]
X_sp = sp.csr_matrix(X)
Y1 = [0, 1, 1]
Y2 = [2, 1, 0]
Y_multilabel = np.array([[0, 1],
                         [1, 1],
                         [1, 0]])

# The defaults kwargs don't work for tiny datasets like those in these tests.
KWARGS = {"random_state": 0, "n_epochs": 100, "batch_size": 1}


# Make a subclass that has its default number of epochs high enough not to fail
# the toy example tests that have only a handful of examples.
class MLPClassifierManyEpochs(MLPClassifier):
    def __init__(self, hidden_units=(256,), batch_size=64,
                 dropout=None, activation=nn.relu, init_scale=0.1):
        super().__init__(hidden_units=hidden_units, batch_size=batch_size,
                         n_epochs=100, dropout=dropout, activation=activation,
                         init_scale=init_scale, random_state=42)


def test_check_estimator():
    """Check adherence to Estimator API."""
    check_estimator(MLPClassifierManyEpochs)


def check_multilabel_predictions(clf, X, y):
    predicted = clf.fit(X, y).predict(X)

    assert_equal(predicted.shape, y.shape)
    assert_array_equal(predicted, y)

    probabilities = clf.predict_proba(X)
    assert_equal(probabilities.shape, y.shape)
    assert_array_equal((probabilities >= 0.5).astype(np.int), predicted)
    assert np.sum(np.abs(probabilities.sum(axis=1) - 1)) > 0.01,\
        "multilabel classifier is outputting distributions"


def test_multilabel():
    check_multilabel_predictions(MLPClassifier(**KWARGS), X, Y_multilabel)
    check_multilabel_predictions(MLPClassifier(**KWARGS), X_sp, Y_multilabel)


def test_predict_2_classes():
    """Test binary classification."""
    check_predictions(MLPClassifier(**KWARGS), X, Y1)
    check_predictions(MLPClassifier(**KWARGS), X_sp, Y1)


def test_predict_3_classes():
    """Test multiclass classification."""
    check_predictions(MLPClassifier(**KWARGS), X, Y2)
    check_predictions(MLPClassifier(**KWARGS), X_sp, Y2)


def test_feed_dict():

    # Note that `tf.nn.dropout` scales the input up by t_dropout during
    # training so that scaling down during prediction isn't needed.
    # https://github.com/tensorflow/tensorflow/blob/2e152ecd67b3c5080f417260bc751e0c6bd7f1d3/tensorflow/python/ops/nn_ops.py#L1081-L1082.

    # Instantiate an MLP and mock out things that would be set in fit.
    mlp = MLPClassifier(dropout=0.5)
    mlp.input_targets_ = "input_targets"
    mlp._input_indices = "input_indices"
    mlp._input_values = "input_values"
    mlp._input_shape = "input_shape"
    mlp._dropout = "t_dropout"

    # sparse, targets given for training
    mlp.is_sparse_ = True
    X_sparse = MagicMock()
    X_sparse_dok = MagicMock()
    X_sparse.todok.return_value = X_sparse_dok
    X_sparse_dok.nnz = 100
    y = MagicMock()
    fd = mlp._make_feed_dict(X_sparse, y)
    expected_keys = {'input_shape', 'input_values', 'input_indices',
                     'input_targets', 't_dropout'}
    assert set(fd.keys()) == expected_keys
    assert fd['t_dropout'] == 0.5

    # sparse, no targets given
    fd = mlp._make_feed_dict(X_sparse)
    expected_keys = {'input_shape', 'input_values', 'input_indices',
                     't_dropout'}
    assert set(fd.keys()) == expected_keys
    assert fd['t_dropout'] == 1.0

    # dense, targets given for training
    mlp.is_sparse_ = False
    X_dense = MagicMock()
    fd = mlp._make_feed_dict(X_dense, y)
    expected_keys = {'input_values', 't_dropout', 'input_targets'}
    assert set(fd.keys()) == expected_keys
    assert fd['t_dropout'] == 0.5

    # dense, no targets given
    fd = mlp._make_feed_dict(X_dense)
    expected_keys = {'input_values', 't_dropout'}
    assert set(fd.keys()) == expected_keys
    assert fd['t_dropout'] == 1.0


def test_dropout():
    """Test binary classification."""
    # Check that predictions are deterministic.
    clf = MLPClassifier(dropout=0.5, **KWARGS)
    clf.fit(X_sp, Y1)
    y_pred1 = clf.predict_proba(X_sp)
    for _ in range(100):
        y_pred_i = clf.predict_proba(X_sp)
        assert_array_almost_equal(y_pred1, y_pred_i)

    check_predictions(
        MLPClassifier(dropout=0.5, **KWARGS), X, Y1)
    check_predictions(
        MLPClassifier(dropout=0.5, **KWARGS), X_sp, Y1)


def test_multiple_layers():
    for n_layers in range(3):
        clf = MLPClassifier(hidden_units=(8,) * n_layers, **KWARGS)
        target = iris.target_names[iris.target]
        clf.fit(iris.data, target)
        y_pred = clf.predict(iris.data)
        accuracy = accuracy_score(target, y_pred)
        # Just make sure the model doesn't crash and isn't terrible.
        assert accuracy > 0.9, \
            "low accuracy ({}) with {} layers".format(accuracy, n_layers)


def test_persistence():
    """Test that models can be pickled and reloaded."""
    clf = MLPClassifier(random_state=42)
    target = iris.target_names[iris.target]
    clf.fit(iris.data, target)
    probs1 = clf.predict_proba(iris.data)
    b = BytesIO()
    pickle.dump(clf, b)
    clf2 = pickle.loads(b.getvalue())
    probs2 = clf2.predict_proba(iris.data)
    assert_array_almost_equal(probs1, probs2)


def test_replicability():
    clf = MLPClassifier(dropout=0.5, random_state=42)
    target = iris.target_names[iris.target]
    probs1 = clf.fit(iris.data, target).predict_proba(iris.data)
    probs2 = clf.fit(iris.data, target).predict_proba(iris.data)
    assert_array_almost_equal(probs1, probs2)


def test_partial_fit():
    X, y = iris.data, iris.target

    # Predict on the training set and check that it (over)fit as expected.
    clf = MLPClassifier(n_epochs=1)
    for _ in range(30):
        clf.partial_fit(X, y)
    y_pred = clf.predict(X)
    assert ((y_pred - y) ** 2).mean() < 10

    # Check that the classes argument works.
    clf = MLPClassifier(n_epochs=1)
    clf.partial_fit(X[:10], y[:10], classes=np.unique(y))
    for _ in range(30):
        clf.partial_fit(X, y)

    # Check that using the classes argument wrong will fail.
    with pytest.raises(ValueError):
        clf = MLPClassifier(n_epochs=1)
        clf.partial_fit(X, y, classes=np.array([0, 1]))

    # Check that fitting twice works.
    clf = MLPClassifier(n_epochs=1)
    clf.fit(X, y)
    assert np.array_equal(clf.classes_, np.unique(y))
    y_binary = (y == y[0]).astype(float)
    clf.fit(X, y_binary)
    assert np.array_equal(clf.classes_, np.unique(y_binary))
