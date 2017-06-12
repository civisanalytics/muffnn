"""
Tests for the FM classifier.

based in part on sklearn's logistic tests:
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/tests/test_logistic.py
"""

from io import BytesIO
import pickle

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import load_iris
from sklearn.linear_model.tests.test_logistic import check_predictions
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold

from ..fm_classifier import FMClassifier

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
        super().__init__(rank=1, solver='L-BFGS-B', random_state=2)


def test_check_estimator():
    """Check adherence to Estimator API."""
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
