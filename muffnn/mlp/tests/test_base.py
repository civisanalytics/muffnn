from io import BytesIO
import pickle

import pytest
import numpy as np
import scipy.sparse
import tensorflow as tf
from tensorflow.python.ops import nn

import muffnn.mlp.base as base

from unittest import mock


class SimpleTestEstimator(base.MLPBaseEstimator):

    _input_indices = 'input_indices'
    _input_values = 'input_values'
    _sample_weight = 'sample_weight'
    _input_shape = 'input_shape'
    input_targets_ = 'input_targets'
    _keep_prob = 1.0

    def __init__(self, hidden_units=(256,), batch_size=64, n_epochs=5,
                 keep_prob=1.0, activation=nn.relu,
                 random_state=None, monitor=None,
                 solver=tf.train.AdamOptimizer, solver_kwargs=None,
                 transform_layer_index=None):
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.keep_prob = keep_prob
        self.activation = activation
        self.random_state = random_state
        self.monitor = monitor
        self.solver = solver
        self.solver_kwargs = solver_kwargs
        self.transform_layer_index = transform_layer_index

    def _init_model_output(self, t):
        self.input_targets_ = tf.placeholder(tf.int64, [None], "targets")
        self.output_layer_ = tf.nn.softmax(t)
        return t

    def predict(*_, **__):
        pass

    def _init_model_objective_fn(self, t):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=t, labels=self.input_targets_)
        self._obj_func = tf.reduce_mean(cross_entropy)


def test_make_feed_dict_csr():
    X = scipy.sparse.csr_matrix((
        np.array([1, 2, 3, 4, 5, 6]),
        (np.array([0, 0, 1, 2, 2, 2], dtype=np.int64),
         np.array([0, 2, 2, 0, 1, 2], dtype=np.int64))),
        shape=(3, 3))
    y = np.array([1, 2, 3])

    clf = SimpleTestEstimator()
    clf.is_sparse_ = True

    feed_dict = clf._make_feed_dict(X, y)
    assert_feed_dict_equals(feed_dict, X)
    np.testing.assert_array_equal(feed_dict['input_targets'], y)


def test_make_feed_dict_other():
    X = scipy.sparse.coo_matrix((
        np.array([1, 2, 3, 4, 5, 6]),
        (np.array([0, 0, 1, 2, 2, 2], dtype=np.int64),
         np.array([0, 2, 2, 0, 1, 2], dtype=np.int64))),
        shape=(3, 3))
    y = np.array([1, 2, 3])

    clf = SimpleTestEstimator()
    clf.is_sparse_ = True

    feed_dict = clf._make_feed_dict(X, y)
    assert_feed_dict_equals(feed_dict, X)
    np.testing.assert_array_equal(feed_dict['input_targets'], y)


@mock.patch.object(base.tf, 'Session')
def test_fit_monitor(mock_Session):
    # Ensure that we loop through batches and epochs appropriately
    # while respecting the monitor function's ability to cause
    # early stopping.
    mock_Session().run.return_value = (
        mock.MagicMock(), mock.MagicMock())
    mock_eval = mock.MagicMock(return_value=False)

    X = np.reshape(np.arange(9), (3, 3))
    y = np.arange(3)

    clf = SimpleTestEstimator(n_epochs=10, batch_size=2)
    clf.fit(X, y, monitor=mock_eval)
    # two batches per epoch + an initialize variables call
    assert mock_Session().run.call_count == 21
    assert mock_eval.call_count == 10

    mock_Session.reset_mock()
    mock_eval.reset_mock()
    clf = SimpleTestEstimator(n_epochs=10, batch_size=3)
    clf.fit(X, y, monitor=mock_eval)
    # one batch per epoch
    assert mock_Session().run.call_count == 11
    assert mock_eval.call_count == 10

    mock_Session.reset_mock()
    mock_eval.reset_mock()
    mock_eval = mock.MagicMock(return_value=True)
    clf = SimpleTestEstimator(n_epochs=10, batch_size=3)
    clf.fit(X, y, monitor=mock_eval)
    # Our monitor returns true, so it should only do one epoch
    assert mock_Session().run.call_count == 2
    assert mock_eval.call_count == 1


def assert_feed_dict_equals(feed_dict, matrix):
    """Ensure that the sparse matrix described by `feed_dict` represents
    the same data as `matrix`.
    """
    indices = zip(*feed_dict['input_indices'])
    result_matrix = scipy.sparse.bsr_matrix(
        (feed_dict['input_values'], indices), shape=feed_dict['input_shape'])

    np.testing.assert_array_equal(matrix.toarray(), result_matrix.toarray())


def test_partial_fit_random_state():
    """Ensure partial_fit doesn't reset the random state used for shuffling.

    Mock the feed dict function so as to see which examples get fed into the
    TensorFlow graph. The examples are shuffled by the random state instance,
    so the ordering should differ in successive `partial_fit` calls.
    """

    # Make data long enough that coincidental ordering matches are unlikely.
    y = np.arange(50)
    X = np.expand_dims(y, 1)

    clf = SimpleTestEstimator(random_state=42, n_epochs=1)
    clf.is_sparse_ = False

    # Note that the first `partial_fit` call will use the random state to set
    # TensorFlow's seed in addition to shuffling examples, so we'll check the
    # result of multiple `partial_fit` calls.
    clf.partial_fit(X, y)

    # Pickle the (partially) fitted estimator) to make sure that the random
    # state works as expected after pickling and unpickling.
    buf = BytesIO()
    pickle.dump(clf, buf)

    # Instrument the function for making TF inputs.
    mock_make_feed_dict = mock.MagicMock()
    mock_make_feed_dict.side_effect = clf._make_feed_dict
    clf._make_feed_dict = mock_make_feed_dict

    # Run partial_fit many times and make sure the example orders are unique.
    n_calls = 50
    for _ in range(n_calls):
        clf.partial_fit(X, y)
    unique_orderings = {tuple(x[0][1]) for x
                        in mock_make_feed_dict.call_args_list}
    assert len(unique_orderings) == n_calls

    # Now unpickle the model, run partial_fit, and make sure the results are
    # the same.
    buf.seek(0)
    clf = pickle.load(buf)
    mock_make_feed_dict = mock.MagicMock()
    mock_make_feed_dict.side_effect = clf._make_feed_dict
    clf._make_feed_dict = mock_make_feed_dict
    for _ in range(n_calls):
        clf.partial_fit(X, y)

    unique_orderings2 = {tuple(x[0][1]) for x
                         in mock_make_feed_dict.call_args_list}

    assert unique_orderings == unique_orderings2


def test_transform_layer_index_out_of_range():
    """Ensure the base class raises a ValueError if the transform_layer_index
    is out of range"""

    y = np.arange(50)
    X = np.expand_dims(y, 1)

    with pytest.raises(ValueError):
        clf = SimpleTestEstimator(transform_layer_index=-2, hidden_units=[])
        clf.partial_fit(X, y)

    with pytest.raises(ValueError):
        clf = SimpleTestEstimator(transform_layer_index=1, hidden_units=[256])
        clf.partial_fit(X, y)
