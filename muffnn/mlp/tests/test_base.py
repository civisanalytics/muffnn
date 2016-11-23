import unittest.mock

import numpy as np
import scipy.sparse
import tensorflow as tf
from tensorflow.python.ops import nn

import civistext.mlp.base as base


class TestEstimator(base.MLPBaseEstimator):

    _input_indices = 'input_indices'
    _input_values = 'input_values'
    _input_shape = 'input_shape'
    input_labels_ = 'input_labels'

    def __init__(self, hidden_units=(256,), batch_size=64, n_epochs=5,
                 dropout=None, activation=nn.relu, init_scale=0.1,
                 random_state=None, monitor=None):
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout = dropout
        self.activation = activation
        self.init_scale = init_scale
        self.random_state = random_state
        self.monitor = monitor

    def _init_model_output(self, t, _):
        self.input_labels_ = tf.placeholder(tf.int64, [None], "labels")
        self.output_layer_ = tf.nn.softmax(t)
        return t

    def predict(*_, **__):
        pass

    def _init_model_objective_fn(self, t):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            t, self.input_labels_)
        self._obj_func = tf.reduce_mean(cross_entropy)

def test_make_feed_dict_csr():
    X = scipy.sparse.csr_matrix((
        np.array([1, 2, 3, 4, 5, 6]),
        (np.array([0, 0, 1, 2, 2, 2]), np.array([0, 2, 2, 0, 1, 2]))),
        shape=(3, 3))
    y = np.array([1, 2, 3])

    clf = TestEstimator()
    clf.is_sparse_ = True

    feed_dict = clf._make_feed_dict(X, y)
    assert_feed_dict_equals(feed_dict, X)
    np.testing.assert_array_equal(feed_dict['input_labels'], y)


def test_make_feed_dict_other():
    X = scipy.sparse.coo_matrix((
        np.array([1, 2, 3, 4, 5, 6]),
        (np.array([0, 0, 1, 2, 2, 2]), np.array([0, 2, 2, 0, 1, 2]))),
        shape=(3, 3))
    y = np.array([1, 2, 3])

    clf = TestEstimator()
    clf.is_sparse_ = True

    feed_dict = clf._make_feed_dict(X, y)
    assert_feed_dict_equals(feed_dict, X)
    np.testing.assert_array_equal(feed_dict['input_labels'], y)


@unittest.mock.patch.object(base.tf, 'Session')
def test_fit_monitor(mock_Session):
    # Ensure that we loop through batches and epochs appropriately
    # while respecting the monitor function's ability to cause
    # early stopping.
    mock_Session().run.return_value = (
        unittest.mock.MagicMock(), unittest.mock.MagicMock())
    mock_eval = unittest.mock.MagicMock(return_value=False)

    X = np.reshape(np.arange(9), (3, 3))
    y = np.arange(3)

    clf = TestEstimator(n_epochs=10, batch_size=2)
    clf.fit(X, y, monitor=mock_eval)
    # two batches per epoch + an initialize variables call
    assert mock_Session().run.call_count == 21
    assert mock_eval.call_count == 10

    mock_Session.reset_mock()
    mock_eval.reset_mock()
    clf = TestEstimator(n_epochs=10, batch_size=3)
    clf.fit(X, y, monitor=mock_eval)
    # one batch per epoch
    assert mock_Session().run.call_count == 11
    assert mock_eval.call_count == 10

    mock_Session.reset_mock()
    mock_eval.reset_mock()
    mock_eval = unittest.mock.MagicMock(return_value=True)
    clf = TestEstimator(n_epochs=10, batch_size=3)
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
