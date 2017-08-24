"""
A Deep Neural Network (multilayer Perceptron) sklearn-style estimator abstract
class.

Similar to sklearn.neural_network.MLPClassifier, but using TensorFlow.
"""
from __future__ import print_function
from __future__ import division

from abc import ABCMeta, abstractmethod
import six
import logging
import re
from warnings import warn

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.utils import (check_array, check_random_state, check_X_y,
                           DataConversionWarning)

from sklearn.exceptions import NotFittedError

import tensorflow as tf
from tensorflow.python.framework.ops import Graph
from tensorflow.python.framework import random_seed as tf_random_seed

from muffnn.core import TFPicklingBase, affine


_LOGGER = logging.getLogger(__name__)


@six.add_metaclass(ABCMeta)
class MLPBaseEstimator(TFPicklingBase, BaseEstimator):
    """Base class for multilayer perceptron models

    Notes
    -----
    There is currently no dropout between the sparse input layer and first
    hidden layer. Dropout on the sparse input layer would undo the benefits of
    sparsity because the dropout layer is dense.
    """

    def _transform_targets(self, y):
        # This can be overridden to, e.g., map label names to indices when
        # fitting a classifier.
        return y

    def fit(self, X, y, monitor=None):
        """Fit the model.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Training data
        y : numpy array of shape [n_samples, n_targets]
            Target values
        monitor : callable, optional
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator, and a dictionary with
            {'loss': loss_value} representing the loss calculated by the
            objective function at this iteration.
            If the callable returns True the fitting procedure is stopped.
            The monitor can be used for various things such as computing
            held-out estimates, early stopping, model introspection,
            and snapshoting.

        Returns
        -------
        self : returns an instance of self.
        """
        _LOGGER.info("Fitting %s", re.sub(r"\s+", r" ", repr(self)))

        # Mark the model as not fitted (i.e., not fully initialized based on
        # the data).
        self._is_fitted = False

        # Call partial fit, which will initialize and then train the model.
        return self.partial_fit(X, y, monitor=monitor)

    def _fit_targets(self, y):
        # This can be overwritten to set instance variables that pertain to the
        # targets (e.g., an array of class labels).
        pass

    def partial_fit(self, X, y, monitor=None, **kwargs):
        """Fit the model on a batch of training data.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Training data
        y : numpy array of shape [n_samples, n_targets]
            Target values
        monitor : callable, optional
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator, and a dictionary with
            {'loss': loss_value} representing the loss calculated by the
            objective function at this iteration.
            If the callable returns True the fitting procedure is stopped.
            The monitor can be used for various things such as computing
            held-out estimates, early stopping, model introspection,
            and snapshoting.

        Returns
        -------
        self : returns an instance of self.
        """

        X, y = self._check_inputs(X, y)
        assert self.batch_size > 0, "batch_size <= 0"

        # Initialize the model if it hasn't been already by a previous call.
        if self._is_fitted:
            y = self._transform_targets(y)
        else:
            self._random_state = check_random_state(self.random_state)
            self._fit_targets(y, **kwargs)
            y = self._transform_targets(y)

            self.is_sparse_ = sp.issparse(X)
            self.input_layer_sz_ = X.shape[1]

            # Instantiate the graph.  TensorFlow seems easier to use by just
            # adding to the default graph, and as_default lets you temporarily
            # set a graph to be treated as the default graph.
            self.graph_ = Graph()
            with self.graph_.as_default():
                tf_random_seed.set_random_seed(
                    self._random_state.randint(0, 10000000))

                tf.get_variable_scope().set_initializer(
                    tf.uniform_unit_scaling_initializer(self.init_scale))

                self._build_tf_graph()

                # Train model parameters.
                self._session.run(tf.global_variables_initializer())

            # Set an attributed to mark this as at least partially fitted.
            self._is_fitted = True

        # Train the model with the given data.
        with self.graph_.as_default():
            n_examples = X.shape[0]
            indices = np.arange(n_examples)

            for epoch in range(self.n_epochs):
                self._random_state.shuffle(indices)
                for start_idx in range(0, n_examples, self.batch_size):
                    batch_ind = indices[start_idx:start_idx + self.batch_size]
                    feed_dict = self._make_feed_dict(X[batch_ind],
                                                     y[batch_ind])
                    obj_val, _ = self._session.run(
                        [self._obj_func, self._train_step],
                        feed_dict=feed_dict)
                    _LOGGER.debug("objective: %.4f, epoch: %d, idx: %d",
                                  obj_val, epoch, start_idx)

                _LOGGER.info("objective: %.4f, epoch: %d, idx: %d",
                             obj_val, epoch, start_idx)

                if monitor:
                    stop_early = monitor(epoch, self, {'loss': obj_val})
                    if stop_early:
                        _LOGGER.info(
                            "stopping early due to monitor function.")
                        return self

        return self

    def _check_inputs(self, X, y):
        # Check that the input X is an array or sparse matrix.
        # Convert to CSR if it's in another sparse format.
        X, y = check_X_y(X, y, accept_sparse='csr', multi_output=True)

        if y.ndim == 2 and y.shape[1] == 1:
            # Following
            # https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/ensemble/forest.py#L223,
            # issue a warning if an Nx1 array was provided.
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)
            y = y[:, 0]
        return X, y

    def __getstate__(self):
        # Handles TF persistence
        state = super(MLPBaseEstimator, self).__getstate__()

        # Add attributes of this estimator
        state.update(dict(activation=self.activation,
                          batch_size=self.batch_size,
                          keep_prob=self.keep_prob,
                          hidden_units=self.hidden_units,
                          init_scale=self.init_scale,
                          random_state=self.random_state,
                          n_epochs=self.n_epochs,
                          solver=self.solver,
                          solver_kwargs=self.solver_kwargs
                          ))

        # Add fitted attributes if the model has been fitted.
        if self._is_fitted:
            state['input_layer_sz_'] = self.input_layer_sz_
            state['is_sparse_'] = self.is_sparse_
            state['_random_state'] = self._random_state

        return state

    @abstractmethod
    def _init_model_output(self, t):
        pass

    @abstractmethod
    def _init_model_objective_fn(self, t):
        pass

    def _set_up_graph(self):
        """Initialize TF objects (needed before fitting or restoring)."""

        # A placeholder to control dropout for training vs. prediction.
        self._keep_prob = \
            tf.placeholder(dtype=np.float32, shape=(), name="keep_prob")

        # Input layers.
        if self.is_sparse_:
            self._input_indices = \
                tf.placeholder(np.int64, [None, 2], "input_indices")
            self._input_values = \
                tf.placeholder(np.float32, [None], "input_values")
            self._input_shape = \
                tf.placeholder(np.int64, [2], "input_shape")
            # t will be the current layer as we build up the graph below.
            t = tf.SparseTensor(self._input_indices, self._input_values,
                                self._input_shape)
        else:
            self._input_values = \
                tf.placeholder(np.float32, [None, self.input_layer_sz_],
                               "input_values")
            t = self._input_values

        # Hidden layers.
        for i, layer_sz in enumerate(self.hidden_units):
            if self.is_sparse_ and i == 0:
                t = affine(t, layer_sz, input_size=self.input_layer_sz_,
                           scope='layer_%d' % i, sparse_input=True)
            else:
                if self.keep_prob != 1.0:
                    t = tf.nn.dropout(t, keep_prob=self._keep_prob)
                t = affine(t, layer_sz, scope='layer_%d' % i)
            t = t if self.activation is None else self.activation(t)

        # The output layer and objective function depend on the model
        # (e.g., classification vs regression).
        t = self._init_model_output(t)
        self._init_model_objective_fn(t)

        self._train_step = self.solver(
            **self.solver_kwargs if self.solver_kwargs else {}).minimize(
            self._obj_func)

    def _make_feed_dict(self, X, y=None):
        # Make the dictionary mapping tensor placeholders to input data.

        if self.is_sparse_:
            indices, values = _sparse_matrix_data(X)

            feed_dict = {
                self._input_indices: indices,
                self._input_values: values,
                self._input_shape: X.shape
            }
        else:
            feed_dict = {
                self._input_values: X
            }

        if y is None:
            # If y is None, then we are doing prediction and should fix
            # dropout.
            feed_dict[self._keep_prob] = 1.0
        else:
            feed_dict[self.input_targets_] = y
            feed_dict[self._keep_prob] = self.keep_prob

        return feed_dict

    def _compute_output(self, X):
        """Get the outputs of the network, for use in prediction methods."""

        if not self._is_fitted:
            raise NotFittedError("Call fit before prediction")

        X = check_array(X, accept_sparse=['csr', 'dok', 'lil', 'csc', 'coo'])

        if self.is_sparse_:
            # For sparse input, make the input a CSR matrix since it can be
            # indexed by row.
            X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
        elif sp.issparse(X):
            # Convert sparse input to dense.
            X = X.todense().A

        # Make predictions in batches.
        pred_batches = []
        start_idx = 0
        n_examples = X.shape[0]
        with self.graph_.as_default():
            while start_idx < n_examples:
                X_batch = \
                    X[start_idx:min(start_idx + self.batch_size, n_examples)]
                feed_dict = self._make_feed_dict(X_batch)
                start_idx += self.batch_size
                pred_batches.append(
                    self._session.run(self.output_layer_, feed_dict=feed_dict))
        y_pred = np.concatenate(pred_batches)
        return y_pred

    @abstractmethod
    def predict(self, X):
        pass


def _sparse_matrix_data(X):
    """Prepare the sparse matrix for conversion to TensorFlow.

    Parameters
    ----------
    X : sparse matrix

    Returns
    -------
    indices : numpy array with shape (X.nnz, 2)
              describing the indices with values in X.
    values : numpy array with shape (X.nnz)
             describing the values at each index
    """
    if sp.isspmatrix_csr(X):
        return _csr_data(X)
    else:
        return _csr_data(X.tocsr())


def _csr_data(X):
    """Prepare the CSR sparse matrix for conversion to TensorFlow.

    Parameters
    ----------
    X : sparse matrix in CSR format

    Returns
    -------
    indices : numpy array with shape (X.nnz, 2)
              describing the indices with values in X.
    values : numpy array with shape (X.nnz)
             describing the values at each index
    """
    indices = np.zeros((X.nnz, 2))
    values = np.zeros(X.nnz)
    i = 0
    for row_idx in range(X.shape[0]):
        column_indices = X.indices[X.indptr[row_idx]: X.indptr[row_idx + 1]]
        row_values = X.data[X.indptr[row_idx]: X.indptr[row_idx + 1]]
        for column_idx, row_value in zip(column_indices, row_values):
            indices[i][0] = row_idx
            indices[i][1] = column_idx
            values[i] = row_value
            i += 1
    return indices, values
