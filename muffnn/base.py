#!/usr/bin/env python3

"""
A Deep Neural Network (multilayer Perceptron) sklearn-style estimator abstract
class.

Similar to sklearn.neural_network.MLPClassifier, but using TensorFlow.
"""

from abc import ABCMeta, abstractmethod
import logging
import re
import os
from tempfile import NamedTemporaryFile
from warnings import warn

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.utils import (check_array, check_random_state, check_X_y,
                           DataConversionWarning)

from sklearn.utils.validation import NotFittedError

import tensorflow as tf
from tensorflow.python.framework.ops import Graph
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import random_seed as tf_random_seed
from tensorflow.python.ops import init_ops


_LOGGER = logging.getLogger(__name__)


def _affine(input_tensor, output_size, bias=True, bias_start=0.0,
            input_sz=None, scope="affine", sparse_input=False):
    # loosely based on tensorflow.python.ops.rnn_cell.linear.

    # The input size is needed for sparse matrices.
    if input_sz is None:
        input_sz = input_tensor.get_shape().as_list()[1]

    with vs.variable_scope(scope):
        W_0 = vs.get_variable(
            "weights0",
            [input_sz, output_size])
        matmul = tf.sparse_tensor_dense_matmul if sparse_input else tf.matmul
        t = matmul(input_tensor, W_0)

        if bias:
            b_0 = vs.get_variable(
                "bias0",
                [output_size],
                initializer=init_ops.constant_initializer(bias_start))
            t = tf.add(t, b_0)
    return t


class MLPBaseEstimator(BaseEstimator, metaclass=ABCMeta):
    """Base class for multilayer perceptron models

    Notes
    -----
    There is currently no dropout for sparse input layers.
    """

    def _preprocess_targets(self, y):
        # This can be overridden to, e.g., map label names to indices when
        # fitting a classifier.
        return y

    def fit(self, X, y):
        """Fit the model.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Training data
        y : numpy array of shape [n_samples, n_targets]
            Target values

        Returns
        -------
        self : returns an instance of self.
        """
        _LOGGER.info("Fitting %s", re.sub(r"\s+", r" ", repr(self)))

        # Mark the model as not fitted (i.e., not fully initialized based on
        # the data).
        self._is_fitted = False

        # Call partial fit, which will initialize and then train the model.
        return self.partial_fit(X, y)

    def _fit_targets(self, y):
        # This can be overwritten to set instance variables that pertain to the
        # targets (e.g., an array of class labels).
        pass

    def partial_fit(self, X, y, **kwargs):
        """Fit the model on a batch of training data.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Training data
        y : numpy array of shape [n_samples, n_targets]
            Target values

        Returns
        -------
        self : returns an instance of self.
        """

        X, y = self._check_inputs(X, y)

        random_state = check_random_state(self.random_state)
        assert self.batch_size > 0, "batch_size <= 0"

        # Initialize the model if it hasn't been already by a previous call.
        if self._is_fitted:
            y = self._preprocess_targets(y)
        else:
            self._fit_targets(y, **kwargs)
            y = self._preprocess_targets(y)

            self.is_sparse_ = sp.issparse(X)
            self.input_layer_sz_ = X.shape[1]

            # Instantiate the graph.  TensorFlow seems easier to use by just
            # adding to the default graph, and as_default lets you temporarily
            # set a graph to be treated as the default graph.
            self.graph_ = Graph()
            with self.graph_.as_default():
                tf_random_seed.set_random_seed(
                    random_state.randint(0, 10000000))

                tf.get_variable_scope().set_initializer(
                    tf.uniform_unit_scaling_initializer(self.init_scale))

                self._set_up_graph()
                self._session = tf.Session()

                # Train model parameters.
                self._session.run(tf.initialize_all_variables())

            # Set an attributed to mark this as at least partially fitted.
            self._is_fitted = True

        # Train the model with the given data.
        with self.graph_.as_default():
            n_examples = X.shape[0]
            start_idx = 0
            epoch = 0
            indices = np.arange(n_examples)
            random_state.shuffle(indices)

            while True:
                batch_ind = indices[start_idx:start_idx + self.batch_size]
                feed_dict = self._make_feed_dict(X[batch_ind],
                                                 y[batch_ind])
                obj_val, _ = self._session.run(
                    [self._obj_func, self._train_step], feed_dict=feed_dict)
                _LOGGER.info("objective: %.4f, epoch: %d, idx: %d",
                             obj_val, epoch, start_idx)
                start_idx += self.batch_size
                if start_idx > n_examples - self.batch_size:
                    start_idx = 0
                    epoch += 1
                    if epoch >= self.n_epochs:
                        break
                    random_state.shuffle(indices)

        return self

    @property
    def _is_fitted(self):
        """Return True if the model has been at least partially fitted.

        Returns
        -------
        bool

        Notes
        -----
        This is to indicate whether, e.g., the TensorFlow graph for the model
        has been created.
        """
        return getattr(self, '_fitted', False)

    @_is_fitted.setter
    def _is_fitted(self, b):
        """Set whether the model has been at least partially fitted.

        Parameters
        ----------
        b : bool
            True if the model has been fitted.
        """
        self._fitted = b

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
        # Override __getstate__ so that TF model parameters are pickled
        # properly.
        if self._is_fitted:
            tempfile = NamedTemporaryFile(delete=False)
            tempfile.close()
            try:
                # Serialize the model and read it so it can be pickled.
                self._saver.save(self._session, tempfile.name)
                with open(tempfile.name, 'rb') as f:
                    saved_model = f.read()
            finally:
                os.unlink(tempfile.name)

        # Note: don't include the graph since it can be recreated.
        state = dict(
            activation=self.activation,
            batch_size=self.batch_size,
            dropout=self.dropout,
            hidden_units=self.hidden_units,
            init_scale=self.init_scale,
            random_state=self.random_state,
            n_epochs=self.n_epochs,
        )

        # Add fitted attributes if the model has been fitted.
        if self._is_fitted:
            state['_fitted'] = True
            state['input_layer_sz_'] = self.input_layer_sz_
            state['is_sparse_'] = self.is_sparse_
            state['_saved_model'] = saved_model

        return state

    def __setstate__(self, state):
        # Override __setstate__ so that TF model parameters are unpickled
        # properly.
        for k, v in state.items():
            if k == '_saved_model':
                continue
            self.__dict__[k] = v

        if state.get('_fitted', False):
            tempfile = NamedTemporaryFile(delete=False)
            tempfile.close()
            try:
                # Write out the serialized model that can be restored by TF.
                with open(tempfile.name, 'wb') as f:
                    f.write(state['_saved_model'])
                self.graph_ = Graph()
                with self.graph_.as_default():
                    self._set_up_graph()
                    self._session = tf.Session()
                    self._saver.restore(self._session, tempfile.name)
            finally:
                os.unlink(tempfile.name)

    @abstractmethod
    def _init_model_output(self, t):
        pass

    @abstractmethod
    def _init_model_objective_fn(self, t):
        pass

    def _set_up_graph(self):
        """Initialize TF objects (needed before fitting or restoring)."""

        # A placeholder to control dropout for training vs. prediction.
        self._dropout = \
            tf.placeholder(dtype=np.float32, shape=(), name="dropout")

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
                t = _affine(t, layer_sz, input_sz=self.input_layer_sz_,
                            scope='layer_%d' % i, sparse_input=True)
            else:
                t = tf.nn.dropout(t, keep_prob=self._dropout)
                t = _affine(t, layer_sz, scope='layer_%d' % i)
            t = t if self.activation is None else self.activation(t)

        # The output layer and objective function depend on the model
        # (e.g., classification vs regression).
        t = self._init_model_output(t)
        self._init_model_objective_fn(t)

        self._train_step = tf.train.AdamOptimizer().minimize(self._obj_func)

        self._saver = tf.train.Saver()

    def _make_feed_dict(self, X, y=None):
        # Make the dictionary mapping tensor placeholders to input data.

        if self.is_sparse_:
            # TF's sparse matrix is initialized by DoK data.
            X = X.todok()
            # Make sure the input is a 2-d array.
            indices = np.array(list(X.keys()) if X.nnz > 0
                               else np.zeros((0, 2)))
            values = np.array(list(X.values()))

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
            feed_dict[self._dropout] = 1.0
        else:
            feed_dict[self.input_targets_] = y
            feed_dict[self._dropout] = \
                self.dropout if self.dropout is not None else 1.0

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
