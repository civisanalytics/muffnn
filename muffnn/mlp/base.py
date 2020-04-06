"""
A Deep Neural Network (multilayer Perceptron) sklearn-style estimator abstract
class.

Similar to sklearn.neural_network.MLPClassifier, but using TensorFlow.
"""

from abc import ABCMeta, abstractmethod
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


class MLPBaseEstimator(TFPicklingBase, BaseEstimator, metaclass=ABCMeta):
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

    def fit(self, X, y, monitor=None, sample_weight=None):
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
        sample_weight : numpy array of shape [n_samples,]
            Per-sample weights. Re-scale the loss per sample.
            Higher weights force the estimator to put more emphasis
            on these samples. Sample weights are normalized per-batch.

        Returns
        -------
        self : returns an instance of self.
        """
        _LOGGER.info("Fitting %s", re.sub(r"\s+", r" ", repr(self)))

        # Mark the model as not fitted (i.e., not fully initialized based on
        # the data).
        self._is_fitted = False

        # Call partial fit, which will initialize and then train the model.
        return self.partial_fit(X, y,
                                monitor=monitor,
                                sample_weight=sample_weight)

    def _fit_targets(self, y):
        # This can be overwritten to set instance variables that pertain to the
        # targets (e.g., an array of class labels).
        pass

    def partial_fit(self, X, y, monitor=None, sample_weight=None, **kwargs):
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
        sample_weight : numpy array of shape [n_samples,]
            Per-sample weights. Re-scale the loss per sample.
            Higher weights force the estimator to put more emphasis
            on these samples. Sample weights are normalized per-batch.

        Returns
        -------
        self : returns an instance of self.
        """

        X, y = self._check_inputs(X, y)
        assert self.batch_size > 0, "batch_size <= 0"

        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)

        # Initialize the model if it hasn't been already by a previous call.
        if self._is_fitted:
            y = self._transform_targets(y)
        else:
            self._random_state = check_random_state(self.random_state)
            self._fit_targets(y, **kwargs)
            y = self._transform_targets(y)

            self.is_sparse_ = sp.issparse(X)
            self.input_layer_sz_ = X.shape[1]

            # Set which layer transform function points to
            if self.transform_layer_index is None:
                self._transform_layer_index = len(self.hidden_units) - 1
            else:
                self._transform_layer_index = self.transform_layer_index

            if (self._transform_layer_index < -1 or
                    self._transform_layer_index >= len(self.hidden_units)):
                raise ValueError(
                    "`transform_layer_index` must be in the range "
                    "[-1, len(hidden_units)-1]!")

            # Instantiate the graph.  TensorFlow seems easier to use by just
            # adding to the default graph, and as_default lets you temporarily
            # set a graph to be treated as the default graph.
            self.graph_ = Graph()
            with self.graph_.as_default():
                tf_random_seed.set_random_seed(
                    self._random_state.randint(0, 10000000))

                tf.get_variable_scope().set_initializer(
                    tf.contrib.layers.xavier_initializer())

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

                    if sample_weight is None:
                        batch_sample_weight = None
                    else:
                        batch_sample_weight = sample_weight[batch_ind]

                    feed_dict = self._make_feed_dict(
                        X[batch_ind],
                        y[batch_ind],
                        sample_weight=batch_sample_weight)
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
                          random_state=self.random_state,
                          n_epochs=self.n_epochs,
                          solver=self.solver,
                          solver_kwargs=self.solver_kwargs,
                          transform_layer_index=self.transform_layer_index
                          ))

        # Add fitted attributes if the model has been fitted.
        if self._is_fitted:
            state['input_layer_sz_'] = self.input_layer_sz_
            state['is_sparse_'] = self.is_sparse_
            state['_random_state'] = self._random_state
            state['_transform_layer_index'] = self._transform_layer_index

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
                    if self.activation is tf.nn.selu:
                        t = tf.contrib.nn.alpha_dropout(
                            t, keep_prob=self._keep_prob)
                    else:
                        t = tf.nn.dropout(t, keep_prob=self._keep_prob)
                t = affine(t, layer_sz, scope='layer_%d' % i)

            t = t if self.activation is None else self.activation(t)

            # Set transformed layer to hidden layer
            if self._transform_layer_index == i:
                self._transform_layer = t

        # The output layer and objective function depend on the model
        # (e.g., classification vs regression).
        t = self._init_model_output(t)

        # set the transform layer to output logits if we have no hidden layers
        if self._transform_layer_index == -1:
            self._transform_layer = t

        # Prediction gradients (e.g., for analyzing the importance of features)
        # We use the top layer before the output activation function
        # (e.g., softmax, sigmoid) following
        # https://arxiv.org/pdf/1312.6034.pdf
        if self.is_sparse_:
            self._prediction_gradient = None
        else:
            output_shape = self.output_layer_.get_shape()
            # Note: tf.gradients returns a list of gradients dy/dx, one per
            # input tensor x. In other words,
            # [ tensor(n_features x n_gradients) ].
            if len(output_shape) == 1:
                self._prediction_gradient = tf.gradients(
                    t, self._input_values)[0]
            elif len(output_shape) == 2:
                # According to the tf.gradients documentation, it looks like
                # we have to compute gradients separately for each output
                # dimension and then stack them for multiclass/label data.
                self._prediction_gradient = tf.stack([
                    tf.gradients(t[:, i], self._input_values)[0]
                    for i in range(output_shape[1])
                ], axis=1)
            else:  # sanity check
                raise ValueError("Unexpected output shape")

        self._sample_weight = \
            tf.placeholder(np.float32, [None], "sample_weight")

        self._init_model_objective_fn(t)

        self._train_step = self.solver(
            **self.solver_kwargs if self.solver_kwargs else {}).minimize(
            self._obj_func)

    def _make_feed_dict(self, X, y=None, sample_weight=None):
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

        if sample_weight is None:
            feed_dict[self._sample_weight] = np.ones(X.shape[0])
        else:
            feed_dict[self._sample_weight] = sample_weight

        return feed_dict

    def _compute_output(self, X):
        """Get the outputs of the network, for use in prediction methods."""

        if not self._is_fitted:
            raise NotFittedError("Call fit before prediction")

        X = self._check_X(X)

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

    def _check_X(self, X):
        X = check_array(X, accept_sparse=['csr', 'dok', 'lil', 'csc', 'coo'])

        if self.is_sparse_:
            # For sparse input, make the input a CSR matrix since it can be
            # indexed by row.
            X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
        elif sp.issparse(X):
            # Convert sparse input to dense.
            X = X.todense().A

        return X

    @abstractmethod
    def predict(self, X):
        pass

    def transform(self, X, y=None):
        """Transforms input into hidden layer outputs of users choice.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Examples to make predictions about.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.
        """
        if not self._is_fitted:
            raise NotFittedError("Call fit before transform")

        X = self._check_X(X)

        # Make predictions in batches.
        embed_batches = []
        start_idx = 0
        n_examples = X.shape[0]
        with self.graph_.as_default():
            while start_idx < n_examples:
                X_batch = \
                    X[start_idx:min(start_idx + self.batch_size, n_examples)]
                feed_dict = self._make_feed_dict(X_batch)
                start_idx += self.batch_size
                embed_batches.append(self._session.run(
                    self._transform_layer, feed_dict=feed_dict))
        embedding = np.concatenate(embed_batches)
        if embedding.ndim == 1:
            embedding = embedding.reshape(-1, 1)
        return embedding

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.
        y : numpy array of shape [n_samples]
            Target values.
        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)

    def prediction_gradient(self, X):
        """Compute the prediction gradient with respect to the given inputs.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Examples to compute feature importance based on.

        Returns
        -------
        numpy array
            Array of gradients for each example. For single-class or regression
            problems, this will be of shape [n_samples, n_features].
            For multilabel or multiclass problems, it will be of shape
            [n_samples, n_classes, n_features].

        Notes
        -----
        This is related to what is sometimes referred to as
        "sensitivity analysis" (e.g., Simonyan et al., ICLR 2014). There are
        more complex methods for computing the effects of input features on
        outputs (e.g., SHAP, LIME, layerwise relevance propagation, DeepLIFT),
        but using prediction gradients is fast and simple, and also has
        theoretical connections to other methods.  See, e.g., Shrikumar (2017,
        https://arxiv.org/abs/1704.02685) for discussion and further
        references. This function can also be used for the "gradient x input"
        technique described by Shrikumar (2017).

        When using this method, be careful to consider the variance of the
        features when interpreting the results. You may want to use scale
        the values to [0, 1] or to have a mean of 0 and variance of 1.
        """
        if not self._is_fitted:
            raise NotFittedError("Call fit first.")

        if self.is_sparse_:
            raise NotImplementedError("Not implemented for sparse inputs.")

        X = self._check_X(X)

        # Compute gradients in batches.
        imprt_vals = []
        start_idx = 0
        n_examples = X.shape[0]
        with self.graph_.as_default():
            while start_idx < n_examples:
                X_batch = \
                    X[start_idx:min(start_idx + self.batch_size, n_examples)]
                feed_dict = self._make_feed_dict(X_batch)
                start_idx += self.batch_size
                imprt_vals.append(
                    self._session.run(self._prediction_gradient,
                                      feed_dict=feed_dict))

        imprt_vals = np.concatenate(imprt_vals)

        return imprt_vals


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
    indices = np.zeros((X.nnz, 2), dtype=np.int64)
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
