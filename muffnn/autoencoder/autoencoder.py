"""
Autoencoder in scikit-learn style with TensorFlow
"""
import logging
import re

import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import NotFittedError

from ..core import TFPicklingBase, affine


_LOGGER = logging.getLogger(__name__)


class Autoencoder(TFPicklingBase, TransformerMixin, BaseEstimator):
    """
    An autoencoder using TensorFlow.

    Parameters
    ----------
    hidden_units : tuple or list, optional
        A list of integers indicating the number of hidden layers and their
        sizes. The list is reflected about the last element to form the
        autoencoder. For example, if hidden_units is (256, 32, 16) then
        the autoencoder has layers

            [input_layer, 256, 32, 16, 32, 256, input_layer]

    batch_size : int, optional
        The batch size for learning and prediction. If there are fewer
        examples than the batch size during fitting, then the the number of
        examples will be used instead.
    n_epochs : int, optional
        The number of epochs (iterations through the training data) when
        fitting.
    dropout : float or None, optional
        The dropout probability. If None, then dropout will not be used.
        Note that dropout is not applied to the input layer.
    activation : (callable, callable), optional
        The activation functions for

            (the bulk of the autoencoder and encoding layer, output layer).

        See `tensorflow.nn` for various options.
    random_state: int, RandomState instance or None, optional
        If int, the random number generator seed. If RandomState instance,
        the random number generator itself. If None, then `np.random` will be
        used.
    learning_rate : float, optional
        Learning rate for Adam.
    metric : string, optional
        Default metric for the autoencoder. Options are

            'mse' - mean square error
            'cross-entropy' - cross-entropy

        Note that the 'cross-entropy' metric forces the activation of the
        output layer to be tensorflow.nn.sigmoid. Any dimensions indicated by
        the options `categorical_begin`, `categorical_size` or
        `discrete_indices` use a cross-entropy metric with
        `tensorflow.nn.softmax` or `tensorflow.nn.sigmoid` output layer
        activations.
    discrete_indices : array-like, 1d, optional
        Array of indices for which `tf.nn.sigmoid` will be used for the
        output layer activation and the cross-entropy will be used as the
        metric for the autoencoder.
    categorical_begin : array-like, shape (n_categorial), optional
        Array of the start of the one-hot encoded values for any
        categorically expanded variables. This forces the output layer to have
        activation `tensorflow.nn.softmax` for these variables and
        cross-entropy for metric for these variables.
        You must specify both `categorical_begin` and `categorical_size` if
        either is specified.
    categorical_size : array-like, shape (n_categorial), optional
        Array of the number of categories for any categorically expanded
        variables. This forces the output layer to have activation
        `tensorflow.nn.softmax` for these variables and cross-entropy for
        metric for these variables. You must specify both `categorical_begin`
        and `categorical_size` if either is specified.

    Attributes
    ----------
    graph_ : tensorflow.python.framework.ops.Graph
        The TensorFlow graph for the model.
    input_layer_sz_ : int
        The dimensionality of the input (i.e., number of features).
    discrete_indices_ : array-like, 1d or None
        Indices at which sigmoid activations were used, if any.
    categorical_begin_ : array-like, 1d or None
        Starting locations of one-hot encoded variables, if any.
    categorical_size_ : array-like, 1d or None
        Sizes of one-hot encoding of variables, if any.

    Methods
    -------
    fit : Fit the autoencoder.
    partial_fit : Update the fit of the autoencoder on a mini-batch.
    transform : Encoded input data.
    inverse_transform : Decode input data.
    score : Return the mean score for input data.
    score_samples : Return the score for each element in input data.

    Notes
    -----
    `Adam
    <https://www.tensorflow.org/versions/r0.8/api_docs/python/train.html#AdamOptimizer>`
    is used for optimization.
    Xavier initialization (
    `<https://www.tensorflow.org/versions/r0.8/api_docs/python/contrib.layers.html#xavier_initializer>`
    ) is used for the weights.
    """

    def __init__(self, hidden_units=(16,), batch_size=128, n_epochs=5,
                 dropout=None,
                 activation=(tf.nn.sigmoid, tf.nn.sigmoid),
                 random_state=None, learning_rate=1e-3,
                 metric='mse', discrete_indices=None, categorical_size=None,
                 categorical_begin=None):
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout = dropout
        self.activation = activation
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.metric = metric
        self.discrete_indices = discrete_indices
        self.categorical_begin = categorical_begin
        self.categorical_size = categorical_size

    def _set_up_graph(self):
        """Initialize TF objects (needed before fitting or restoring)."""

        # A placeholder to control dropout for training vs. prediction.
        self._dropout = tf.placeholder(dtype=tf.float32,
                                       shape=(),
                                       name="dropout")

        # Input values.
        self._input_values = tf.placeholder(tf.float32,
                                            [None, self.input_layer_sz_],
                                            "input_values")
        t = self._input_values

        # Fan in layers.
        for i, layer_sz in enumerate(self.hidden_units):
            # Don't dropout inputs.
            if i > 0:
                t = tf.nn.dropout(t, keep_prob=1.0 - self._dropout)
            t = affine(t, layer_sz, scope='layer_%d' % i)
            if self.activation and self.activation[0]:
                t = self.activation[0](t)

        # Encoded values.
        self._encoded_values = t

        # Fan out layers.
        second_layers \
            = list(self.hidden_units[::-1][1:]) + [self.input_layer_sz_]
        for i, layer_sz in enumerate(second_layers):
            t = tf.nn.dropout(t, keep_prob=1.0 - self._dropout)
            t = affine(t,
                       layer_sz,
                       scope='layer_%d' % (i + len(self.hidden_units)))
            if (i < len(second_layers) - 1 and
                    self.activation and
                    self.activation[0]):
                t = self.activation[0](t)

        # Finally do outputs and objective function.
        t, scores = self._build_output_layer_and_scores(t)
        self._output_values = t
        self._scores = scores
        self._obj_func = tf.reduce_mean(self._scores)

        # Training w/ Adam for now.
        self._train_step = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self._obj_func)

    def _build_output_layer_and_scores(self, t):
        """Add ops for output layer and scores to the graph."""
        scores = 0.0

        # Normalization factor for objective function. Is set to the total
        # number of variables below.
        num_vars_norm = 0.0

        # Below a transpose on `t` is needed if the TF ops `scatter_update` or
        # `gather` are used on the features dimension of `t` (i.e., dim 2).
        # A shortcut has been added here to avoid a transpose if it is not
        # needed.
        if (self.categorical_begin_ is None and
                self.categorical_size_ is None and
                self.discrete_indices_ is None):

            # Norm. factor should be the number of variables, which is just the
            # input layer size in this case.
            num_vars_norm = self.input_layer_sz_

            if self.metric == 'mse':
                if self.activation and self.activation[1]:
                    t = self.activation[1](t)
                diff = t - self._input_values
                scores = tf.reduce_sum(tf.square(diff),
                                       reduction_indices=[1])
            elif self.metric == 'cross-entropy':
                scores += tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        t, self._input_values),
                    reduction_indices=[1])
                t = tf.nn.sigmoid(t)
            else:
                raise ValueError('Metric "%s" is not allowed!' % self.metric)
        else:
            # The transpose since gather and scatter work on first dim.
            t = tf.transpose(t)

            # Indices to use for the default metric. Indices for categorical
            # or discrete var are removed.
            def_indices = set(range(self.input_layer_sz_))

            # Categorical vars w/ one-hot encoding.
            if self.categorical_begin_ and self.categorical_size_:
                # Use number of variables, not the total one-hot encoding
                # length for normalization.
                num_vars_norm += len(self.categorical_begin_)

                # Softmax all of the categorical stuff and use cross-entropy.
                for begin, size in zip(self.categorical_begin_,
                                       self.categorical_size_):
                    # Remove categorical stuff from def_indices.
                    def_indices -= set(range(begin, begin+size))

                    scores += tf.reduce_sum(
                        tf.nn.softmax_cross_entropy_with_logits(
                            tf.slice(t, [begin, 0], [size, -1]),
                            tf.slice(self._input_values,
                                     [begin, 0], [size, -1])),
                        reduction_indices=[0])

                    t = tf.scatter_update(
                        t,
                        list(range(begin, begin+size)),
                        tf.nn.softmax(tf.slice(t, [begin, 0], [size, -1])))

            # Discrete 0/1 stuff.
            if self.discrete_indices_:
                # Remove discrete stuff.
                def_indices -= set(self.discrete_indices_.tolist())
                num_vars_norm += len(self.discrete_indices_)

                # Sigmoid output w/ cross-entropy.
                tsub = tf.gather(t, self.discrete_indices_.tolist())
                isub = tf.gather(self._input_values,
                                 self.discrete_indices_.tolist())
                scores += tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(tsub, isub),
                    reduction_indices=[0])

                t = tf.scatter_update(t,
                                      self.discrete_indices_.tolist(),
                                      tf.nn.sigmoid(tsub))

            # Anything left uses the default metric.
            if def_indices:
                def_indices = sorted(list(def_indices))
                num_vars_norm += len(def_indices)

                if self.metric == 'mse':
                    if self.activation and self.activation[1]:
                        tsub = tf.gather(t, def_indices)
                        tsub = self.activation[1](tsub)
                        t = tf.scatter_update(t, def_indices, tsub)
                    else:
                        tsub = tf.gather(t, def_indices)
                    isub = tf.gather(self._input_values, def_indices)
                    diff = isub - tsub
                    scores += tf.reduce_sum(tf.square(diff),
                                            reduction_indices=[0]) / 2.0
                elif self.metric == 'cross-entropy':
                    tsub = tf.gather(t, def_indices)
                    isub = tf.gather(self._input_values, def_indices)
                    scores += tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(tsub, isub),
                        reduction_indices=[0])
                    t = tf.scatter_update(t, def_indices, tf.nn.sigmoid(tsub))
                else:
                    raise ValueError('Metric "%s" is not allowed!' %
                                     self.metric)

            # Undo the transpose here. Scores are transposed too.
            t = tf.transpose(t)
            scores = tf.transpose(scores)

        return t, scores / num_vars_norm

    def _make_feed_dict(self, X, inverse=False, training=False):
        # Make the dictionary mapping tensor placeholders to input data.

        # Convert sparse inputs to dense.
        if sp.issparse(X):
            X = X.todense().A

        if inverse:
            feed_dict = {self._encoded_values: X}
        else:
            feed_dict = {self._input_values: X}

        if not training:
            # If not training, fix the dropout to zero so keep_prob = 1.0.
            feed_dict[self._dropout] = 0.0
        else:
            feed_dict[self._dropout] = \
                self.dropout if self.dropout is not None else 0.0

        return feed_dict

    def _check_data(self, X):
        """check input data

        Raises an error if number of features doesn't match.

        If the estimator has not yet been fitted, then do nothing.
        """

        if self._is_fitted:
            if X.shape[1] != self.input_layer_sz_:
                raise ValueError("Number of features in the input data does "
                                 "not match the number assumed by the "
                                 "estimator!")

    def __getstate__(self):
        # Handles TF persistence
        state = super().__getstate__()

        # Add attributes of this estimator
        state.update(dict(activation=self.activation,
                          batch_size=self.batch_size,
                          dropout=self.dropout,
                          hidden_units=self.hidden_units,
                          random_state=self.random_state,
                          n_epochs=self.n_epochs,
                          learning_rate=self.learning_rate,
                          metric=self.metric,
                          discrete_indices=self.discrete_indices,
                          categorical_begin=self.categorical_begin,
                          categorical_size=self.categorical_size,
                          ))

        # Add fitted attributes if the model has been fitted.
        if self._is_fitted:
            state['input_layer_sz_'] = self.input_layer_sz_
            state['discrete_indices_'] = self.discrete_indices_
            state['categorical_begin_'] = self.categorical_begin_
            state['categorical_size_'] = self.categorical_size_

        return state

    def fit(self, X, y=None):
        """Fit the autoencoder.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Training data

        Returns
        -------
        self : returns an instance of self.
        """
        _LOGGER.info("Fitting %s", re.sub(r"\s+", r" ", repr(self)))

        # Mark the model as not fitted (i.e., not fully initialized based on
        # the data).
        self._is_fitted = False

        # Call partial fit, which will initialize and then train the model.
        return self.partial_fit(X)

    def partial_fit(self, X, y=None):
        """Fit the autoencoder on a batch of training data.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Training data

        Returns
        -------
        self : returns an instance of self.
        """

        # Check that the input X is an array or sparse matrix.
        # Each batch will be converted to dense on-the-fly.
        # For sparse input, make the input a CSR matrix since it can be
        # indexed by row.
        X = check_array(X, accept_sparse=['csr'])

        random_state = check_random_state(self.random_state)
        assert self.batch_size > 0, "batch_size <= 0"

        # Initialize the model if it hasn't been already by a previous call.
        if not self._is_fitted:
            self.input_layer_sz_ = X.shape[1]

            # Check and set categorical and discrete indices.
            self.discrete_indices_ = (np.atleast_1d(self.discrete_indices)
                                      if self.discrete_indices else None)

            if self.categorical_begin and self.categorical_size:
                self.categorical_begin_ = np.atleast_1d(self.categorical_begin)
                self.categorical_size_ = np.atleast_1d(self.categorical_size)

                sb = self.categorical_begin_.shape[0]
                ss = self.categorical_size_.shape[0]
                if sb != ss:
                    raise ValueError("categorical_begin and categorical_size "
                                     "must have the same shape!")
            else:
                self.categorical_size_ = None
                self.categorical_begin_ = None

            # Instantiate the graph.  TensorFlow seems easier to use by just
            # adding to the default graph, and as_default lets you temporarily
            # set a graph to be treated as the default graph.
            self.graph_ = tf.Graph()
            with self.graph_.as_default():
                tf.set_random_seed(random_state.randint(0, 10000000))

                tf.get_variable_scope().set_initializer(
                    tf.contrib.layers.xavier_initializer())

                self._build_tf_graph()

                # Train model parameters.
                self._session.run(tf.initialize_all_variables())

            # Set an attributed to mark this as at least partially fitted.
            self._is_fitted = True
        else:
            # Check input data against internal data.
            # Raises an error on failure.
            self._check_data(X)

        # Train the model with the given data.
        with self.graph_.as_default():
            n_examples = X.shape[0]
            start_idx = 0
            epoch = 0
            indices = np.arange(n_examples)
            random_state.shuffle(indices)

            while True:
                batch_ind = indices[start_idx:start_idx + self.batch_size]
                feed_dict = self._make_feed_dict(X[batch_ind], training=True)
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

    def transform(self, X, y=None):
        """Encode data with the autoencoder.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Data to encode

        Returns
        -------
        Z : numpy array
            Encoded data.
        """

        if not self._is_fitted:
            raise NotFittedError("Call fit before transform!")

        # For sparse input, make the input a CSR matrix since it can be
        # indexed by row.
        X = check_array(X, accept_sparse=['csr'])

        # Check input data against internal data.
        # Raises an error on failure.
        self._check_data(X)

        # Make predictions in batches.
        pred_batches = []
        start_idx = 0
        n_examples = X.shape[0]
        with self.graph_.as_default():
            while start_idx < n_examples:
                X_batch = \
                    X[start_idx:min(start_idx + self.batch_size, n_examples)]
                feed_dict = self._make_feed_dict(X_batch, training=False)
                start_idx += self.batch_size
                pred_batches.append(self._session.run(self._encoded_values,
                                                      feed_dict=feed_dict))
        return np.concatenate(pred_batches)

    def inverse_transform(self, X, y=None, return_sparse=False):
        """Decode data with the autoencoder.

        Note that transform and inverse_transform will in general not be exact
        inverses of each other for an autoencoder.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Encoded data to decode

        return_sparse : bool, optional
            If True, return a sparse matrix.

        Returns
        -------
        Z : numpy array
            Decoded data.
        """

        if not self._is_fitted:
            raise NotFittedError("Call fit before inverse_transform!")

        X = check_array(X)

        # Make predictions in batches.
        pred_batches = []
        start_idx = 0
        n_examples = X.shape[0]
        with self.graph_.as_default():
            while start_idx < n_examples:
                X_batch = \
                    X[start_idx:min(start_idx + self.batch_size, n_examples)]
                feed_dict = self._make_feed_dict(X_batch,
                                                 inverse=True,
                                                 training=False)
                start_idx += self.batch_size

                X_batch_pred = self._session.run(self._output_values,
                                                 feed_dict=feed_dict)

                if return_sparse:
                    X_batch_pred = sp.csr_matrix(X_batch_pred)

                pred_batches.append(X_batch_pred)

        if return_sparse:
            X_pred = sp.vstack(pred_batches)
        else:
            X_pred = np.concatenate(pred_batches)

        return X_pred

    def score(self, X, y=None):
        """Score the autoencoder with `X`.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Data to score the autoencoder with.

        Returns
        -------
        score : float
            The mean score over `X`.
        """
        return np.mean(self.score_samples(X))

    def score_samples(self, X, y=None):
        """Score the autoencoder on each element of `X`.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Data to score the autoencoder with.

        Returns
        -------
        scores : numpy array
            The score for each element of `X`.
        """

        if not self._is_fitted:
            raise NotFittedError("Call fit before transform!")

        # For sparse input, make the input a CSR matrix since it can be
        # indexed by row.
        X = check_array(X, accept_sparse=['csr'])

        # Check input data against internal data.
        # Raises an error on failure.
        self._check_data(X)

        # Make predictions in batches.
        scores = []
        start_idx = 0
        n_examples = X.shape[0]
        with self.graph_.as_default():
            while start_idx < n_examples:
                X_batch = \
                    X[start_idx:min(start_idx + self.batch_size, n_examples)]
                feed_dict = self._make_feed_dict(X_batch, training=False)
                start_idx += self.batch_size
                scores.append(self._session.run(self._scores,
                                                feed_dict=feed_dict))
        return np.concatenate(scores)
