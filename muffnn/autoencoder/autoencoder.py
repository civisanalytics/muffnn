"""
Autoencoder in scikit-learn style with TensorFlow
"""
import logging
import re
import warnings

import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import NotFittedError

from muffnn.core import TFPicklingBase, affine


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
        [input_layer_size, 256, 32, 16, 32, 256, input_layer_size].
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
    hidden_activation : tensorflow graph operation, optional
        The activation function for the hidden layers and encoding layer.
        See `tensorflow.nn` for various options.
    output_activation : tensorflow graph operation, optional
        The activation function for the output layer.
        See `tensorflow.nn` for various options.
        If `metric` is set to 'cross-entropy', then only
        `tensorflow.nn.sigmoid` or `tensorflow.nn.softmax` are valid
        options.
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
        Note that this will be overridden for columns specified by
        `binary_indices` or `categorical_indices`
    binary_indices : array-like, shape (n_binary,), optional
        Array of indices for which `tf.nn.sigmoid` will be used for the
        output layer activation and cross-entropy will be used in the loss
        function.
    categorical_indices : array-like, shape (n_categorical, 2), optional
        An array where each row specifies a range of indices for a categorical
        variable that has been expanded to multiple indices (e.g.,
        one-hot encoded).
        The first column contains start indices. The second column contains
        lengths. For each range of indices, the output layer will use
        `tensorflow.nn.softmax` activation and the loss function will use
        cross-entropy.

    Attributes
    ----------
    graph_ : tensorflow.python.framework.ops.Graph
        The TensorFlow graph for the model.
    input_layer_size_ : int
        The dimensionality of the input (i.e., number of features).

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
                 dropout=None, hidden_activation=tf.nn.sigmoid,
                 output_activation=tf.nn.sigmoid, random_state=None,
                 learning_rate=1e-3, metric='mse', binary_indices=None,
                 categorical_indices=None):
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout = dropout
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.metric = metric
        self.binary_indices = binary_indices
        self.categorical_indices = categorical_indices

    def _set_up_graph(self):
        """Initialize TF objects (needed before fitting or restoring)."""

        # A placeholder to control dropout for training vs. prediction.
        self._dropout = tf.placeholder(dtype=tf.float32,
                                       shape=(),
                                       name="dropout")

        # Input values.
        self._input_values = tf.placeholder(tf.float32,
                                            [None, self.input_layer_size_],
                                            "input_values")
        t = self._input_values

        # These masks are for construction the mixed metric output layer and
        # scores. TensorFlow does not support scatter operations into Tesnors
        # (i.e., the results of TF graph operations). Thus we use masks to
        # place the right data in the right spot.
        # The masks are `1` where the data should be and `0` otherwise.
        # Thus you can do things like
        # `data_new = msk * tf.exp(data) + (1 - msk) * data`
        # to compute `tf.exp` of just a part of the data where the masks
        # are `1`.
        self._default_msk = tf.placeholder(tf.float32,
                                           [None, self.input_layer_size_],
                                           "default_msk")

        self._binary_msk = tf.placeholder(tf.float32,
                                          [None, self.input_layer_size_],
                                          "binary_msk")

        self._categorical_msks = tf.placeholder(
            tf.float32,
            [None, None, self.input_layer_size_],
            "categorical_msks")

        # Fan in layers.
        for i, layer_sz in enumerate(self.hidden_units):
            # Don't dropout inputs.
            if i > 0:
                t = tf.nn.dropout(t, keep_prob=1.0 - self._dropout)
            t = affine(t, layer_sz, scope='layer_%d' % i)
            if self.hidden_activation is not None:
                t = self.hidden_activation(t)

        # Encoded values.
        self._encoded_values = t

        # Fan out layers.
        second_layers \
            = list(self.hidden_units[::-1][1:]) + [self.input_layer_size_]
        for i, layer_sz in enumerate(second_layers):
            t = tf.nn.dropout(t, keep_prob=1.0 - self._dropout)
            t = affine(t,
                       layer_sz,
                       scope='layer_%d' % (i + len(self.hidden_units)))
            if (i < len(second_layers) - 1 and
                    self.hidden_activation is not None):
                t = self.hidden_activation(t)

        # Finally do outputs and objective function.
        self._output_values, self._scores \
            = self._build_output_layer_and_scores(t)
        self._obj_func = tf.reduce_mean(self._scores)

        # Training w/ Adam for now.
        # Catching a warning related to TensorFlow sparse to dense conversions
        # from the graph ops for the scores for mixed metrics:
        # '.../tensorflow/python/ops/gradients.py:90: UserWarning: Converting
        #  sparse IndexedSlices to a dense Tensor of unknown shape. This may
        #  consume a large amount of memory.
        #  "Converting sparse IndexedSlices to a dense Tensor of unknown
        #  shape."'
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=("Converting sparse IndexedSlices to a dense Tensor "
                         "of unknown shape"),
                module='tensorflow')
            self._train_step = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(self._obj_func)

    def _build_output_layer_and_scores(self, t):
        """Add ops for output layer and scores to the graph.

        Here `t` is the output layer before the output activation has been
        applied.
        """
        scores = 0.0

        # The first `if` case here is for a single output variable type
        # and metric. The `else` is for mixed output types and metrics.
        if (self._categorical_indices is None and
                self._binary_indices is None):

            if self.metric == 'mse':
                if self.output_activation is not None:
                    t = self.output_activation(t)

                diff = t - self._input_values
                scores = tf.reduce_sum(tf.square(diff),
                                       reduction_indices=[1])
            elif self.metric == 'cross-entropy':
                if self.output_activation is tf.nn.sigmoid:
                    scores += tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            t, self._input_values),
                        reduction_indices=[1])
                    t = tf.nn.sigmoid(t)
                elif self.output_activation is tf.nn.softmax:
                    scores += tf.nn.softmax_cross_entropy_with_logits(
                        t, self._input_values, dim=1)
                    t = tf.nn.softmax(t)
                else:
                    raise ValueError("Only `tensorflow.nn.sigmoid` and "
                                     "`tensorflow.nn.softmax` output "
                                     "activations can be used with the "
                                     "'cross-entropy' metric!")
            else:
                raise ValueError('Metric "%s" is not allowed!' % self.metric)
        else:
            # A transpose on `t` is needed if the TF op `gather` is used
            # on the features dimension of `t` (i.e., dim 2) since
            # gather works on first dim in tensorflow.
            t = tf.transpose(t)

            # Note that the code below uses the masks denoting where the
            # variables of each type are stored. This allows tensorflow to
            # compute the proper outputs and scores for each variable. We are
            # using masks here because we cannot scatter into `Tensors` (i.e,
            # we cannot use a tensorflow scatter operation on the result of a
            # tnesorflow operation).

            # Categorical vars w/ one-hot encoding.
            # Softmax all of the categorical stuff and use cross-entropy.
            if self._categorical_indices is not None:
                min_float32 = tf.constant(tf.float32.min)
                for i, begin, size in zip(
                        range(self._categorical_indices.shape[0]),
                        self._categorical_indices[:, 0],
                        self._categorical_indices[:, 1]):
                    scores += tf.nn.softmax_cross_entropy_with_logits(
                        tf.slice(t, [begin, 0], [size, -1]),
                        tf.slice(tf.transpose(self._input_values),
                                 [begin, 0], [size, -1]),
                        dim=0)

                    # This one is painful. TensorFlow does not, AFAIK,
                    # support assignments to Tensors that come out of
                    # operations (it does support assignments to Variables).
                    # So I am using a precomputed mask to to update certain
                    # values. Because softmax has a sumexp operation,
                    # you have to set the elements not updated to logits
                    # equal to the most negative float. Then they come out
                    # to zero after the exp and so fall out of the sumexp.
                    msk = tf.transpose(self._categorical_msks[i, :, :])
                    softmax = tf.nn.softmax(
                        msk * t + (1.0 - msk) * min_float32,
                        dim=0)
                    t = (1.0 - msk) * t + msk * softmax

            # Discrete 0/1 stuff.
            # Sigmoid output w/ cross-entropy.
            if self._binary_indices is not None:
                tsub = tf.gather(t, self._binary_indices)
                isub = tf.gather(tf.transpose(self._input_values),
                                 self._binary_indices)
                scores += tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(tsub, isub),
                    reduction_indices=[0])

                binary_msk = tf.transpose(self._binary_msk)
                t = ((1.0 - binary_msk) * t +
                     binary_msk * tf.nn.sigmoid(t))

            # Anything left uses the default metric.
            if self._default_indices is not None:
                default_msk = tf.transpose(self._default_msk)

                if self.metric == 'mse':
                    if self.output_activation is not None:
                        t = ((1.0 - default_msk) * t +
                             default_msk * self.output_activation(t))
                    tsub = tf.gather(t, self._default_indices)
                    isub = tf.gather(tf.transpose(self._input_values),
                                     self._default_indices)
                    diff = isub - tsub
                    scores += tf.reduce_sum(tf.square(diff),
                                            reduction_indices=[0])
                elif self.metric == 'cross-entropy':
                    tsub = tf.gather(t, self._default_indices)
                    isub = tf.gather(tf.transpose(self._input_values),
                                     self._default_indices)

                    if self.output_activation is tf.nn.sigmoid:
                        scores += tf.reduce_sum(
                            tf.nn.sigmoid_cross_entropy_with_logits(tsub,
                                                                    isub),
                            reduction_indices=[0])
                        t = ((1.0 - default_msk) * t +
                             default_msk * tf.nn.sigmoid(t))
                    elif self.output_activation is tf.nn.softmax:
                        # More pain here. The softmax has an implicit sum,
                        # and since we cannot scatter into Tensors, we
                        # break up the comp into the denominator and the
                        # numerator using the mask.
                        scores += tf.nn.softmax_cross_entropy_with_logits(
                            tsub, isub, dim=0)
                        log_softmax_denom = tf.reduce_logsumexp(
                            tsub, reduction_indices=[0])
                        t = ((1.0 - default_msk) * t +
                             default_msk * tf.exp(t - log_softmax_denom))
                    else:
                        raise ValueError("Only `tensorflow.nn.sigmoid` and "
                                         "`tensorflow.nn.softmax` output "
                                         "activations can be used with the "
                                         "'cross-entropy' metric!")
                else:
                    raise ValueError('Metric "%s" is not allowed!' %
                                     self.metric)

            # Undo the transpose here.
            t = tf.transpose(t)

        return t, scores

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

        feed_dict[self._binary_msk] \
            = self._binary_msk_values[0:X.shape[0], :]

        feed_dict[self._default_msk] \
            = self._default_msk_values[0:X.shape[0], :]

        feed_dict[self._categorical_msks] \
            = self._categorical_msks_values[:, 0:X.shape[0], :]

        return feed_dict

    def _check_data(self, X):
        """check input data

        Raises an error if number of features doesn't match.

        If the estimator has not yet been fitted, then do nothing.
        """

        if self._is_fitted:
            if X.shape[1] != self.input_layer_size_:
                raise ValueError("Number of features in the input data does "
                                 "not match the number assumed by the "
                                 "estimator!")

    def __getstate__(self):
        # Handles TF persistence
        state = super().__getstate__()

        # Add attributes of this estimator
        state.update(dict(hidden_activation=self.hidden_activation,
                          output_activation=self.output_activation,
                          batch_size=self.batch_size,
                          dropout=self.dropout,
                          hidden_units=self.hidden_units,
                          random_state=self.random_state,
                          n_epochs=self.n_epochs,
                          learning_rate=self.learning_rate,
                          metric=self.metric,
                          binary_indices=self.binary_indices,
                          categorical_indices=self.categorical_indices,
                          ))

        # Add fitted attributes if the model has been fitted.
        if self._is_fitted:
            state['input_layer_size_'] = self.input_layer_size_
            state['_binary_indices'] = self._binary_indices
            state['_categorical_indices'] = self._categorical_indices
            state['_binary_msk_values'] = self._binary_msk_values
            state['_default_msk_values'] = self._default_msk_values
            state['_categorical_msks_values'] = self._categorical_msks_values
            state['_random_state'] = self._random_state

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

        # Initialize the model if it hasn't been already by a previous call.
        if not self._is_fitted:
            self._random_state = check_random_state(self.random_state)
            assert self.batch_size > 0, "batch_size <= 0"

            self.input_layer_size_ = X.shape[1]

            # Indices for default metric, used for updates in metric.
            def_indices = set(range(X.shape[1]))

            # Check and set categorical and discrete indices.
            self._binary_indices = (np.atleast_1d(self.binary_indices)
                                    if self.binary_indices is not None
                                    else None)

            self._binary_msk_values \
                = np.zeros((self.batch_size, self.input_layer_size_))
            if self._binary_indices is not None:
                def_indices -= set(self._binary_indices.tolist())
                self._binary_msk_values[:, self._binary_indices] = 1

            if self.categorical_indices is not None:
                self._categorical_indices \
                    = np.atleast_2d(self.categorical_indices)

                self._categorical_msks_values = []
                for begin, size in zip(self._categorical_indices[:, 0],
                                       self._categorical_indices[:, 1]):
                    # Remove categorical stuff from def_indices.
                    def_indices -= set(range(begin, begin + size))

                    # Keep track of the mask.
                    msk = np.zeros((self.batch_size, self.input_layer_size_))
                    msk[:, range(begin, begin + size)] = 1
                    self._categorical_msks_values.append(msk)

                self._categorical_msks_values \
                    = np.array(self._categorical_msks_values)
            else:
                self._categorical_indices = None
                self._categorical_msks_values \
                    = np.empty((1, self.batch_size, self.input_layer_size_))

            # Finally set the default indices and mask.
            self._default_indices = np.array(list(def_indices), dtype=int)
            self._default_msk_values \
                = np.zeros((self.batch_size, self.input_layer_size_))
            if len(self._default_indices) > 0:
                self._default_msk_values[:, self._default_indices] = 1
            else:
                self._default_indices = None

            # Instantiate the graph.  TensorFlow seems easier to use by just
            # adding to the default graph, and as_default lets you temporarily
            # set a graph to be treated as the default graph.
            self.graph_ = tf.Graph()
            with self.graph_.as_default():
                tf.set_random_seed(self._random_state.randint(0, 10000000))

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
            self._random_state.shuffle(indices)

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
                    self._random_state.shuffle(indices)

        return self

    def transform(self, X, y=None):
        """Encode data with the autoencoder.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Data to encode

        Returns
        -------
        numpy array of shape [n_samples, hidden_units[-1]]
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

        This method can be useful if further modeling on the encoded data
        generates new encoded data vectors (e.g., generating new data by
        fitting a density estimate to the encoded data).

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, hidden_units[-1]]
            Encoded data to decode. Input shape should

        return_sparse : bool, optional
            If True, return a sparse matrix.

        Returns
        -------
        numpy array of shape [n_samples, n_features]
            Decoded data.
        """

        if not self._is_fitted:
            raise NotFittedError("Call fit before inverse_transform!")

        X = check_array(X)

        if X.shape[1] != self._encoded_values.get_shape()[1]:
            raise ValueError("Number of features in the encoded data does "
                             "not match the number assumed by the "
                             "estimator!")

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
