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
from sklearn.exceptions import NotFittedError

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
    keep_prob : float, optional
        The probability of keeping values in dropout. A value of 1.0 means that
        dropout will not be used. cf. `TensorFlow documentation
        <https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#dropout>`
    hidden_activation : tensorflow graph operation, optional
        The activation function for the hidden layers.
        See `tensorflow.nn` for various options.
        None is equivalent to a linear activation.
    encoding_activation : tensorflow graph operation, optional
        The activation function for the encoding layer.
        See `tensorflow.nn` for various options.
        None is equivalent to a linear activation.
    output_activation : tensorflow graph operation, optional
        The activation function for the output layer.
        See `tensorflow.nn` for various options.
        None is equivalent to a linear activation.
        If `loss` is set to 'cross-entropy', then only
        `tensorflow.nn.sigmoid` or `tensorflow.nn.softmax` are valid
        options.
    random_state: int, RandomState instance or None, optional
        If int, the random number generator seed. If RandomState instance,
        the random number generator itself. If None, then `np.random` will be
        used.
    learning_rate : float, optional
        Learning rate for Adam.
    loss : string, optional
        Default loss for the autoencoder. Options are
            'mse' - mean square error
            'cross-entropy' - cross-entropy
        Note that this will be overridden for columns specified by
        `sigmoid_indices` or `softmax_indices`
    sigmoid_indices : array-like, shape (n_sigmoid,), optional
        Array of indices for which `tf.nn.sigmoid` will be used for the
        output layer activation and cross-entropy will be used in the loss
        function.
    softmax_indices : array-like, shape (n_softmax, 2), optional
        An array where each row specifies a range of indices for a categorical
        variable that has been expanded to multiple indices (e.g.,
        one-hot encoded) or set of variables which should sum to unity.
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
    If your data is of all one type (e.g., all binary 0/1 labels or a single
    set of variables that always sum to unity), then you can directly change
    the output activation and loss as opposed to using the `sigmoid_indices`
    or `softmax_indices` keywords. These keywords are for data which has
    mixed type (e.g., one wants some data to be treated with the MSE loss
    while using a cross-entropy loss elsewhere.)

    `Adam
    <https://www.tensorflow.org/versions/r0.8/api_docs/python/train.html#AdamOptimizer>`
    is used for optimization.

    Xavier initialization (
    `<https://www.tensorflow.org/versions/r0.8/api_docs/python/contrib.layers.html#xavier_initializer>`
    ) is used for the weights.
    """

    def __init__(self, hidden_units=(16,), batch_size=128, n_epochs=5,
                 keep_prob=1.0, hidden_activation=tf.nn.relu,
                 encoding_activation=None,
                 output_activation=None, random_state=None,
                 learning_rate=1e-3, loss='mse', sigmoid_indices=None,
                 softmax_indices=None):
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.keep_prob = keep_prob
        self.hidden_activation = hidden_activation
        self.encoding_activation = encoding_activation
        self.output_activation = output_activation
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.loss = loss
        self.sigmoid_indices = sigmoid_indices
        self.softmax_indices = softmax_indices

    def _set_up_graph(self):
        """Initialize TF objects (needed before fitting or restoring)."""

        # A placeholder to control dropout for training vs. prediction.
        self._keep_prob = tf.placeholder(dtype=tf.float32,
                                         shape=(),
                                         name="keep_prob")

        # Input values.
        self._input_values = tf.placeholder(tf.float32,
                                            [None, self.input_layer_size_],
                                            "input_values")
        t = self._input_values

        # These masks are for construction the mixed loss output layer and
        # scores. TensorFlow does not support scatter operations into Tesnors
        # (i.e., the results of TF graph operations). Thus we use masks to
        # place the right data in the right spot.
        # The masks are type `tf.bool` to be used with `tf.where`.
        self._default_msk = tf.placeholder(tf.bool,
                                           [None, self.input_layer_size_],
                                           "default_msk")

        self._sigmoid_msk = tf.placeholder(tf.bool,
                                           [None, self.input_layer_size_],
                                           "sigmoid_msk")

        self._softmax_msks = tf.placeholder(
            tf.bool,
            [None, None, self.input_layer_size_],
            "softmax_msks")

        # Fan in layers.
        for i, layer_sz in enumerate(self.hidden_units):
            if self.keep_prob != 1.0:
                t = tf.nn.dropout(t, keep_prob=self._keep_prob)
            t = affine(t, layer_sz, scope='layer_%d' % i)
            if (self.hidden_activation is not None and
                    i < len(self.hidden_units) - 1):
                t = self.hidden_activation(t)
            if (self.encoding_activation is not None and
                    i == len(self.hidden_units) - 1):
                t = self.encoding_activation(t)

        # Encoded values.
        self._encoded_values = t

        # Fan out layers.
        second_layers \
            = list(self.hidden_units[::-1][1:]) + [self.input_layer_size_]
        for i, layer_sz in enumerate(second_layers):
            if self.keep_prob != 1.0:
                t = tf.nn.dropout(t, keep_prob=self._keep_prob)
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
        # from the graph ops for the scores for mixed losses:
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
        # The first `if` case here is for a single output variable type
        # and loss. The `else` is for mixed output types and losses.
        if (self._softmax_indices is None and self._sigmoid_indices is None):
            return self._build_one_type_output_layer(t)
        else:
            return self._build_mixed_type_output_layer(t)

    def _build_one_type_output_layer(self, t):
        """Add ops for output layer and scores to the graph for
        one kind of output.

        Here `t` is the output layer before the output activation has been
        applied.
        """

        if self.loss == 'mse':
            if self.output_activation is not None:
                t = self.output_activation(t)
            diff = t - self._input_values
            scores = tf.reduce_sum(tf.square(diff), axis=1)
        elif self.loss == 'cross-entropy':
            if self.output_activation is tf.nn.sigmoid:
                scores = tf.reduce_sum(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=t, labels=self._input_values), axis=1)
                t = tf.nn.sigmoid(t)
            elif self.output_activation is tf.nn.softmax:
                scores = tf.nn.softmax_cross_entropy_with_logits(
                    logits=t, labels=self._input_values, dim=1)
                t = tf.nn.softmax(t)
            else:
                raise ValueError("Only `tensorflow.nn.sigmoid` and "
                                 "`tensorflow.nn.softmax` output "
                                 "activations can be used with the "
                                 "'cross-entropy' loss!")
        else:
            raise ValueError('Loss "%s" is not allowed!' % self.loss)

        return t, scores

    def _build_mixed_type_output_layer(self, t):
        """Add ops for output layer and scores to the graph for
        mixed outputs.

        Here `t` is the output layer before the output activation has been
        applied.
        """
        scores = 0.0

        # A transpose on `t` is needed if the TF op `gather` is used
        # on the features dimension of `t` (i.e., dim 2) since
        # gather works on first dim in tensorflow.
        t = tf.transpose(t)

        # Note that the code below uses the masks denoting where the
        # variables of each type are stored. This allows tensorflow to
        # compute the proper outputs and scores for each variable. We are
        # using masks here because we cannot scatter into `Tensors` (i.e,
        # we cannot use a tensorflow scatter operation on the result of a
        # tensorflow operation).

        # Softmax vars (will sum to unity) w/ cross-entropy.
        if self._softmax_indices is not None:
            for i, begin, size in zip(
                    range(self._softmax_indices.shape[0]),
                    self._softmax_indices[:, 0],
                    self._softmax_indices[:, 1]):
                tsub = tf.slice(t, [begin, 0], [size, -1])
                scores += tf.nn.softmax_cross_entropy_with_logits(
                    logits=tsub,
                    labels=tf.slice(tf.transpose(self._input_values),
                                    [begin, 0], [size, -1]),
                    dim=0)

                # This one is painful. TensorFlow does not, AFAIK,
                # support assignments to Tensors that come out of
                # operations (it does support assignments to Variables).
                # So I am using a precomputed mask to to update certain
                # values. Because softmax has a sumexp operation,
                # I split the softmax in two, doing the sumexp on
                # the proper subset of the elements.
                msk = tf.transpose(self._softmax_msks[i, :, :])
                log_softmax_denom = tf.reduce_logsumexp(
                    tsub, keep_dims=True, axis=0)
                t = tf.where(msk, tf.exp(t - log_softmax_denom), t)

        # Sigmoid output w/ cross-entropy.
        if self._sigmoid_indices is not None:
            tsub = tf.gather(t, self._sigmoid_indices)
            isub = tf.gather(tf.transpose(self._input_values),
                             self._sigmoid_indices)
            scores += tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=tsub,
                                                        labels=isub),
                axis=0)
            sigmoid_msk = tf.transpose(self._sigmoid_msk)
            t = tf.where(sigmoid_msk, tf.nn.sigmoid(t), t)

        # Anything left uses the default loss.
        if self._default_indices is not None:
            default_msk = tf.transpose(self._default_msk)

            if self.loss == 'mse':
                if self.output_activation is not None:
                    t = tf.where(default_msk, self.output_activation(t), t)
                tsub = tf.gather(t, self._default_indices)
                isub = tf.gather(tf.transpose(self._input_values),
                                 self._default_indices)
                diff = isub - tsub
                scores += tf.reduce_sum(tf.square(diff), axis=0)
            elif self.loss == 'cross-entropy':
                tsub = tf.gather(t, self._default_indices)
                isub = tf.gather(tf.transpose(self._input_values),
                                 self._default_indices)

                if self.output_activation is tf.nn.sigmoid:
                    scores += tf.reduce_sum(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=tsub,
                                                                labels=isub),
                        axis=0)
                    t = tf.where(default_msk, tf.nn.sigmoid(t), t)
                elif self.output_activation is tf.nn.softmax:
                    # More pain here. The softmax has an implicit sum,
                    # and since we cannot scatter into Tensors, we
                    # break up the comp into the denominator and the
                    # numerator using the mask.
                    scores += tf.nn.softmax_cross_entropy_with_logits(
                        logits=tsub, labels=isub, dim=0)
                    log_softmax_denom = tf.reduce_logsumexp(
                        tsub, axis=0, keep_dims=True)
                    t = tf.where(
                        default_msk, tf.exp(t - log_softmax_denom), t)
                else:
                    raise ValueError("Only `tensorflow.nn.sigmoid` and "
                                     "`tensorflow.nn.softmax` output "
                                     "activations can be used with the "
                                     "'cross-entropy' loss!")
            else:
                raise ValueError('Loss "%s" is not allowed!' %
                                 self.loss)

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

        # If not training, turn off dropout (i.e., set keep_prob = 1.0).
        feed_dict[self._keep_prob] = self.keep_prob if training else 1.0

        feed_dict[self._sigmoid_msk] \
            = self._sigmoid_msk_values[0:X.shape[0], :]

        feed_dict[self._default_msk] \
            = self._default_msk_values[0:X.shape[0], :]

        feed_dict[self._softmax_msks] \
            = self._softmax_msks_values[:, 0:X.shape[0], :]

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
        state = super(Autoencoder, self).__getstate__()

        # Add attributes of this estimator
        state.update(dict(hidden_activation=self.hidden_activation,
                          encoding_activation=self.encoding_activation,
                          output_activation=self.output_activation,
                          batch_size=self.batch_size,
                          keep_prob=self.keep_prob,
                          hidden_units=self.hidden_units,
                          random_state=self.random_state,
                          n_epochs=self.n_epochs,
                          learning_rate=self.learning_rate,
                          loss=self.loss,
                          sigmoid_indices=self.sigmoid_indices,
                          softmax_indices=self.softmax_indices,
                          ))

        # Add fitted attributes if the model has been fitted.
        if self._is_fitted:
            state.update(dict(input_layer_size_=self.input_layer_size_,
                              _sigmoid_indices=self._sigmoid_indices,
                              _softmax_indices=self._softmax_indices,
                              _sigmoid_msk_values=self._sigmoid_msk_values,
                              _default_msk_values=self._default_msk_values,
                              _softmax_msks_values=self._softmax_msks_values,
                              _random_state=self._random_state))

        return state

    def _build_sigmoid_and_softmax_indices(self):
        """Construct and error check sigmoid and softmax indices.

        Raises an error if any entry in the feature space is marked as both
        sigmoid and softmax.
        """
        # Indices for default loss, used for updates in building
        # the output layer and loss.
        def_indices = set(range(self.input_layer_size_))

        # Sigmoid indices.
        self._sigmoid_indices = (np.atleast_1d(self.sigmoid_indices)
                                 if self.sigmoid_indices is not None
                                 else None)
        self._sigmoid_msk_values \
            = np.zeros((self.batch_size, self.input_layer_size_))
        if self._sigmoid_indices is not None:
            def_indices -= set(self._sigmoid_indices.tolist())
            self._sigmoid_msk_values[:, self._sigmoid_indices] = 1
        self._sigmoid_msk_values = self._sigmoid_msk_values.astype(bool)

        # Softmax indices.
        if self.softmax_indices is not None:
            self._softmax_indices \
                = np.atleast_2d(self.softmax_indices)

            self._softmax_msks_values = []
            for begin, size in zip(self._softmax_indices[:, 0],
                                   self._softmax_indices[:, 1]):
                # Remove softmax stuff from def_indices.
                def_indices -= set(range(begin, begin + size))

                # Keep track of the mask.
                msk = np.zeros((self.batch_size, self.input_layer_size_))
                msk[:, range(begin, begin + size)] = 1
                self._softmax_msks_values.append(msk)

            self._softmax_msks_values \
                = np.array(self._softmax_msks_values)
        else:
            self._softmax_indices = None
            self._softmax_msks_values \
                = np.empty((1, self.batch_size, self.input_layer_size_))
        self._softmax_msks_values \
            = self._softmax_msks_values.astype(bool)

        # Default indices.
        self._default_indices = np.array(list(def_indices), dtype=int)
        self._default_msk_values \
            = np.zeros((self.batch_size, self.input_layer_size_))
        if len(self._default_indices) > 0:
            self._default_msk_values[:, self._default_indices] = 1
        else:
            self._default_indices = None
        self._default_msk_values = self._default_msk_values.astype(bool)

        # Error check the results.
        # We rebuild the sets here to error check the code above too.
        # Always play good defense.
        sigmoid_set = set()
        if self._sigmoid_indices is not None:
            sigmoid_set = set(self._sigmoid_indices.tolist())

        softmax_set = set()
        if self._softmax_indices is not None:
            for begin, size in zip(self._softmax_indices[:, 0],
                                   self._softmax_indices[:, 1]):
                softmax_set |= set(range(begin, begin + size))

        if not sigmoid_set.isdisjoint(softmax_set):
            raise ValueError("Sigmoid indices and softmax indices cannot"
                             " overlap!")

    def partial_fit(self, X, y=None, monitor=None, **kwargs):
        """Fit the autoencoder on a batch of training data.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Training data
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

            self._build_sigmoid_and_softmax_indices()

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
                self._session.run(tf.global_variables_initializer())

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
                batch_ind = indices[
                    start_idx:min(start_idx + self.batch_size, n_examples)]
                feed_dict = self._make_feed_dict(X[batch_ind], training=True)
                obj_val, _ = self._session.run(
                    [self._obj_func, self._train_step], feed_dict=feed_dict)
                _LOGGER.debug(
                    "objective: %.4f, epoch: %d, idx: %d",
                    obj_val, epoch, start_idx)

                start_idx += self.batch_size

                # are we at the end of an epoch?
                if start_idx > n_examples - self.batch_size:
                    _LOGGER.info(
                        "objective: %.4f, epoch: %d, idx: %d",
                        obj_val, epoch, start_idx)

                    if monitor:
                        stop_early = monitor(epoch, self, {'loss': obj_val})
                        if stop_early:
                            _LOGGER.info(
                                "stopping early due to monitor function.")
                            return self

                    start_idx = 0
                    epoch += 1
                    if epoch >= self.n_epochs:
                        break
                    self._random_state.shuffle(indices)

        return self

    def fit(self, X, y=None, monitor=None, **kwargs):
        """Fit the autoencoder.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Training data
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
        return self.partial_fit(X, monitor=monitor, **kwargs)

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
