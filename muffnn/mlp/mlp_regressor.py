"""
A Deep Neural Network (multilayer Perceptron) sklearn-style regressor.

Similar to sklearn.neural_network.MLPRegressor, but using TensorFlow.
"""

import logging
from warnings import warn

import numpy as np
from sklearn.base import RegressorMixin

import tensorflow as tf
from tensorflow.python.ops import nn
from muffnn.mlp.base import MLPBaseEstimator
from muffnn.core import affine


_LOGGER = logging.getLogger(__name__)


class MLPRegressor(MLPBaseEstimator, RegressorMixin):
    """
    A deep neural network (multilayer perceptron) regressor using TensorFlow.

    Parameters
    ----------
    hidden_units : tuple or list, optional
        A list of integers indicating the number of hidden layers and their
        sizes.
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
    activation : callable, optional
        The activation function.  See tensorflow.python.ops.nn. Setting this to
        tf.nn.selu will also cause alpha dropout to be used, implementing a
        Self-Normalizing Neural Network (Klambauer et al., 2017).
    random_state : int, RandomState instance or None, optional
        If int, the random number generator seed. If RandomState instance,
        the random number generator itself. If None, then `np.random` will be
        used.
    solver : a subclass of `tf.train.Optimizer`, optional
        The solver to use to minimize the loss.
    solver_kwargs : dict, optional
        Additional keyword arguments to pass to `solver` upon construction.
        See the TensorFlow documentation for possible options. Typically,
        one would want to set the `learning_rate`.
    transform_layer_index : int, optional
        The index of the hidden layer to use to transform inputs. If not given,
        it defaults to the last hidden layer or output logits in the case that
        no hidden layers are used.

    Attributes
    ----------
    input_layer_sz_ : int
        The dimensionality of the input (i.e., number of features).
    is_sparse_ : bool
        Whether a model taking sparse input was fit.
    target_mean_ : float
        The mean of the target values passed to the `fit` method.
    target_sd_ : float
        The standard deviation of the target values passed to the `fit` method.
    graph_ : tensorflow.python.framework.ops.Graph
        The TensorFlow graph for the model

    Notes
    -----
    The fitted mean and standard deviations for the targets make training
    easier since the fitting algorithm won't have to learn them.

    There is currently no dropout between the sparse input layer and first
    hidden layer. Dropout on the sparse input layer would undo the benefits of
    sparsity because the dropout layer is dense.
    """

    def __init__(self, hidden_units=(256,), batch_size=64, n_epochs=5,
                 keep_prob=1.0, activation=nn.relu,
                 random_state=None, solver=tf.train.AdamOptimizer,
                 solver_kwargs=None, transform_layer_index=None):
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.keep_prob = keep_prob
        self.activation = activation
        self.random_state = random_state
        self.solver = solver
        self.solver_kwargs = solver_kwargs
        self.transform_layer_index = transform_layer_index

    def _init_model_output(self, t):
        if self.is_sparse_ and not self.hidden_units:
            t = affine(t, 1, input_size=self.input_layer_sz_,
                       scope='output_layer', sparse_input=True)
        else:
            if self.keep_prob != 1.0:
                if self.activation is tf.nn.selu:
                    t = tf.contrib.nn.alpha_dropout(
                        t, keep_prob=self._keep_prob)
                else:
                    t = tf.nn.dropout(t, keep_prob=self._keep_prob)
            t = affine(t, 1, scope='output_layer')

        self.input_targets_ = tf.placeholder(tf.float32, [None], "targets")
        t = tf.reshape(t, [-1])  # Convert to 1d tensor.
        self.output_layer_ = t

        return t

    def _transform_targets(self, y):
        # Standardize the targets for fitting, and store the M and SD values
        # for prediction.

        y_centered = y - self.target_mean_

        if self.target_sd_ <= 0:
            return y_centered

        return y_centered / self.target_sd_

    def _fit_targets(self, y):
        # Store the mean and S.D. of the targets so we can have standardized
        # y for training but still make predictions on the original scale.

        self.target_mean_ = np.mean(y)

        self.target_sd_ = np.std(y - self.target_mean_)
        if self.target_sd_ <= 0:
            warn("No variance in regression targets.")

    def _init_model_objective_fn(self, t):
        mse = (self.input_targets_ - t) ** 2
        self._obj_func = tf.divide(
            tf.reduce_sum(
                tf.multiply(mse, self._sample_weight)),
            tf.reduce_sum(self._sample_weight))

    def predict(self, X):
        """Make predictions for the given input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Examples to make predictions about.

        Returns
        -------
        C : array, shape = (n_samples,)
            Predicted values
        """
        y_pred = self._compute_output(X)

        # Put the prediction back on the scale of the target values
        # (cf. _transform_targets).
        if self.target_sd_ > 0.0:
            y_pred *= self.target_sd_
        y_pred += self.target_mean_

        return y_pred

    def __getstate__(self):
        state = super(MLPRegressor, self).__getstate__()

        # Add the fitted attributes particular to this subclass.
        if self._is_fitted:
            state['target_mean_'] = self.target_mean_
            state['target_sd_'] = self.target_sd_

        return state
