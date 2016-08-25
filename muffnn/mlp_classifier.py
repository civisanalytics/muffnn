#!/usr/bin/env python3

"""
A Deep Neural Network (multilayer Perceptron) sklearn-style classifier.

Similar to sklearn.neural_network.MLPClassifier, but using TensorFlow.
"""

import logging

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import type_of_target

import tensorflow as tf
from tensorflow.python.ops import nn
from muffnn.base import MLPBaseEstimator, _affine


_LOGGER = logging.getLogger(__name__)


class MLPClassifier(MLPBaseEstimator, ClassifierMixin):
    """
    A deep neural network (multilayer perceptron) classifier using TensorFlow.

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
    dropout : float or None, optional
        The dropout probability.  If None, then dropout will not be used.
    activation : callable, optional
        The activation function.  See tensorflow.python.ops.nn.
    init_scale : float, optional
        The scale for the initialization function.  The TF default initializer,
        `uniform_unit_scaling_initializer
        <https://www.tensorflow.org/versions/r0.8/api_docs/python/state_ops.html#uniform_unit_scaling_initializer>`,
        is used.
    random_state: int, RandomState instance or None, optional
        If int, the random number generator seed. If RandomState instance,
        the random number generator itself. If None, then `np.random` will be
        used.

    Attributes
    ----------
    input_layer_sz_ : int
        The dimensionality of the input (i.e., number of features).
    is_sparse_ : bool
        Whether a model taking sparse input was fit.
    classes_ : list
        A list of the class labels.
    graph_ : tensorflow.python.framework.ops.Graph
        The TensorFlow graph for the model

    Notes
    -----
    tensorflow's Adam implementation is used for optimization.

    For multilabel classification, one can pass a 2D int array with 0 or more
    1s per row to `fit`.

    There is currently no dropout for sparse input layers.
    """

    def __init__(self, hidden_units=(256,), batch_size=64, n_epochs=5,
                 dropout=None, activation=nn.relu, init_scale=0.1,
                 random_state=None):
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout = dropout
        self.activation = activation
        self.init_scale = init_scale
        self.random_state = random_state

    def _init_model_output(self, t):
        n_classes = len(self.classes_)

        if self.multilabel_:
            output_size = n_classes
        elif n_classes > 2:
            output_size = n_classes
        else:
            output_size = 1

        if self.is_sparse_ and not self.hidden_units:
            t = _affine(t, output_size, input_sz=self.input_layer_sz_,
                        scope='output_layer', sparse_input=True)
        else:
            t = tf.nn.dropout(t, keep_prob=self._dropout)
            t = _affine(t, output_size, scope='output_layer')

        if self.multilabel_:
            self.input_targets_ = \
                tf.placeholder(tf.int64, [None, n_classes], "targets")
            self.output_layer_ = tf.nn.sigmoid(t)
        elif n_classes > 2:
            self.input_targets_ = tf.placeholder(tf.int64, [None], "targets")
            self.output_layer_ = tf.nn.softmax(t)
        else:
            self.input_targets_ = tf.placeholder(tf.int64, [None], "targets")
            t = tf.reshape(t, [-1])  # Convert to 1d tensor.
            self.output_layer_ = tf.nn.sigmoid(t)
        return t

    def _init_model_objective_fn(self, t):
        n_classes = len(self.classes_)
        if self.multilabel_:
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                t, tf.cast(self.input_targets_, np.float32))
        elif n_classes > 2:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                t, self.input_targets_)
        else:
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                t, tf.cast(self.input_targets_, np.float32))
        self._obj_func = tf.reduce_mean(cross_entropy)

    def _preprocess_targets(self, y):
        target_type = type_of_target(y)

        if target_type in ['binary', 'multiclass']:
            # Note: np.unique returns values in sorted order.
            self.classes_, y_ind = np.unique(y, return_inverse=True)
            self.multilabel_ = False
        elif target_type == 'multilabel-indicator':
            self.classes_ = np.array([0, 1])
            y_ind = y
            self.multilabel_ = True
        else:
            # Raise an error, as in
            # sklearn.utils.multiclass.check_classification_targets.
            raise ValueError("Unknown label type: %r" % y)

        return y_ind

    def predict_proba(self, X):
        """Predict probabilities for each class.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Examples to make predictions about.

        Returns
        -------
        C : array, shape = (n_samples, n_classes)
            Predicted probabilities for each class
        """
        y_pred = self._compute_output(X)

        if len(y_pred.shape) == 1:
            # The TF models returns a 1d array for binary models.
            # To conform with sklearn's LogisticRegression, return a 2D array.
            y_pred = np.column_stack((1.0 - y_pred, y_pred))

        return y_pred

    def predict(self, X):
        """Make predictions.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Examples to make predictions about.

        Returns
        -------
        C : array
            Predicted values. For multiclass or binary classification, this
            returns a 1-d array with the highest scoring (most probable) labels
            for each input. For multilabel, it returns a 2-d array with rows of
            0/1 indicators, one per label, for each input.
        """
        class_probs = self.predict_proba(X)

        if self.multilabel_:
            return (class_probs >= 0.5).astype(np.int)
        else:
            indices = class_probs.argmax(axis=1)
            return self.classes_[indices]

    def __getstate__(self):
        state = super().__getstate__()

        # Add the fitted attributes particular to this subclass.
        if getattr(self, '_fitted', False):
            state['classes_'] = self.classes_
            state['multilabel_'] = self.multilabel_

        return state
