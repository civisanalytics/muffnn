"""
A Deep Neural Network (multilayer Perceptron) sklearn-style classifier.

Similar to sklearn.neural_network.MLPClassifier, but using TensorFlow.
"""

import logging

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target

import tensorflow as tf
from tensorflow.python.ops import nn
from muffnn.mlp.base import MLPBaseEstimator
from muffnn.core import affine


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
    classes_ : list
        A list of the class labels.
    graph_ : tensorflow.python.framework.ops.Graph
        The TensorFlow graph for the model

    Notes
    -----
    For multilabel classification, one can pass a 2D int array with 0 or more
    1s per row to `fit`.

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

        if self.multilabel_:
            output_size = self.n_classes_
        elif self.n_classes_ > 2:
            output_size = self.n_classes_
        else:
            output_size = 1

        if self.is_sparse_ and not self.hidden_units:
            t = affine(t, output_size, input_size=self.input_layer_sz_,
                       scope='output_layer', sparse_input=True)
        else:
            if self.keep_prob != 1.0:
                if self.activation is tf.nn.selu:
                    t = tf.contrib.nn.alpha_dropout(
                        t, keep_prob=self._keep_prob)
                else:
                    t = tf.nn.dropout(t, keep_prob=self._keep_prob)
            t = affine(t, output_size, scope='output_layer')

        if self.multilabel_:
            self.input_targets_ = \
                tf.placeholder(tf.int64, [None, self.n_classes_], "targets")
            self.output_layer_ = tf.nn.sigmoid(t)
            self._zeros = tf.zeros_like(self.output_layer_)
        elif self.n_classes_ > 2:
            self.input_targets_ = tf.placeholder(tf.int64, [None], "targets")
            self.output_layer_ = tf.nn.softmax(t)
        else:
            self.input_targets_ = tf.placeholder(tf.int64, [None], "targets")
            t = tf.reshape(t, [-1])  # Convert to 1d tensor.
            self.output_layer_ = tf.nn.sigmoid(t)
        return t

    def _init_model_objective_fn(self, t):

        def reduce_weighted_mean(loss, weights):
            weighted = tf.multiply(loss, weights)
            return tf.divide(tf.reduce_sum(weighted),
                             tf.reduce_sum(weights))

        if self.multilabel_:
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=t, labels=tf.cast(self.input_targets_, np.float32))
            y_finite = tf.equal(self.input_targets_, -1)

            # reshape the weights of shape (batch_size,) to shape
            # (batch_size, n_classes_) using ``tile`` to place the
            # values from self._sample_weight into each column of the
            # resulting matrix.
            # This allows us to arrive at the correct divisor in the
            # weighted mean calculation by summing the matrix.
            sample_weight = tf.reshape(self._sample_weight, (-1, 1))
            sample_weight = tf.tile(sample_weight, (1, self.n_classes_))

            self._obj_func = reduce_weighted_mean(
                tf.where(y_finite, self._zeros, cross_entropy),
                tf.where(y_finite, self._zeros, sample_weight))
        elif self.n_classes_ > 2:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=t, labels=self.input_targets_)
            self._obj_func = reduce_weighted_mean(
                cross_entropy, self._sample_weight)
        else:
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=t, labels=tf.cast(self.input_targets_, np.float32))
            self._obj_func = reduce_weighted_mean(
                cross_entropy, self._sample_weight)

    def partial_fit(self, X, y,
                    monitor=None, sample_weight=None, classes=None):
        """Fit the model on a batch of training data.

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples, n_features]
            Training data
        y : numpy array of shape [n_samples, n_targets]
            Target values
        classes : array, shape (n_classes,)
            Classes to be used across calls to partial_fit.  If not set in the
            first call, it will be inferred from the given targets. If
            subsequent calls include additional classes, they will fail.
        monitor : callable, optional
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator, and a dictionary with
            {'loss': loss_value} representing the loss calculated by the
            objective function at this iteration.
            If the callable returns True the fitting procedure is stopped.
            The monitor can be used for various things such as computing
            held-out estimates, early stopping, model introspection,
            and snapshoting.
        sample_weight : numpy array of shape (n_samples,)
            Per-sample weights. Re-scale the loss per sample.
            Higher weights force the estimator to put more emphasis
            on these samples. Sample weights are normalized per-batch.

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        This is based on
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
        """
        return super(MLPClassifier, self).partial_fit(
            X, y,
            monitor=monitor, sample_weight=sample_weight, classes=classes)

    def _is_multilabel(self, y):
        """
        Return whether the given target array corresponds to a multilabel
        problem.
        """
        temp_y = y.copy()
        temp_y[np.zeros_like(temp_y, dtype=bool) | (temp_y == -1)] = 1
        target_type = type_of_target(temp_y)

        if target_type in ['binary', 'multiclass']:
            return False
        elif target_type == 'multilabel-indicator':
            return True
        else:
            # Raise an error, as in
            # sklearn.utils.multiclass.check_classification_targets.
            raise ValueError("Unknown label type: %s" % target_type)

    def _fit_targets(self, y, classes=None):
        self.multilabel_ = self._is_multilabel(y)

        # If provided, use classes to fit the encoded and set classes_.
        # Otherwise, find the unique classes in y.
        if classes is not None:
            y = classes

        if self.multilabel_:
            self._enc = None
            self.classes_ = np.arange(y.shape[1])
            self.n_classes_ = y.shape[1]
        else:
            self._enc = LabelEncoder().fit(y)
            self.classes_ = self._enc.classes_
            self.n_classes_ = len(self.classes_)

    def _transform_targets(self, y):
        return y if self.multilabel_ else self._enc.transform(y)

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
        state = super(MLPClassifier, self).__getstate__()

        # Add the fitted attributes particular to this subclass.
        if self._is_fitted:
            state['_enc'] = self._enc
            state['classes_'] = self.classes_
            state['multilabel_'] = self.multilabel_
            state['n_classes_'] = self.n_classes_

        return state

    def score(self, X, y):
        accuracy = np.array(y) == self.predict(X)
        accuracy = accuracy[np.array(y) != -1].mean()
        return accuracy
