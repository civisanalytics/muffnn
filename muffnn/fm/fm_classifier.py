import logging
import re

import scipy.sparse as sp
import numpy as np
import tensorflow as tf

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils import check_X_y, check_array, check_random_state
from sklearn.utils.multiclass import type_of_target
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder

from muffnn.core import TFPicklingBase


_LOGGER = logging.getLogger(__name__)


class FMClassifier(TFPicklingBase, ClassifierMixin, BaseEstimator):
    """Factorization machine classifier.

    Parameters
    ----------
    rank : int, optional
        Rank of the underlying low-rank representation.
    batch_size : int, optional
        The batch size for learning and prediction. If there are fewer
        examples than the batch size during fitting, then the the number of
        examples will be used instead.
    n_epochs : int, optional
        The number of epochs (iterations through the training data) when
        fitting. These are counted for the positive training examples, not
        the unlabeled data.
    random_state: int, RandomState instance or None, optional
        If int, the random number generator seed. If RandomState instance,
        the random number generator itself. If None, then `np.random` will be
        used.
    lambda_v : float, optional
        L2 regularization strength for the low-rank embedding.
    lambda_beta : float, optional
        L2 regularization strength for the linear coefficients.
    init_scale : float, optional
        Standard deviation of random normal initialization.
    solver : a subclass of `tf.train.Optimizer` or str, optional
        Solver to use. If a string is passed, then the corresponding solver
        from `scipy.optimize.minimize` is used.
    solver_kwargs : dict, optional
        Additional keyword arguments to pass to `solver` upon construction.
        See the TensorFlow documentation for possible options. Typically,
        one would want to set the `learning_rate`.

    Attributes
    ----------
    n_dims_ : int
        Number of input dimensions.
    classes_ : array
        Classes from the data.
    n_classes_ : int
        Number of classes.
    is_sparse_ : bool
        Whether a model taking sparse input was fit.
    """
    def __init__(self, rank=8, batch_size=64, n_epochs=5,
                 random_state=None, lambda_v=0.0,
                 lambda_beta=0.0, solver=tf.train.AdadeltaOptimizer,
                 init_scale=0.1, solver_kwargs=None):
        self.rank = rank
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.lambda_v = lambda_v
        self.lambda_beta = lambda_beta
        self.solver = solver
        self.init_scale = init_scale
        self.solver_kwargs = solver_kwargs

    def _set_up_graph(self):
        """Initialize TF objects (needed before fitting or restoring)."""

        # Input values.
        if self.is_sparse_:
            self._x_inds = tf.placeholder(tf.int64, [None, 2], "x_inds")
            self._x_vals = tf.placeholder(tf.float32, [None], "x_vals")
            self._x_shape = tf.placeholder(tf.int64, [2], "x_shape")
            self._x = tf.sparse_reorder(
                tf.SparseTensor(self._x_inds, self._x_vals, self._x_shape))
            x2 = tf.sparse_reorder(
                tf.SparseTensor(self._x_inds,
                                self._x_vals * self._x_vals,
                                self._x_shape))
            matmul = tf.sparse_tensor_dense_matmul
        else:
            self._x = tf.placeholder(tf.float32, [None, self.n_dims_], "x")
            x2 = self._x * self._x
            matmul = tf.matmul

        self._sample_weight = \
            tf.placeholder(np.float32, [None], "sample_weight")

        if self._output_size == 1:
            self._y = tf.placeholder(tf.float32, [None], "y")
        else:
            self._y = tf.placeholder(tf.int32, [None], "y")

        with tf.variable_scope("fm"):
            self._v = tf.get_variable(
                "v", [self.rank, self.n_dims_, self._output_size])
            self._beta = tf.get_variable(
                "beta", [self.n_dims_, self._output_size])
            self._beta0 = tf.get_variable("beta0", [self._output_size])

        vx = tf.stack([matmul(self._x, self._v[i, :, :])
                       for i in range(self.rank)], axis=-1)
        v2 = self._v * self._v
        v2x2 = tf.stack([matmul(x2, v2[i, :, :])
                         for i in range(self.rank)], axis=-1)
        int_term = 0.5 * tf.reduce_sum(tf.square(vx) - v2x2, axis=-1)
        self._logit_y_proba \
            = self._beta0 + matmul(self._x, self._beta) + int_term

        def reduce_weighted_mean(loss, weights):
            weighted = tf.multiply(loss, weights)
            return tf.divide(tf.reduce_sum(weighted),
                             tf.reduce_sum(weights))

        if self._output_size == 1:
            self._logit_y_proba = tf.squeeze(self._logit_y_proba)
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self._logit_y_proba,
                labels=self._y)
            self._obj_func = reduce_weighted_mean(
                cross_entropy, self._sample_weight)
            self._y_proba = tf.sigmoid(self._logit_y_proba)
        else:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self._logit_y_proba,
                labels=self._y)
            self._obj_func = reduce_weighted_mean(
                cross_entropy, self._sample_weight)
            self._y_proba = tf.nn.softmax(self._logit_y_proba)

        if self.lambda_v > 0:
            self._obj_func \
                += self.lambda_v * tf.reduce_sum(tf.square(self._v))

        if self.lambda_beta > 0:
            self._obj_func \
                += self.lambda_beta * tf.reduce_sum(tf.square(self._beta))

        if isinstance(self.solver, str):
            from tensorflow.contrib.opt import ScipyOptimizerInterface

            self._train_step = ScipyOptimizerInterface(
                self._obj_func,
                method=self.solver,
                options=self.solver_kwargs if self.solver_kwargs else {})
        else:
            self._train_step = self.solver(
                **self.solver_kwargs if self.solver_kwargs else {}).minimize(
                self._obj_func)

    def _make_feed_dict(self, X, y, sample_weight=None):
        # Make the dictionary mapping tensor placeholders to input data.
        if self.is_sparse_:
            x_inds = np.vstack(X.nonzero())
            x_srt = np.lexsort(x_inds[::-1, :])
            x_inds = x_inds[:, x_srt].T.astype(np.int64)
            x_vals = np.squeeze(np.array(
                X[x_inds[:, 0], x_inds[:, 1]])).astype(np.float32)
            x_shape = np.array(X.shape).astype(np.int64)
            feed_dict = {self._x_inds: x_inds,
                         self._x_vals: x_vals,
                         self._x_shape: x_shape}
        else:
            feed_dict = {self._x: X.astype(np.float32)}

        if self._output_size == 1:
            feed_dict[self._y] = y.astype(np.float32)
        else:
            feed_dict[self._y] = y.astype(np.int32)

        if sample_weight is None:
            feed_dict[self._sample_weight] = np.ones(X.shape[0])
        else:
            feed_dict[self._sample_weight] = sample_weight

        return feed_dict

    def _check_data(self, X):
        """check input data

        Raises an error if number of features doesn't match.
        If the estimator has not yet been fitted, then do nothing.
        """

        if self._is_fitted:
            if X.shape[1] != self.n_dims_:
                raise ValueError("Number of features in the input data does "
                                 "not match the number assumed by the "
                                 "estimator!")

    def __getstate__(self):
        # Handles TF persistence
        state = super(FMClassifier, self).__getstate__()

        # Add attributes of this estimator
        state.update(dict(rank=self.rank,
                          batch_size=self.batch_size,
                          n_epochs=self.n_epochs,
                          random_state=self.random_state,
                          lambda_v=self.lambda_v,
                          lambda_beta=self.lambda_beta,
                          solver=self.solver,
                          init_scale=self.init_scale,
                          solver_kwargs=self.solver_kwargs))

        # Add fitted attributes if the model has been fitted.
        if self._is_fitted:
            state['n_dims_'] = self.n_dims_
            state['_random_state'] = self._random_state
            state['_enc'] = self._enc
            state['classes_'] = self.classes_
            state['n_classes_'] = self.n_classes_
            state['_output_size'] = self._output_size
            state['is_sparse_'] = self.is_sparse_

        return state

    def fit(self, X, y, monitor=None, sample_weight=None):
        """Fit the classifier.

        Parameters
        ----------
        X : numpy array or sparse matrix [n_samples, n_features]
            Training data.
        y : numpy array [n_samples]
            Targets.
        monitor : callable, optional
            The monitor is called after each iteration with the current
            iteration, a reference to the estimator, and a dictionary with
            {'loss': loss_value} representing the loss calculated by the
            objective function at this iteration.
            If the callable returns True the fitting procedure is stopped.
            The monitor can be used for various things such as computing
            held-out estimates, early stopping, model introspection,
            and snapshotting.
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
        return self.partial_fit(X, y, monitor=monitor,
                                sample_weight=sample_weight)

    def partial_fit(self, X, y, classes=None, monitor=None,
                    sample_weight=None):
        """Fit the classifier.

        Parameters
        ----------
        X : numpy array or sparse matrix [n_samples, n_features]
            Training data.
        y : numpy array [n_samples]
            Targets.
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
            and snapshotting.
        sample_weight : numpy array of shape [n_samples,]
            Per-sample weights. Re-scale the loss per sample.
            Higher weights force the estimator to put more emphasis
            on these samples. Sample weights are normalized per-batch.

        Returns
        -------
        self : returns an instance of self.
        """

        X, y = check_X_y(X, y, accept_sparse='csr')

        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)

        # check target type
        target_type = type_of_target(y)
        if target_type not in ['binary', 'multiclass']:
            # Raise an error, as in
            # sklearn.utils.multiclass.check_classification_targets.
            raise ValueError("Unknown label type: %s" % target_type)

        # Initialize the model if it hasn't been already by a previous call.
        if not self._is_fitted:
            self._random_state = check_random_state(self.random_state)
            assert self.batch_size > 0, "batch_size <= 0"

            self.n_dims_ = X.shape[1]

            if classes is not None:
                self._enc = LabelEncoder().fit(classes)
            else:
                self._enc = LabelEncoder().fit(y)

            self.classes_ = self._enc.classes_
            self.n_classes_ = len(self.classes_)

            if self.n_classes_ <= 2:
                self._output_size = 1
            else:
                self._output_size = self.n_classes_

            if sp.issparse(X):
                self.is_sparse_ = True
            else:
                self.is_sparse_ = False

            # Instantiate the graph.  TensorFlow seems easier to use by just
            # adding to the default graph, and as_default lets you temporarily
            # set a graph to be treated as the default graph.
            self.graph_ = tf.Graph()
            with self.graph_.as_default():
                tf.set_random_seed(self._random_state.randint(0, 10000000))

                tf.get_variable_scope().set_initializer(
                    tf.random_normal_initializer(stddev=self.init_scale))

                self._build_tf_graph()

                # Train model parameters.
                self._session.run(tf.global_variables_initializer())

            # Set an attributed to mark this as at least partially fitted.
            self._is_fitted = True

        # Check input data against internal data.
        # Raises an error on failure.
        self._check_data(X)

        # transform targets
        if sp.issparse(y):
            y = y.toarray()
        y = self._enc.transform(y)

        # Train the model with the given data.
        with self.graph_.as_default():
            if not isinstance(self.solver, str):
                n_examples = X.shape[0]
                indices = np.arange(n_examples)

                for epoch in range(self.n_epochs):
                    self._random_state.shuffle(indices)
                    for start_idx in range(0, n_examples, self.batch_size):
                        max_ind = min(start_idx + self.batch_size, n_examples)
                        batch_ind = indices[start_idx:max_ind]

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
            else:
                feed_dict = self._make_feed_dict(
                    X, y, sample_weight=sample_weight)
                self._train_step.minimize(self._session,
                                          feed_dict=feed_dict)

        return self

    def predict_log_proba(self, X):
        """Compute log p(y=1).

        Parameters
        ----------
        X : numpy array or sparse matrix [n_samples, n_features]
            Data.

        Returns
        -------
        numpy array [n_samples]
            Log probabilities.
        """
        if not self._is_fitted:
            raise NotFittedError("Call fit before predict_log_proba!")
        return np.log(self.predict_proba(X))

    def predict_proba(self, X):
        """Compute p(y=1).

        Parameters
        ----------
        X : numpy array or sparse matrix [n_samples, n_features]
            Data.

        Returns
        -------
        numpy array [n_samples]
            Probabilities.
        """

        if not self._is_fitted:
            raise NotFittedError("Call fit before predict_proba!")

        X = check_array(X, accept_sparse='csr')

        # Check input data against internal data.
        # Raises an error on failure.
        self._check_data(X)

        # Compute weights in batches.
        probs = []
        start_idx = 0
        n_examples = X.shape[0]
        with self.graph_.as_default():
            while start_idx < n_examples:
                X_batch = \
                    X[start_idx:min(start_idx + self.batch_size, n_examples)]
                feed_dict = self._make_feed_dict(
                    X_batch, np.zeros(self.n_dims_))
                start_idx += self.batch_size
                probs.append(np.atleast_1d(self._y_proba.eval(
                    session=self._session, feed_dict=feed_dict)))

        probs = np.concatenate(probs, axis=0)
        if probs.ndim == 1:
            return np.column_stack([1.0 - probs, probs])
        else:
            return probs

    def predict(self, X):
        """Compute the predicted class.

        Parameters
        ----------
        X : numpy array or sparse matrix [n_samples, n_features]
            Data.

        Returns
        -------
        numpy array [n_samples]
            Predicted class.
        """
        if not self._is_fitted:
            raise NotFittedError("Call fit before predict!")
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
