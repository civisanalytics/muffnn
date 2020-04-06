import logging
from warnings import warn
import re

import scipy.sparse as sp
import numpy as np
import tensorflow as tf

from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils import check_X_y, check_array, check_random_state
from sklearn.exceptions import NotFittedError

from muffnn.core import TFPicklingBase


_LOGGER = logging.getLogger(__name__)


class FMRegressor(TFPicklingBase, RegressorMixin, BaseEstimator):
    """Factorization machine regressor.

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

        self._y = tf.placeholder(tf.float32, [None], "y")

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
        self._y_hat \
            = self._beta0 + matmul(self._x, self._beta) + int_term

        self._y_hat = tf.squeeze(self._y_hat)
        mse = tf.square(self._y - self._y_hat)
        self._obj_func = tf.divide(
            tf.reduce_sum(
                tf.multiply(mse, self._sample_weight)),
            tf.reduce_sum(self._sample_weight))

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

        if sample_weight is None:
            feed_dict[self._sample_weight] = np.ones(X.shape[0])
        else:
            feed_dict[self._sample_weight] = sample_weight

        feed_dict[self._y] = y.astype(np.float32)

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

    def _fit_target(self, y):
        # Store the mean and S.D. of the targets so we can have standardized
        # y for training but still make predictions on the original scale.

        self.target_mean_ = np.mean(y)

        self.target_sd_ = np.std(y - self.target_mean_)
        if self.target_sd_ <= 0:
            warn("No variance in regression targets.")

    def _transform_target(self, y):
        # Standardize the targets for fitting, and store the M and SD values
        # for prediction.

        y_centered = y - self.target_mean_

        if self.target_sd_ <= 0:
            return y_centered

        return y_centered / self.target_sd_

    def __getstate__(self):
        # Handles TF persistence
        state = super(FMRegressor, self).__getstate__()

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
            state['_output_size'] = self._output_size
            state['is_sparse_'] = self.is_sparse_
            state['target_mean_'] = self.target_mean_
            state['target_sd_'] = self.target_sd_

        return state

    def fit(self, X, y, monitor=None, sample_weight=None):
        """Fit the classifier.

        Parameters
        ----------
        X : numpy array or sparse matrix [n_samples, n_features]
            Training data.
        y : numpy array [n_samples]
            Outcome.
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
        return self.partial_fit(X, y,
                                monitor=monitor,
                                sample_weight=sample_weight)

    def partial_fit(self, X, y, monitor=None, sample_weight=None):
        """Fit the classifier.

        Parameters
        ----------
        X : numpy array or sparse matrix [n_samples, n_features]
            Training data.
        y : numpy array [n_samples]
            Outcome.
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

        X, y = check_X_y(X, y, accept_sparse='csr', multi_output=False)

        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)

        # Initialize the model if it hasn't been already by a previous call.
        if not self._is_fitted:
            self._random_state = check_random_state(self.random_state)
            assert self.batch_size > 0, "batch_size <= 0"
            self._fit_target(y)
            y = self._transform_target(y)

            self.n_dims_ = X.shape[1]
            self._output_size = 1

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
        else:
            y = self._transform_target(y)

        # Check input data against internal data.
        # Raises an error on failure.
        self._check_data(X)

        # transform targets
        if sp.issparse(y):
            y = y.toarray()

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

    def predict(self, X):
        """Predict expected y values.

        Parameters
        ----------
        X : numpy array or sparse matrix [n_samples, n_features]
            Data.

        Returns
        -------
        numpy array [n_samples]
            Estimated regression predictions.
        """

        if not self._is_fitted:
            raise NotFittedError("Call fit before predict!")

        X = check_array(X, accept_sparse='csr')

        # Check input data against internal data.
        # Raises an error on failure.
        self._check_data(X)

        # Compute weights in batches.
        yhat = []
        start_idx = 0
        n_examples = X.shape[0]
        with self.graph_.as_default():
            while start_idx < n_examples:
                X_batch = \
                    X[start_idx:min(start_idx + self.batch_size, n_examples)]
                feed_dict = self._make_feed_dict(
                    X_batch, np.zeros(self.n_dims_))
                start_idx += self.batch_size
                yhat.append(np.atleast_1d(self._y_hat.eval(
                    session=self._session, feed_dict=feed_dict)))

        yhat = np.concatenate(yhat, axis=0)

        # Put the prediction back on the scale of the target values
        # (cf. _transform_targets).
        if self.target_sd_ > 0.0:
            yhat *= self.target_sd_
        yhat += self.target_mean_

        return yhat
