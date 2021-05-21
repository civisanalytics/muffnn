import logging
import time

import civis
from itertools import product
import numpy as np
import pandas as pd
from patsy import dmatrix
import scipy
import scipy.sparse as sp
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
import tensorflow as tf

from muffnn import FMClassifier

logger = logging.getLogger('FMClassifier')
logger.setLevel('DEBUG')


class TF2MockFMClassifier(FMClassifier, tf.Module):
    def __init__(self, kwargs={'solver': tf.keras.optimizers.Adam,
                               'solver_kwargs': {'learning_rate': 0.1},
                               }):
        super().__init__(**kwargs)

    def _init_vars(self, x, y, classes=None):
        """Initialize TF objects (needed before fitting or restoring)."""
        if not self._is_fitted:
            self._random_state = check_random_state(self.random_state)
            assert self.batch_size > 0, "batch_size <= 0"

            self.n_dims_ = x.shape[1]

            if classes is not None:
                self._enc = LabelEncoder().fit(classes)
            else:
                self._enc = LabelEncoder().fit(y)

            self.classes_ = self._enc.classes_
            self.n_classes_ = len(self.classes_)

            if self.n_classes_ <= 2:
                self._output_size = 1
            else:
                self._output_size = self.n_classes

            if sp.issparse(x):
                self.is_sparse_ = True
            else:
                self.is_sparse_ = False

            tf.random.set_seed(self._random_state.randint(0, 10000000))
            self._v = tf.Variable(tf.ones(shape=(self.rank, self.n_dims_, self._output_size)),
                                  name="v")
            self._beta = tf.Variable(tf.ones(shape=(self.n_dims_, self._output_size)),
                                     name="beta")
            self._beta0 = tf.Variable(tf.zeros(shape=(self._output_size)), name="beta0")

            self._solver = self.solver(**self.solver_kwargs if self.solver_kwargs else {})

    def __call__(self, x, v, beta, beta0):
        x2 = x * x
        vx = tf.stack([tf.linalg.matmul(x, v[i, :, :])
                       for i in range(self.rank)], axis=-1)
        v2 = v * v
        v2x2 = tf.stack([tf.linalg.matmul(x2, v2[i, :, :])
                         for i in range(self.rank)], axis=-1)
        int_term = 0.5 * tf.math.reduce_sum(tf.square(vx) - v2x2, axis=-1)
        return beta0 + tf.linalg.matmul(x, beta) + int_term

    def _fit(self, x, y, sample_weight=None, classes=None):
        def loss_fn(y, logits, sample_weights):
            def reduce_weighted_mean(loss, weights):
                weighted = tf.math.multiply(loss, weights)
                return tf.math.divide(tf.math.reduce_sum(weighted),
                                      tf.math.reduce_sum(weights))
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits,
                labels=y)
            val = reduce_weighted_mean(cross_entropy, sample_weights)
            if self.lambda_v > 0:
                val += tf.keras.regularizers.L2(self.lambda_v)(self._v)

            if self.lambda_beta > 0:
                val += tf.keras.regularizers.L2(self.lambda_beta)(self._beta)

            return val

        self._is_fitted = False
        self._init_vars(x, y, classes)

        self._x = x.astype(np.float32)
        self._y = y.astype(np.float32)
        n_examples = self._x.shape[0]

        if sample_weight is not None:
            self._sample_weight = sample_weight.astype(np.float32)
        else:
            self._sample_weight = np.ones(self._x.shape[0]).astype(np.float32)

        @tf.function(input_signature=[
            tf.TensorSpec(shape=(None, self.n_dims_), dtype=np.float32),
            tf.TensorSpec(shape=(None,), dtype=np.float32),
            tf.TensorSpec(shape=(None,), dtype=np.float32)
        ])
        def train_step(x, y, sample_weights):
            with tf.GradientTape() as tape:
                logits = tf.squeeze(self(x, self._v, self._beta, self._beta0))
                obj_val = loss_fn(y, logits, sample_weights)

            # gradients = tape.gradient(obj_val, [self._v, self._beta, self._beta0])
            #
            # self._solver.apply_gradients(zip(gradients, [self._v, self._beta, self._beta0]))
            self._solver.minimize(obj_val, self.trainable_variables, tape=tape)
            return logits, obj_val#, gradients

        (self._logit_y_proba,
         self._obj_val) = train_step(self._x, self._y, self._sample_weight)
        self._is_fitted = True

        self._train_set = tf.data.Dataset.from_tensor_slices(
            (self._x, self._y, self._sample_weight))
        start_time = time.time()
        for epoch in range(self.n_epochs):
            train_set = (self._train_set
                         .shuffle(buffer_size=n_examples,
                                  seed=self._random_state.randint(0, 10000000))
                         .batch(self.batch_size)
                         .prefetch(2))
            for step, (_x, _y, _wt) in enumerate(train_set):
                (self._logit_y_proba,
                 self._obj_val) = train_step(_x, _y, _wt)
                logger.debug("objective: %.4f, epoch: %d, step: %d",
                             float(self._obj_val), epoch, step)

            logger.debug("objective: %.4f, epoch: %d, step: %d",
                         float(self._obj_val), epoch, step)
        duration = time.time() - start_time
        logger.debug("Training in batches took %.4f s", duration)

        return self

    def _predict_proba(self, x):
        if not self._is_fitted:
            raise NotFittedError('Must fit the new FM classifier first!')

        @tf.function(input_signature=[
            tf.TensorSpec(shape=(None, self.n_dims_), dtype=np.float32)
        ])
        def _predict(x):
            return tf.squeeze(tf.math.sigmoid(
                self(x, self._v, self._beta, self._beta0)))

        self._x = x.astype(np.float32)
        self.test_set = tf.data.Dataset.from_tensor_slices(self._x)
        test_set = self.test_set.batch(self.batch_size).prefetch(2)

        probs = []
        start_time = time.time()
        for batch in test_set:
            probs.append(np.atleast_1d(_predict(batch)))
        duration = time.time() - start_time
        logger.debug("Predicting in batches took %.4f s", duration)

        probs = np.concatenate(probs, axis=0)
        if probs.ndim == 1:
            return np.column_stack([1. - probs, probs])
        else:
            return probs

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

    def __getstate__(self):
        # Override __getstate__ so that TF model parameters are pickled
        # properly.
        state = {}
        state.update(dict(
            rank=self.rank,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            random_state=self.random_state,
            lambda_v=self.lambda_v,
            lambda_beta=self.lambda_beta,
            solver=self.solver,
            init_scale=self.init_scale,
            solver_kwargs=self.solver_kwargs,
            n_dims_=self.n_dims_,
            is_sparse_=self.is_sparse_,
            _fitted=self._fitted,
        ))

        if self._fitted:
            weights = {}
            for var in self.trainable_variables:
                name = '_' + var.name.split(':')[0]
                weights.update({name: tf.io.serialize_tensor(var)})
            state.update(dict(
                variables=weights,
            ))

        return state

    def __setstate__(self, state):
        # Override __setstate__ so that TF model parameters are unpickled
        # properly.
        for k, v in state.items():
            if k != 'variables':
                self.__dict__[k] = v
        if self.__dict__['_fitted']:
            for name, weight in state['variables'].items():
                replace_name = name.replace('_', '')
                new_var = tf.io.parse_tensor(weight, out_type=np.float32)
                self.__dict__[name] = tf.Variable(
                    new_var,
                    dtype=np.float32,
                    name=replace_name)

        return self


ncol = 10
form = ' + '.join([f'x{str(i)}' for i in range(ncol)])
interaction_iter = product([i for i in range(ncol)], [i for i in range(ncol)])
form += ' - 1 + '
form += ' + '.join(
    [f'x{str(i)}:x{str(j)}' for (i, j) in interaction_iter if i < j])

np.random.seed(1)
nonsparse_x = np.random.binomial(1, .5, 20000).reshape((2000, ncol))
dmat = dmatrix(form,
               data=pd.DataFrame(nonsparse_x).rename(columns={
                i: f'x{str(i)}' for i in range(ncol)}))
betas = np.random.standard_normal(dmat.shape[1])

lin_fx_sd = 1
interaction_fx_sd = 0.25
betas[0:(ncol - 1)] /= betas[0:(ncol - 1)].std() / lin_fx_sd
betas[ncol:] /= betas[ncol:].std() / interaction_fx_sd
probs = scipy.special.expit(dmat @ betas)
binary_y = np.random.binomial(1, probs)
no_sample_weight = np.ones(nonsparse_x.shape[0])
fm1 = TF2MockFMClassifier()
fm1._fit(nonsparse_x, binary_y)
fm1_preds = fm1._predict_proba(nonsparse_x)

pickled_fm = pickle.dumps(fm1)
loaded_fm = pickle.loads(pickled_fm)

loaded_preds = loaded_fm._predict_proba(nonsparse_x)
