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


class TF2MockFMClassifier(FMClassifier):
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
            self._solver.minimize(obj_val, [self._v, self._beta, self._beta0], tape=tape)
            return logits, obj_val#, gradients

        (self._logit_y_proba,
         self._obj_val) = train_step(self._x, self._y, self._sample_weight)
        # self._train_step = self.solver.apply_gradients(
        #     zip(self._gradients, [self._v, self._beta, self._beta0]))
        # self._train_step = self.solver.minimize(self._obj_val,
        #                                         [self._v, self._beta, self._beta0],
        #                                         tape=self._tape)
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
                # self._train_step = self.solver.apply_gradients(
                #     zip(self._gradients, [self._v, self._beta, self._beta0]))
                # self._train_step = self.solver.minimize(self._obj_val,
                #                                         [self._v, self._beta, self._beta0],
                #                                         tape=self._tape)

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
preds = fm1._predict_proba(nonsparse_x)
np.mean(preds[:, 1])
preds.argmax(axis=1)
binary_y.mean()
int_term = 0.5 * tf.math.reduce_sum(
    tf.math.square(tf.stack([tf.linalg.matmul(fm1._x, fm1._v[i, :, :])
                   for i in range(fm1.rank)], axis=-1)) -
    tf.stack([tf.linalg.matmul(fm1._x * fm1._x, (fm1._v * fm1._v)[i, :, :])
              for i in range(fm1.rank)], axis=-1), axis=-1)
tf.sigmoid(int_term + fm1._beta0 + tf.linalg.matmul(fm1._x, fm1._beta))

fm2 = FMClassifier(solver=tf.train.AdamOptimizer,
                   solver_kwargs={'learning_rate': 0.01},
                   random_state=2045)
fm2.fit(nonsparse_x, binary_y)
data = tf.data.Dataset.from_tensor_slices(nonsparse_x)
fm1(nonsparse_x.astype(np.float32), fm1._v, fm1._beta, fm1._beta0)
preds = pd.DataFrame(fm2.predict_proba(nonsparse_x))
civis.io.dataframe_to_civis(preds, "redshift-general", "survey.old_muffnn_fm_preds",
                            api_key='LMO7hW61K5wHGBp_6dlNOcfUp5qc6YqstL-ZWkGE-Gg')
civis.io.dataframe_to_civis(pd.DataFrame(binary_y), "redshift-general",
                            "survey.fm_y_values",
                            api_key='LMO7hW61K5wHGBp_6dlNOcfUp5qc6YqstL-ZWkGE-Gg')
help(civis.io.dataframe_to_civis)
lm = LogisticRegression(C=0.1)
lm_preds = cross_val_predict(lm, nonsparse_x, binary_y, cv=10, method='predict_proba')
roc_auc_score(binary_y, lm_preds[:, 1])
