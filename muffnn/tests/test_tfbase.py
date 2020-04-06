"""
Tests for TFPicklingBase
"""

import io
import pickle
import numpy as np
import tensorflow as tf
from sklearn.utils.testing import assert_array_almost_equal

from muffnn.core import TFPicklingBase


# example class for using TFPicklingBase - adds a scalar to input 1d
# arrays
class TFAdder(TFPicklingBase):
    def __init__(self, add_val):
        # real scikit-learn estimators should do all of this work in the
        # fit method
        self.add_val = float(add_val)
        self.graph_ = tf.Graph()
        with self.graph_.as_default():
            self._build_tf_graph()
            self._session.run(tf.global_variables_initializer())
            self._is_fitted = True

    def _set_up_graph(self):
        self._a = tf.placeholder(tf.float32, shape=[None], name='a')
        self._add_val = tf.Variable(self.add_val,
                                    name='add_val',
                                    dtype=tf.float32)
        self._sum = tf.add(self._a, self._add_val, name='sum')

    def add(self, a):
        with self.graph_.as_default():
            val = self._session.run(self._sum, feed_dict={self._a: a})
        return val

    def __getstate__(self):
        state = super(TFAdder, self).__getstate__()

        # add add_val to state
        state['add_val'] = self.add_val

        return state


def test_add():
    """Test that child class interface to base class works as expected"""
    a = np.arange(10)
    adder = TFAdder(11)
    assert_array_almost_equal(adder.add(a), a + 11)


def test_pickling():
    """Test that models can be pickled and reloaded."""
    adder1 = TFAdder(11)
    b = io.BytesIO()
    pickle.dump(adder1, b)
    adder2 = pickle.loads(b.getvalue())
    a = np.arange(10)
    assert_array_almost_equal(adder1.add(a), adder2.add(a))
