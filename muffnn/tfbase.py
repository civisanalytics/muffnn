#!/usr/bin/env python3
"""
A mixin for pickling TensorFlow-based scikit-learn estimators.
"""
import os
from abc import ABCMeta, abstractmethod
from tempfile import NamedTemporaryFile
import tensorflow as tf


class TFPicklingBase(metaclass=ABCMeta):
    """Base class for pickling TensorFlow-based scikit-learn estimators.

    This base class defines a few standard attributes to enable fairly transparent pickling
    of TensorFlow models.

    Upon pickling an object, if the self._is_fitted property is True:

        1. The session at self._session is saved using the saver at self._saver to a temporary
            file.

        2. The saved data is then read into memory and attached to the object state at
            '_saved_model'.

        3. The fitted state of the model is saved at '_fitted' as True.

    Upon unpickling the object:

        1. All things in the state of the object are set using self.__dict__ except the
            '_saved_model' entry.

        2. If the '_fitted' key is in the state of the object and is True

            2a. The '_saved_model' entry is written to a temporary file.
            2b. A new TF graph is instantiated at self.graph_.
            2c. self._build_tf_graph() is called. This sets a Saver at self._saver and a Session
                at self._session.
            2d. The self._saver is used to restore previous session to the current one.

    To use this base class properly, the child class needs to

        1. Implement the abstract method self._set_up_graph. This method should simply build the
            required TF ops.

        2. Exactly once, instantiate a TF graph at self.graph_ and then call self._build_tf_graph
            inside this context block. self._build_tf_graph will call self._set_up_graph and
            further instantiate the Saver and Session.

        3. After 2. is done, set `self._is_fitted = True`.

        4. Make sure override __getstate__ to store any extra information about your estimator
           to the state of the object. When doing this, call `state = super().__getstate__()` and
           then append to the `state`.

    See the example below.

    Example
    -------

    ```python
    # example class for using TFPicklingBase - adds a scalar to input 1d arrays
    class TFAdder(TFPicklingBase):
        def __init__(self, add_val):
            # real scikit-learn estimators should do all of this work in the fit method
            self.add_val = float(add_val)
            self.graph_ = tf.Graph()
            with self.graph_.as_default():
                self._build_tf_graph()
                self._session.run(tf.initialize_all_variables())
                self._is_fitted = True

        def _set_up_graph(self):
            self._a = tf.placeholder(tf.float32, shape=[None], name='a')
            self._add_val = tf.Variable(self.add_val,  name='add_val', dtype=tf.float32)
            self._sum = tf.add(self._a, self._add_val, name='sum')

        def add(self, a):
            with self.graph_.as_default():
                val = self._session.run(self._sum, feed_dict={self._a: a})
            return val

        def __getstate__(self):
            state = super().__getstate__()

            # add add_val to state
            state['add_val'] = self.add_val

            return state
    ```
    """

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
        if self._is_fitted:
            tempfile = NamedTemporaryFile(delete=False)
            tempfile.close()
            try:
                # Serialize the model and read it so it can be pickled.
                self._saver.save(self._session, tempfile.name)
                with open(tempfile.name, 'rb') as f:
                    saved_model = f.read()
            finally:
                os.unlink(tempfile.name)

        # Note: don't include the graph since it should be recreated.
        state = {}

        # Add fitted attributes if the model has been fitted.
        if self._is_fitted:
            state['_fitted'] = True
            state['_saved_model'] = saved_model

        return state

    def __setstate__(self, state):
        # Override __setstate__ so that TF model parameters are unpickled
        # properly.
        for k, v in state.items():
            if k != '_saved_model':
                self.__dict__[k] = v

        if state.get('_fitted', False):
            tempfile = NamedTemporaryFile(delete=False)
            tempfile.close()
            try:
                # Write out the serialized model that can be restored by TF.
                with open(tempfile.name, 'wb') as f:
                    f.write(state['_saved_model'])
                self.graph_ = tf.Graph()
                with self.graph_.as_default():
                    self._build_tf_graph()
                    self._saver.restore(self._session, tempfile.name)
            finally:
                os.unlink(tempfile.name)

    def _build_tf_graph(self):
        """build the TF graph, setup model saving and setup a TF session

        Notes
        -----
        This method initializes a TF Saver and a TF Session via

            ```python
            self._saver = tf.train.Saver()
            self._session = tf.Session()
            ```

        These calls are made after self._set_up_graph() is called.

        See the main class docs for how to properly call this method from a child class.
        """
        self._set_up_graph()
        self._saver = tf.train.Saver()
        self._session = tf.Session()

    @abstractmethod
    def _set_up_graph(self):
        """abstract method to assemble the TF graph for estimator

        Notes
        -----
        Child classes should add the TF ops they want to implement here.
        """
        pass
