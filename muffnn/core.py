from __future__ import print_function
from __future__ import division

import os
import glob
import tarfile
from abc import ABCMeta, abstractmethod
import six
import tensorflow as tf


if six.PY3:
    from tempfile import TemporaryDirectory
else:
    # Backport TemporaryDirectory; this was introduced in Python 3.2
    from tempfile import mkdtemp
    import shutil as _shutil
    import sys as _sys
    import warnings as _warnings

    class ResourceWarning(Warning):
        pass

    # from civis-python (civis.compat)
    # copied here to avoid the dependency
    class TemporaryDirectory(object):
        """Create and return a temporary directory.  This has the same
        behavior as mkdtemp but can be used as a context manager.  For
        example:

            with TemporaryDirectory() as tmpdir:
                ...

        Upon exiting the context, the directory and everything contained
        in it are removed.

        This is a port of the Python 3.2+ TemporaryDirectory object,
        modified slightly to work with Python 2.7. Python 3 docs are at
        https://docs.python.org/3/library/tempfile.html#tempfile.TemporaryDirectory
        """

        def __init__(self, suffix='', prefix='tmp', dir=None):
            self._closed = False
            self.name = None  # Handle mkdtemp raising an exception
            self.name = mkdtemp(suffix, prefix, dir)

        def __repr__(self):
            return "<{} {!r}>".format(self.__class__.__name__, self.name)

        def __enter__(self):
            return self.name

        def __exit__(self, exc, value, tb):
            self.cleanup()

        def __del__(self):
            # Issue a ResourceWarning if implicit cleanup needed
            self.cleanup(_warn=True)

        def cleanup(self, _warn=False):
            if self.name and not self._closed:
                try:
                    _shutil.rmtree(self.name)
                except (TypeError, AttributeError) as ex:
                    # Issue #10188: Emit a warning on stderr
                    # if the directory could not be cleaned
                    # up due to missing globals
                    if "None" not in str(ex):
                        raise
                    print("ERROR: {!r} while cleaning up "
                          "{!r}".format(ex, self,),
                          file=_sys.stderr)
                    return
                self._closed = True
                if _warn:
                    _warnings.warn("Implicitly cleaning up {!r}".format(self),
                                   ResourceWarning)


def affine(input_tensor, output_size, bias=True, bias_start=0.0,
           input_size=None, scope="affine", sparse_input=False):
    """Add an affine transformation of `input_tensor` to the current graph.

    Note: This op is loosely based on tensorflow.python.ops.rnn_cell.linear.

    An affine transformation is a linear transformation with a shift,
    `t = tf.matmul(input_tensor, W) + b`.

    Parameters
    ----------
    input_tensor : tensorflow Tensor object, rank 2
        Input tensor to be transformed.
    output_size : int
        The output will be size [a, output_size] where `input_tensor` has
        shape [a, b].
    bias : bool, optional
        If True, apply a bias to the transformation. If False, only a linear
        transformation is applied (i.e., `t = tf.matmul(W, input_tensor)`).
    bias_start : float, optional
        The initial value for the bias `b`.
    input_size : int, optional
        Second dimension of the rank 2 input tensor. Required for sparse input
        tensors.
    sparse_input : bool, optional
        Set to True if `input_tensor` is sparse.

    Returns
    -------
    t : tensorflow tensor object
        The affine transformation of `input_tensor`.
    """

    # The input size is needed for sparse matrices.
    if input_size is None:
        input_size = input_tensor.get_shape().as_list()[1]

    with tf.variable_scope(scope):
        W_0 = tf.get_variable(
            "weights0",
            [input_size, output_size])
        # If the input is sparse, then use a special matmul routine.
        matmul = tf.sparse_tensor_dense_matmul if sparse_input else tf.matmul
        t = matmul(input_tensor, W_0)

        if bias:
            b_0 = tf.get_variable(
                "bias0",
                [output_size],
                initializer=tf.constant_initializer(bias_start))
            t = tf.add(t, b_0)
    return t


@six.add_metaclass(ABCMeta)
class TFPicklingBase(object):
    """Base class for pickling TensorFlow-based scikit-learn estimators.

    This base class defines a few standard attributes to enable fairly
    transparent pickling of TensorFlow models. Note that TensorFlow has
    a custom saving mechanism that makes pickling (and thus using it in
    scikit-learn, etc.) not straightforward.

    NOTE: This base class must come first in the list of classes any child
    class inherits from.

    When pickling an object, if the `self._is_fitted` property is True:

        1. The session at `self._session` is saved using the saver at
            `self._saver` to a temporary file.

        2. The saved data is then read into memory and attached to the
            object state at '_saved_model'.

        3. The fitted state of the model is saved at '_fitted' as True.

    When unpickling the object:

        1. All variables in the state of the object are set using
            `self.__dict__` except the '_saved_model' entry.

        2. If the '_fitted' key is in the state of the object and is True

            2a. The '_saved_model' entry is written to a temporary file.
            2b. A new TF graph is instantiated at `self.graph_`.
            2c. `self._build_tf_graph()`` is called. This instantiates a
                `tf.Saver` at `self._saver` and a `tf.Session` at
                `self._session`.
            2d. The `self._saver` is used to restore previous session to the
                current one.

    To use this base class properly, the child class needs to

        1. Implement the abstract method `self._set_up_graph`. This method
            should build the required TF graph.

        2. Exactly once (e.g., in the `fit` method), instantiate a `tf.Graph`
            at `self.graph_` and then call `self._build_tf_graph` inside the
            `tf.Graph` context block. `self._build_tf_graph` will call
            `self._set_up_graph` and further instantiate the `tf.Saver` and
            `tf.Session`.

        3. After 2. is done, set `self._is_fitted = True`.

        4. Make sure override `__getstate__` to store any extra information
           about your estimator to the state of the object. When doing this,
           call `state = super().__getstate__()` and then append to the
           `state`.

    See the example below and also the MLP classes and base class,
    MLPBaseEstimator.

    Example
    -------

    ```python
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
                self._session.run(tf.initialize_all_variables())
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
            with TemporaryDirectory() as tmpdir:
                # Serialize the model.
                self._saver.save(
                    self._session, os.path.join(tmpdir, 'saved_model'))

                # TF writes a bunch of files so tar them.
                fnames = glob.glob(os.path.join(tmpdir, '*'))
                tarname = os.path.join(tmpdir, 'saved_model.tar')
                with tarfile.open(tarname, "w") as tar:
                    for f in fnames:
                        tar.add(f, arcname=os.path.split(f)[-1])

                # Now read the state back into memory.
                with open(tarname, 'rb') as f:
                    saved_model = f.read()

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
            with TemporaryDirectory() as tmpdir:
                # Write out the serialized tarfile.
                tarname = os.path.join(tmpdir, 'saved_model.tar')
                with open(tarname, 'wb') as f:
                    f.write(state['_saved_model'])

                # Untar it.
                with tarfile.open(tarname, 'r') as tar:
                    tar.extractall(path=tmpdir)

                # And restore.
                self.graph_ = tf.Graph()
                with self.graph_.as_default():
                    self._build_tf_graph()
                    self._saver.restore(
                        self._session, os.path.join(tmpdir, 'saved_model'))

    def _build_tf_graph(self):
        """Build the TF graph, setup model saving and setup a TF session.

        Notes
        -----
        This method initializes a TF Saver and a TF Session via

            ```python
            self._saver = tf.train.Saver()
            self._session = tf.Session()
            ```

        These calls are made after `self._set_up_graph()`` is called.

        See the main class docs for how to properly call this method from a
        child class.
        """
        self._set_up_graph()
        self._saver = tf.train.Saver()
        self._session = tf.Session()

    @abstractmethod
    def _set_up_graph(self):
        """Assemble the TF graph for estimator.

        Notes
        -----
        Child classes should add the TF ops to the graph they want to
        implement here.
        """
        pass
