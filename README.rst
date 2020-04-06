muffnn
======

|Travis| |PyPI|

.. |Travis| image:: https://img.shields.io/travis/civisanalytics/muffnn/master.svg
   :alt: Build status
   :target: https://travis-ci.org/civisanalytics/muffnn

.. |PyPI| image:: https://img.shields.io/pypi/v/muffnn.svg
   :target: https://pypi.org/project/muffnn/
   :alt: Latest version on PyPI

`scikit-learn <http://scikit-learn.org>`__-compatible neural network
models implemented in `TensorFlow <https://www.tensorflow.org/>`__

Installation
============

This package currently supports Python 3.5, 3.6, and 3.7.

Installation with ``pip`` is recommended:

.. code:: bash

    pip install muffnn

You can install the dependencies via:

.. code:: bash

    pip install -r requirements.txt

If you have trouble installing TensorFlow, see `this
page <https://www.tensorflow.org/install/>`__ for more details.

For development, a few additional dependencies are needed:

.. code:: bash

    pip install -r dev-requirements.txt

Usage
=====

Each estimator in the code follows the scikit-learn API. Thus usage
follows the scikit-learn conventions:

.. code:: python

    from muffnn import MLPClassifier

    X, y = load_some_data()

    mlp = MLPClassifier()
    mlp.fit(X, y)

    X_new = load_some_unlabeled_data()
    y_pred = mlp.predict(X_new)

Further, serialization of the TensorFlow graph and data is handled
automatically when the object is pickled:

.. code:: python

    import pickle

    with open('est.pkl', 'wb') as fp:
        pickle.dump(est, fp)

Contributing
============

See ``CONTIBUTING.md`` for information about contributing to this
project.

License
=======

BSD-3

See ``LICENSE.txt`` for details.
