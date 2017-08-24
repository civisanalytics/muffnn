# Introduction

This project provides multilayer perceptron predictive models, implemented
using [TensorFlow](https://www.tensorflow.org/) and following the
[scikit-learn](http://scikit-learn.org)
[Predictor API](http://scikit-learn.org/stable/developers/contributing.html#apis-of-scikit-learn-objects).

# Installation

Installation with `pip` is recommended:

```bash
pip install muffnn
```

You can install the dependencies via:

```bash
pip install -r requirements.txt
```

If you have trouble installing TensorFlow, see [this page](https://www.tensorflow.org/install/) for more details.

For development, a few additional dependencies are needed:

```bash
pip install -r dev-requirements.txt
```

# Usage

To use the code, import one of the predictor classes and use it as you would
other predictors such as `LogisticRegression`. An example:

```python
from muffnn import MLPClassifier

X, y = load_some_data()

mlp = MLPClassifier()
mlp.fit(X, y)

X_new = load_some_unlabeled_data()
y_pred = mlp.predict(X_new)
```

# Contributing

See `CONTIBUTING.md` for information about contributing to this project.

# License

BSD-3

See `LICENSE.txt` for details.

# Related Tools

* [sklearn.neural_network](http://scikit-learn.org/dev/modules/classes.html#module-sklearn.neural_network)
* [tensorflow.contrib.learn](https://github.com/tensorflow/tensorflow/tree/r0.10/tensorflow/contrib/learn/python/learn)
* [keras.wrappers.scikit_learn](https://keras.io/scikit-learn-api/)

# Contributors

* [Mike Heilman](https://github.com/mheilman/)
* [Walt Askew](https://github.com/waltaskew/)
* [Matt Becker](https://github.com/beckermr/)
* [Bill Lattner](https://github.com/wlattner/)
* [Sam Weiss](https://github.com/samcarlos)
