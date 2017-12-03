# muffnn

[scikit-learn](http://scikit-learn.org)-compatible neural network models in implemented in [TensorFlow](https://www.tensorflow.org/)

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

Each estimator in the code follows the scikit-learn API. Thus usage follows the scikit-learn conventions:

```python
from muffnn import MLPClassifier

X, y = load_some_data()

mlp = MLPClassifier()
mlp.fit(X, y)

X_new = load_some_unlabeled_data()
y_pred = mlp.predict(X_new)
```

Further, serialization of the TensorFlow graph and data is handled automatically when the object is pickled:

```python
import pickle

with open('est.pkl', 'wb') as fp:
    pickle.dump(est, fp)
```

# Contributing

See `CONTIBUTING.md` for information about contributing to this project.

# License

BSD-3

See `LICENSE.txt` for details.
