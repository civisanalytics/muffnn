# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [1.2.0] - 2017-09-21

### Added

- Python 2.7 compatibility (#57).
- Added a `monitor` keyword to the autoencoder (#58).
- Added a factorization machine classifier (#50).

### Changed

- Moved to Travis instead of CircleCI (#57).
- Upgraded to TensorFlow 1.3.X.
- Upgraded to numpy 1.13.1.
- Upgraded to scipy 0.19.1.
- Upgraded to scikit-learn 0.19.0.
- Upgraded to python 3.6.2.
- Upgraded requirements to match python 2 properly (#59).

### Fixed

- Hid slow import of `tf.contrib` (#54).

## [1.1.2] - 2017-05-22

### Changed

- Upgraded to TensorFlow 1.1.X.

### Fixed

- Fixed bug in grid search over solver settings.
- Fixed bug in `classes_` attribute for multilabel MLP problems.

## [1.1.1] - 2017-03-27

### Fixed

- Added `MANIFEST.in` to fix python packaging bug.

## [1.1.0] - 2017-03-27

### Added

- Add ability to set the solver and its parameters for the `MLPClassifier` and `MLPRegressor`.

### Changed

- Removed Docker build.
- Added `install_requires` to `setup.py`.
- Updated tests of MLP base class to silence `pytest` warning.

### Fixed

- Fixed `score` method for multilabel classification.

## [1.0.0] - 2017-02-23

### Changed

- Upgraded to TensorFlow 1.0.0.

## [0.2.0] - 2016-11-29

### Added

- Add an autencoder implementation.
- Add optional monitor functionality for MLP classes, for logging, early
  stopping, checkpointing, etc.
- Add top-level base class for pickling TensorFlow models.
- Add partial fitting functionality for the MLP.
- Add support for missing labels during multilabel classification.

### Changed

- Make sparse input more efficient in MLP classes.
- Stop adding dropout nodes to MLP graphs if `keep_prob` is 1.
- Change `dropout` keyword argument to `keep_prob` for consistency with
  TensorFlow.
- Updated dependencies (notably, scikit-learn and TensorFlow).

### Fixed

- `LabelEncoder` in the MLPClassifier is pickled properly.
- Fix multilabel classification, which was broken previously.

## [0.1.0] - 2016-08-25

### Added

- Multilayer Perceptron Classifier and Regressor implementations.
