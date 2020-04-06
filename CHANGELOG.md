# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## Unreleased

## [2.3.1] - 2020-04-06

### Fixed
- Fixed an issue with builds failing due to numerical issues (#103)

### Changed
- Increase minimum version of `tensorflow` to v1.15.2 to fix the security vulnerability reported in https://github.com/tensorflow/tensorflow/security/advisories/GHSA-977j-xj7q-2jr9 (#101).
- Dropped support for Python 2.7 and 3.4 (#101).

## [2.3.0] - 2019-11-22

### Changed
- Allowed a recent version of `scikit-learn` (#99).

### Fixed
- Updated tests for changes in new versions of `scipy`, `scikit-learn`, and `flake8` (#98).
- Increased required version of `tensorflow` due to published CVEs in older versions (#98).

## [2.2.0] - 2018-06-07

### Added

- Added `prediction_gradient` method for understanding the impact of different
  features in MLPs with dense inputs.
- Added support for SELU activations with alpha dropout.
- Added sample weights for the `FMClassifier`.
- Added `FMRegressor`.

### Fixed

- Exposed `muffnn.__version__`.
- Fixed bug in `FMClassifier` where it failed for predicting one example.
- Fixed ValueError for type of target in `MLPClassifier` and `FMClassifier` (#90).

### Changed

- Updated requirements on numpy to 1.14 or higher.
- Updated requirements on scipy to 1.0 or higher.

## [2.1.0] - 2018-02-12

### Added

- Added support for the `sample_weight` keyword argument to the `fit`
  method of MLPClassifier and MLPRegressor (#75).

### Changed

- Switched from requiring TensorFlow 1.x to 1.4.x because 1.5.0 was causing
  Travis CI failures with Python 3.6 (#78).

## [2.0.0] - 2018-01-17

### Added

- Added a `transform_layer_index` keyword and `transform` method to the
  MLPClassifier and MLPRegressor to extract features from a hidden layer (#62).

### Changed

- Moved the MLPClassifier and MLPRegressor to using
  [Xavier initialization](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/xavier_initializer) (#68).

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
