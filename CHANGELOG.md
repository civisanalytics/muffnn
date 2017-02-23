# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

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
