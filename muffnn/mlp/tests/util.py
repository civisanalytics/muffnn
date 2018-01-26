import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split


def assert_sample_weights_work(make_dataset_func, dataset_kwargs,
                               make_classifier_func):
    """Ensure we handle sample weights for the given classifier.

    First asserts that estimators trained with no sample weights are
    equivalent to estimators trained with sample weights of all 1s.
    Further asserts that when trying to learn two datasets simultaneously,
    the estimator performs best on the dataset weighted most heavily.

    Parameters
    ----------
    make_dataset_func : function
        Function that when called with ``dataset_kwargs`` will
        return an X and y for testing the given estimator.
    dataset_kwargs : dict
        Arguments that will be passed to ``make_dataset_func``
        when invoked.
    make_classifier_func : function
        Function that when called returns the estimator to be tested.
    """
    X, y = make_dataset_func(random_state=42, **dataset_kwargs)
    X2, y2 = make_dataset_func(random_state=24, **dataset_kwargs)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    # A classifier with no sample weights and a classifier with
    # sample weights of all 1 should be equivalent.
    sample_weight = np.ones(X_train.shape[0])
    clf1 = make_classifier_func().fit(X_train, y_train)
    clf2 = make_classifier_func().fit(
        X_train, y_train,
        sample_weight=sample_weight,
    )
    if hasattr(clf1, 'predict_proba'):
        disagreement = (clf1.predict_proba(X_test) -
                        clf2.predict_proba(X_test))
        assert np.abs(disagreement).mean() < 0.01
    else:
        correlation, _ = pearsonr(clf1.predict(X_test),
                                  clf2.predict(X_test))
        assert correlation > .99

    # When trying to learn two different datasets at once, the
    # classifier should perform best on the dataset more heavily weighted.
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.33, random_state=42)

    # Favor learning y | X
    clf1 = make_classifier_func().fit(
        np.concatenate((X_train, X2_train)),
        np.concatenate((y_train, y2_train)),
        sample_weight=([4] * len(X_train)) + ([.25] * len(X2_train)),
    )
    # Favor learning y2 | X2
    clf2 = make_classifier_func().fit(
        np.concatenate((X_train, X2_train)),
        np.concatenate((y_train, y2_train)),
        sample_weight=([.25] * len(X_train)) + ([4] * len(X2_train)),
    )

    assert clf1.score(X_test, y_test) > clf1.score(X2_test, y2_test)
    assert clf2.score(X2_test, y2_test) > clf2.score(X_test, y_test)
