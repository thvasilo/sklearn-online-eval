import numpy as np
from collections import defaultdict


def interleaved_evaluation(estimator, X, y, scorer):
    """Interleaved evaluation of an estimator by testing then training with
    each example in sequence.
    :param estimator: Has to support a partial_fit function
    :param X: Features, numpy array
    :param y: Labels, numpy column vector
    :param scorer: Has an apply() function that takes a (trained) estimator, X and y as arguments
    and returns a scalar metric.
    :return: A list of scalars, size should be n_samples
    """
    n_samples = X.shape[0]

    test_scores = []
    for i in range(n_samples):
        if i == 0:
            # sklearn does not allow prediction on an untrained model
            estimator.partial_fit(X[i, np.newaxis], y[i, np.newaxis])
            test_scores.append(scorer(estimator, X[i, np.newaxis], y[i, np.newaxis]))
        else:
            test_scores.append(scorer(estimator, X[i, np.newaxis], y[i, np.newaxis]))
            estimator.partial_fit(X[i, np.newaxis], y[i, np.newaxis])

    return test_scores


def prequential_evaluation(estimator, X, y, scoring, window_size=1000):
    """
    Prequential evaluation of an estimator by testing then training with
    each example in sequence. If a window size is set the average of a tumbling window
    is reported.
    :param estimator: Has to support a partial_fit function
    :param X: Features, numpy array
    :param y: Labels, numpy column vector
    :param scoring: callable or dict
        If a callable, it takes a (trained) estimator, X and y as arguments
        and returns a scalar metric.
        dicts should have a mapping metric_name : callable (as above). can be used to report multiple
        metrics.
    :param window_size: The size of the tumbling window, we average the metric every x data points
    :return: List or dict
        If scoring is a callable, will return a list of scores, size should be ceil(n_samples / window_size)
        If scoring is a dict, will return a dict {metric_name: list of scores}, list size should be
        ceil(n_samples / window_size)
    """
    n_samples = X.shape[0]

    if isinstance(scoring, dict):
        # Scores are dicts {metric_name: list of scalar scores}
        window_scores = defaultdict(list)
        test_scores = defaultdict(list)
    else:
        # Scores are lists of scalar scores
        window_scores = []
        test_scores = []
    window_count = 0
    for i in range(n_samples):
        if i == 0:
            # sklearn does not allow prediction on an untrained model
            estimator.partial_fit(X[i, np.newaxis], y[i, np.newaxis])
        if isinstance(scoring, dict):
            for score_name, score_func in scoring.items():
                window_scores[score_name].append(score_func(estimator, X[i, np.newaxis], y[i, np.newaxis]))
        else:
            window_scores.append(scoring(estimator, X[i, np.newaxis], y[i, np.newaxis]))
        if i == 0:
            continue
        window_count += 1 # Easier than checking inside window_scores
        # We add a final result every time we have gather window_size values,
        # or we've reached the end of the data (regardless of number of points in window)
        if window_count == window_size or i == n_samples - 1:
            if isinstance(scoring, dict):
                assert isinstance(window_scores, dict)
                for score_name, score_list in window_scores.items():
                    window_sum = np.sum(score_list)
                    test_scores[score_name].append(window_sum / len(score_list))
            else:
                window_sum = np.sum(window_scores)
                # Divide by current window size here, window is possibly incomplete
                test_scores.append(window_sum / len(window_scores))
            window_scores.clear()
            window_count = 0
        estimator.partial_fit(X[i, np.newaxis], y[i, np.newaxis])

    return test_scores
