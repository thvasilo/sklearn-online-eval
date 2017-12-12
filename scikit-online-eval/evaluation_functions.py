import numpy as np


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


def prequential_evaluation(estimator, X, y, scorer, window_size=1000):
    """
    Prequential evaluation of an estimator by testing then training with
    each example in sequence. If a window size is set the average of a tumbling window
    is reported.
    :param estimator: Has to support a partial_fit function
    :param X: Features, numpy array
    :param y: Labels, numpy column vector
    :param scorer: Has an apply() function that takes a (trained) estimator, X and y as arguments
    and returns a scalar metric.
    :param window_size: The size of the tumbling window, we average the metric every x data points
    :return: A list of scalars, size should be ceil(n_samples / window_size)
    """
    n_samples = X.shape[0]

    test_scores = []
    window_scores = []
    for i in range(n_samples):
        if i == 0:
            # sklearn does not allow prediction on an untrained model
            estimator.partial_fit(X[i, np.newaxis], y[i, np.newaxis])
            window_scores.append(scorer(estimator, X[i, np.newaxis], y[i, np.newaxis]))
        else:
            window_scores.append(scorer(estimator, X[i, np.newaxis], y[i, np.newaxis]))
            # We add a final result every time we have gather window_size values,
            # or we've reached the end of the data (regardless of number of points in window)
            if len(window_scores) == window_size or i == n_samples - 1:
                window_sum = np.sum(window_scores)
                # Divide by current window size here, window is possibly incomplete
                test_scores.append(window_sum / len(window_scores))
                window_scores.clear()
            estimator.partial_fit(X[i, np.newaxis], y[i, np.newaxis])

    return test_scores
