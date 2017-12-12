import numpy as np

from sklearn.metrics.scorer import _BaseScorer


# TODO: Since these break the "y_pred is a column vector convention", is there
# any point in using them as sklearn metrics?
# Maybe it's better to break away from sklearn API
def mean_interval_size(y_true, y_interval):
    """
    Calculates the mean interval size from a interval prediction array.
    :param y_true: Not used, here for compatibility with scorer
    :param y_interval: A n x 2 Numpy array. First column is lower interval, second is upper
    :return: The average size of the intervals
    """
    # y_true is not used, but we keep it to maintain function signature as scorers expect it
    _check_interval_array(y_interval) 

    interval_size = y_interval[:, 1] - y_interval[:, 0]  # Guaranteed to be > 0

    return np.average(interval_size)


def mean_error_rate(y_true, y_interval):
    """
    Calculates the mean error rate in the provided intervals
    :param y_true: A numpy column array of true values
    :param y_interval: A n x 2 Numpy array. First column is lower interval, second is upper
    :return: The ratio of values in y_true that are within their corresponding interval in y_interval
    """
    _check_interval_array(y_interval)

    wrong_intervals = ((y_true < y_interval[:, 0]) | (y_true > y_interval[:, 1])).sum()

    return wrong_intervals / y_true.shape[0]


def _check_interval_array(y_interval):
    # check_consistent_length(y_true, y_interval)
    assert y_interval.shape[1] == 2
    assert np.all(y_interval[:, 0] <= y_interval[:, 1])


# TODO: This doesn't look like a good way to do this currently
# Either properly pass kwargs to predict s.t. quantiles and intervals are calculated,
# or adjust/abandon this class. Also inheriting a private member is not cool.
class IntervalScorer(_BaseScorer):
    def __init__(self, score_func, sign, kwargs):
        super().__init__(score_func, sign, kwargs)

    def __call__(self, estimator, X, y_true, sample_weight=None):
        super().__call__(estimator, X, y_true, sample_weight=sample_weight)

        y_pred = estimator.predict_interval(X, **self._kwargs)

        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:  # tvas: The original passes the kwargs here, either do it that way or don't inherit
            return self._sign * self._score_func(y_true, y_pred)