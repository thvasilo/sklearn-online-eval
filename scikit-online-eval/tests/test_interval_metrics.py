import numpy as np
from numpy.testing import assert_almost_equal
from skgarden import MondrianForestRegressor

from interval_metrics import mean_interval_size, mean_error_rate, IntervalScorer


def test_interval_scorer():
    # Fit a simple linear model
    n_samples = 200
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)
    # simple linear function without noise
    y = np.dot(X, w)

    mfr = MondrianForestRegressor()
    mfr.fit(X, y)
    # Create a scorer that measures the mean interval size
    interval_size_scorer = IntervalScorer(mean_interval_size, sign=-1, kwargs={'confidence': 0.9})
    # Get prediction intervals
    intervals = mfr.predict_interval(X, 0.9)

    interval_size = intervals[:, 1] - intervals[:, 0]
    calc_mean = np.mean(interval_size)
    # Ensure the scorer performs the correct calculation
    assert_almost_equal(interval_size_scorer(mfr, X, y), -1 * calc_mean)


def test_mean_error_rate_metric():
    n_elements = 100
    # Create intervals 1, 5
    intervals = np.concatenate((np.ones((n_elements, 1)), np.ones((n_elements, 1)) * 5), axis=1)
    # Create true values: 90% == 4, 10% = 0
    y1 = np.ones(n_elements - 10) * 4
    y2 = np.zeros(10)
    y_true = np.concatenate((y1, y2))
    # We expect 10% mean error rate
    assert_almost_equal(mean_error_rate(y_true, intervals), 0.1)


def test_mean_interval_size_metric():
    n_elements = 100
    # Create intervals 1, 5
    intervals = np.concatenate((np.ones((n_elements, 1)), np.ones((n_elements, 1)) * 5), axis=1)
    # We expect 4 mean interval size
    assert_almost_equal(mean_interval_size(None, intervals), np.ones(n_elements) * 4)
