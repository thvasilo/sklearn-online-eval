import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, make_scorer

from base_functions import interleaved_evaluation, prequential_evaluation
from skltemplate import TemplateEstimator

from nose.tools import eq_

# TODO: Create common test setup
def test_prequential_scores():
    """Test a dummy regressor"""
    X = np.random.random((100, 10))
    y = X[:, 0]**2
    estimator = TemplateEstimator()
    mse_scorer = make_scorer(mean_squared_error)

    scores = interleaved_evaluation(estimator, X, y, mse_scorer)
    assert_almost_equal(scores, np.zeros(X.shape[0]))


def test_existing_prequential():
    """Test a core sklearn model"""
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)
    # simple linear function without noise
    y = np.dot(X, w)

    estimator = SGDRegressor()
    mse_scorer = make_scorer(mean_squared_error)
    # Ideally we want a test with a known prequential error, but we're not testing
    # the learning algorithm...
    interleaved_evaluation(estimator, X, y, mse_scorer)


def test_prequential_window():
    n_samples = 100
    window_size = 10
    X = np.random.random((n_samples, 10))
    y = X[:, 0] ** 2
    estimator = TemplateEstimator()
    mse_scorer = make_scorer(mean_squared_error)

    scores = prequential_evaluation(estimator, X, y, mse_scorer, window_size=window_size)
    eq_(len(scores),  n_samples // window_size)


def test_prequential_incomplete_window():
    n_samples = 109
    window_size = 10
    X = np.random.random((n_samples, 10))
    y = X[:, 0] ** 2
    estimator = TemplateEstimator()
    mse_scorer = make_scorer(mean_squared_error)

    scores = prequential_evaluation(estimator, X, y, mse_scorer, window_size=window_size)
    eq_(len(scores),  np.ceil(n_samples / window_size))
