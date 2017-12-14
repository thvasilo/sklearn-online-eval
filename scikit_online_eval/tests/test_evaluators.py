import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, make_scorer, r2_score

from evaluation_functions import interleaved_evaluation, prequential_evaluation
from skltemplate import TemplateEstimator

from nose.tools import eq_


# TODO: Create common test setup
def test_interleaved_scores():
    """Test a dummy regressor"""
    X = np.random.random((100, 10))
    y = X[:, 0]**2
    estimator = TemplateEstimator()
    mse_scorer = make_scorer(mean_squared_error)

    scores = interleaved_evaluation(estimator, X, y, mse_scorer)
    assert_almost_equal(scores, np.zeros(X.shape[0]))


def test_interleaved_sklearn():
    """Test interleaved evaluation using a core sklearn model"""
    n_samples = 20
    n_features = 10
    rng = np.random.RandomState(0)
    X = rng.normal(size=(n_samples, n_features))
    w = rng.normal(size=n_features)
    # simple linear function without noise
    y = np.dot(X, w)

    estimator = SGDRegressor()
    mse_scorer = make_scorer(mean_squared_error)
    # TODO: Ideally we want a test with a known interleaved error, but we're not testing
    # the learning algorithm, just checking to see it works and produces output
    interleaved_evaluation(estimator, X, y, mse_scorer)


def test_prequential_window():
    """Test length of produced evaluation when windows are used"""
    n_samples = 100
    window_size = 10
    assert n_samples % window_size == 0  # Meta-test, this needs to hold
    X = np.random.random((n_samples, 10))
    y = X[:, 0] ** 2
    estimator = TemplateEstimator()
    mse_scorer = make_scorer(mean_squared_error)

    scores = prequential_evaluation(estimator, X, y, mse_scorer, window_size=window_size)
    eq_(len(scores),  n_samples / window_size)


def test_prequential_multimetric():
    n_samples = 100
    window_size = 10
    assert n_samples % window_size == 0  # Meta-test, this needs to hold
    X = np.random.random((n_samples, 10))
    y = X[:, 0] ** 2
    estimator = TemplateEstimator()
    mse_scorer = make_scorer(mean_squared_error)
    r2_scorer = make_scorer(r2_score)

    score_dict = {"mse": mse_scorer, "r2": r2_scorer}

    result_dict = prequential_evaluation(estimator, X, y, score_dict, window_size=window_size)

    assert isinstance(result_dict, dict)
    for score_name, score_list in result_dict.items():
        eq_(len(score_list), n_samples / window_size)


def test_prequential_incomplete_window():
    """Test length of produced evaluation when windows are used, and the last
    window has less items than the requested length"""
    n_samples = 109
    window_size = 10
    assert n_samples % window_size != 0
    X = np.random.random((n_samples, 10))
    y = X[:, 0] ** 2
    estimator = TemplateEstimator()
    mse_scorer = make_scorer(mean_squared_error)

    scores = prequential_evaluation(estimator, X, y, mse_scorer, window_size=window_size)
    eq_(len(scores),  np.ceil(n_samples / window_size))
