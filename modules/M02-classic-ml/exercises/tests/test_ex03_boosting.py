import numpy as np
import pytest
from ex03_boosting import Boosting, fit_boosting, predict_boosting
from sklearn.ensemble import GradientBoostingRegressor


def make_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 2))
    return X, np.sin(X[:, 0]) + 0.3 * X[:, 1] ** 2 + rng.normal(0, 0.1, n)


def test_structure():
    X, y = make_data()
    m = fit_boosting(X, y, n_estimators=7, learning_rate=0.2, max_depth=2)
    assert isinstance(m, Boosting)
    assert len(m.trees) == 7 and m.learning_rate == 0.2
    assert m.init == pytest.approx(y.mean())
    assert all(t.get_depth() <= 2 for t in m.trees)


def test_zero_trees_predicts_mean():
    X, y = make_data()
    m = fit_boosting(X, y, n_estimators=5)
    assert np.allclose(predict_boosting(m, X, n_trees=0), y.mean())


def test_training_error_decreases():
    X, y = make_data()
    m = fit_boosting(X, y, n_estimators=40)
    mse = [np.mean((predict_boosting(m, X, n_trees=k) - y) ** 2) for k in range(0, 41, 5)]
    assert all(a > b for a, b in zip(mse, mse[1:]))


def test_matches_sklearn_gradient_boosting():
    X, y = make_data()
    ours = fit_boosting(X, y, n_estimators=50, learning_rate=0.1, max_depth=3)
    sk = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=0).fit(X, y)
    X_new = make_data(100, seed=1)[0]
    assert np.allclose(predict_boosting(ours, X_new), sk.predict(X_new), atol=1e-8)
