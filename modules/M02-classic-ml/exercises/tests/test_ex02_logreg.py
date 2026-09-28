import numpy as np
import pytest
from ex02_logreg import fit, log_loss, loss_and_grad, sigmoid
from sklearn.linear_model import LogisticRegression


def make_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ [1.5, -2.0, 0.5] + 0.3)))).astype(float)
    return np.column_stack([X, np.ones(n)]), y


def test_sigmoid_is_stable_and_correct():
    z = np.array([-1000.0, -2.0, 0.0, 2.0, 1000.0])
    with np.errstate(over="raise", invalid="raise"):  # никаких переполнений
        s = sigmoid(z)
    assert np.allclose(s, [0.0, 1 / (1 + np.e**2), 0.5, 1 / (1 + np.e**-2), 1.0])


def test_log_loss():
    y = np.array([1.0, 0.0])
    assert log_loss(y, np.array([0.8, 0.3])) == pytest.approx(-(np.log(0.8) + np.log(0.7)) / 2)
    assert np.isfinite(log_loss(y, np.array([0.0, 1.0])))


def test_gradient_matches_numerical():
    X, y = make_data(50)
    w = np.array([0.3, -0.2, 0.1, 0.05])
    _, g = loss_and_grad(w, X, y, l2=0.1)
    h = 1e-6
    num = np.array([(loss_and_grad(w + h * e, X, y, 0.1)[0] - loss_and_grad(w - h * e, X, y, 0.1)[0]) / (2 * h)
                    for e in np.eye(4)])
    assert np.allclose(g, num, atol=1e-7)


def test_bias_is_not_regularized():
    X, y = make_data(50)
    w = np.array([0.0, 0.0, 0.0, 2.0])
    loss0, _ = loss_and_grad(w, X, y, l2=0.0)
    loss1, _ = loss_and_grad(w, X, y, l2=10.0)
    assert loss1 == pytest.approx(loss0)


def test_fit_matches_sklearn_without_regularization():
    X, y = make_data()
    w = fit(X, y, l2=0.0, lr=1.0, steps=5000)
    sk = LogisticRegression(C=np.inf).fit(X[:, :-1], y)
    assert np.allclose(w, np.append(sk.coef_[0], sk.intercept_), atol=1e-3)


def test_fit_matches_sklearn_with_l2():
    X, y = make_data()
    lam = 0.05
    w = fit(X, y, l2=lam, lr=1.0, steps=5000)
    # scikit-learn минимизирует C·Σ log-loss + ½‖w‖², это наша функция при C = 1 / (λ n)
    sk = LogisticRegression(C=1 / (lam * len(y))).fit(X[:, :-1], y)
    assert np.allclose(w, np.append(sk.coef_[0], sk.intercept_), atol=1e-3)
    assert np.linalg.norm(w[:-1]) < np.linalg.norm(fit(X, y, l2=0.0, lr=1.0, steps=5000)[:-1])
