import numpy as np
import pytest
from ex03_optim import AdamConfig, AdamState, adam_step, momentum_step, numerical_grad


def test_numerical_grad_quadratic():
    a = np.array([[3.0, 1.0], [1.0, 2.0]])
    x = np.array([0.5, -1.0])
    g = numerical_grad(lambda v: 0.5 * v @ a @ v, x)
    assert np.allclose(g, a @ x, atol=1e-7)


def test_numerical_grad_does_not_modify_input():
    x = np.array([1.0, 2.0, 3.0])
    numerical_grad(lambda v: float(np.sum(np.sin(v))), x)
    assert np.array_equal(x, [1.0, 2.0, 3.0])


def test_numerical_grad_sin():
    x = np.array([0.0, 1.0, 2.0])
    assert np.allclose(numerical_grad(lambda v: float(np.sum(np.sin(v))), x), np.cos(x), atol=1e-8)


def test_momentum_step_pytorch_convention():
    p, v = np.array([1.0, -2.0]), np.array([0.5, 0.5])
    g = np.array([0.2, -0.4])
    new_p, new_v = momentum_step(p, g, v, lr=0.1, beta=0.9)
    assert np.allclose(new_v, [0.65, 0.05])
    assert np.allclose(new_p, [0.935, -2.005])
    assert np.array_equal(p, [1.0, -2.0]) and np.array_equal(v, [0.5, 0.5])


def test_adam_first_step_is_lr_times_sign():
    p = np.array([1.0, -1.0, 0.5])
    g = np.array([1e-4, -30.0, 2.0])  # масштабы градиентов отличаются на 5 порядков
    state = AdamState.zeros_like(p)
    new_p = adam_step(p, g, state, AdamConfig(lr=0.01))
    assert np.allclose(new_p - p, -0.01 * np.sign(g), atol=1e-6)
    assert state.t == 1


def test_adam_matches_pytorch_adamw():
    # эталон получен torch.optim.AdamW(lr=0.01, betas=(0.9, 0.95), eps=1e-8) на тех же данных
    rng = np.random.default_rng(0)
    p0 = rng.normal(size=5)
    grads = rng.normal(size=(10, 5))
    expected = {
        0.0: [0.161658, -0.169732, 0.65852, 0.106688, -0.527007],
        0.1: [0.160317, -0.168193, 0.652076, 0.105485, -0.521842],
    }
    for wd, target in expected.items():
        p, state = p0.copy(), AdamState.zeros_like(p0)
        cfg = AdamConfig(lr=0.01, betas=(0.9, 0.95), weight_decay=wd)
        for g in grads:
            p = adam_step(p, g, state, cfg)
        assert np.allclose(p, target, atol=2e-6), f"weight_decay={wd}"
        assert state.t == 10


def test_weight_decay_is_decoupled():
    p = np.array([2.0, -4.0])
    state = AdamState.zeros_like(p)
    new_p = adam_step(p, np.zeros(2), state, AdamConfig(lr=0.1, weight_decay=0.5))
    # градиент нулевой → остаётся только затухание θ ← θ − η·λ·θ
    assert np.allclose(new_p, p * (1 - 0.1 * 0.5))


def test_adam_minimizes_ill_conditioned_quadratic():
    lam = np.array([1.0, 1000.0])
    p, state, cfg = np.array([3.0, 3.0]), AdamState.zeros_like(np.zeros(2)), AdamConfig(lr=0.05)
    for _ in range(2000):
        p = adam_step(p, lam * p, state, cfg)
    assert np.abs(p).max() < 1e-2
    assert p == pytest.approx(np.zeros(2), abs=1e-2)
