"""Упражнение 3. Численный градиент, momentum и Adam.

Теория: разделы «Градиент», «Momentum и Adam». Урок 2.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def numerical_grad(f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Градиент функции `f` в точке `x` центральной разностью по каждой координате.

    `x` — вектор float; сама `x` не должна меняться.
    """
    # подсказка: (f(x + h·eᵢ) − f(x − h·eᵢ)) / 2h для каждого базисного вектора eᵢ
    grad = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        e = np.zeros_like(x, dtype=float)
        e.flat[i] = h
        grad.flat[i] = (f(x + e) - f(x - e)) / (2 * h)
    return grad


def momentum_step(
    param: np.ndarray, grad: np.ndarray, velocity: np.ndarray, lr: float, beta: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    """Один шаг SGD с momentum (как в PyTorch): v ← β·v + g, θ ← θ − η·v.

    Возвращает пару (новые параметры, новая скорость); входные массивы не меняются.
    """
    new_velocity = beta * velocity + grad
    return param - lr * new_velocity, new_velocity


@dataclass
class AdamState:
    """Состояние Adam для одного массива параметров: моменты и номер шага."""

    m: np.ndarray
    v: np.ndarray
    t: int = 0

    @classmethod
    def zeros_like(cls, param: np.ndarray) -> "AdamState":
        return cls(np.zeros_like(param, dtype=float), np.zeros_like(param, dtype=float))


@dataclass
class AdamConfig:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0  # развязанная регуляризация, как в AdamW


def adam_step(param: np.ndarray, grad: np.ndarray, state: AdamState, cfg: AdamConfig) -> np.ndarray:
    """Один шаг AdamW. Обновляет `state` (m, v, t) на месте и возвращает новые параметры.

    При `weight_decay = 0` это обычный Adam. Регуляризация применяется к параметрам
    до шага: θ ← θ − η·λ·θ, затем θ ← θ − η·m̂ / (√v̂ + ε).
    """
    # подсказка: не забудьте увеличить state.t до вычисления поправок 1 − βᵗ
    beta1, beta2 = cfg.betas
    state.t += 1
    state.m = beta1 * state.m + (1 - beta1) * grad
    state.v = beta2 * state.v + (1 - beta2) * grad**2
    m_hat = state.m / (1 - beta1**state.t)
    v_hat = state.v / (1 - beta2**state.t)
    decayed = param - cfg.lr * cfg.weight_decay * param
    return decayed - cfg.lr * m_hat / (np.sqrt(v_hat) + cfg.eps)
