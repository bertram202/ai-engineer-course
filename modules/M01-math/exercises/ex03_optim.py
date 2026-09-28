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
    raise NotImplementedError("TODO: (f(x + h·eᵢ) − f(x − h·eᵢ)) / 2h для каждого базисного вектора eᵢ")


def momentum_step(
    param: np.ndarray, grad: np.ndarray, velocity: np.ndarray, lr: float, beta: float = 0.9
) -> tuple[np.ndarray, np.ndarray]:
    """Один шаг SGD с momentum (как в PyTorch): v ← β·v + g, θ ← θ − η·v.

    Возвращает пару (новые параметры, новая скорость); входные массивы не меняются.
    """
    raise NotImplementedError("TODO")


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
    raise NotImplementedError("TODO: не забудьте увеличить state.t до вычисления поправок 1 − βᵗ")
