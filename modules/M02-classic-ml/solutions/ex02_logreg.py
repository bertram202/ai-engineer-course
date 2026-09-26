"""Упражнение 2. Логистическая регрессия с L2-регуляризацией с нуля.

Теория: раздел «Линейные модели». Урок 2.
Соглашение: последний столбец матрицы `X` — единицы (свободный член), его вес не регуляризуется.
Функция потерь: средний log-loss + (λ/2)·‖w без свободного члена‖².
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Сигмоида без переполнения при больших |z|."""
    # подсказка: для z < 0 используйте эквивалентную форму e^z / (1 + e^z)
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1 + ez)
    return out


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """Средняя бинарная кросс-энтропия; вероятности обрезаются до [eps, 1 − eps]."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def loss_and_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> tuple[float, np.ndarray]:
    """Значение функции потерь (с регуляризацией) и её градиент по `w`."""
    p = sigmoid(X @ w)
    reg = w.copy()
    reg[-1] = 0.0  # свободный член не штрафуем
    loss = log_loss(y, p) + 0.5 * l2 * float(reg @ reg)
    grad = X.T @ (p - y) / len(y) + l2 * reg
    return loss, grad


def fit(X: np.ndarray, y: np.ndarray, l2: float = 0.0, lr: float = 0.5, steps: int = 3000) -> np.ndarray:
    """Градиентный спуск из нулевой точки; возвращает веса."""
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        w -= lr * loss_and_grad(w, X, y, l2)[1]
    return w
