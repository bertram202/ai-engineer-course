"""Упражнение 2. Логистическая регрессия с L2-регуляризацией с нуля.

Теория: раздел «Линейные модели». Урок 2.
Соглашение: последний столбец матрицы `X` — единицы (свободный член), его вес не регуляризуется.
Функция потерь: средний log-loss + (λ/2)·‖w без свободного члена‖².
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Сигмоида без переполнения при больших |z|."""
    raise NotImplementedError("TODO: для z < 0 используйте эквивалентную форму e^z / (1 + e^z)")


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """Средняя бинарная кросс-энтропия; вероятности обрезаются до [eps, 1 − eps]."""
    raise NotImplementedError("TODO")


def loss_and_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> tuple[float, np.ndarray]:
    """Значение функции потерь (с регуляризацией) и её градиент по `w`."""
    raise NotImplementedError("TODO")


def fit(X: np.ndarray, y: np.ndarray, l2: float = 0.0, lr: float = 0.5, steps: int = 3000) -> np.ndarray:
    """Градиентный спуск из нулевой точки; возвращает веса."""
    raise NotImplementedError("TODO")
