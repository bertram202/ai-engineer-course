"""Упражнение 4. k-means (алгоритм Ллойда) и силуэт с нуля.

Теория: раздел «Кластеризация: k-means». Урок 4. Расстояние — евклидово.
"""

import numpy as np


def assign(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Номер ближайшего центра для каждой точки."""
    raise NotImplementedError("TODO")


def inertia(X: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    """Сумма квадратов расстояний от точек до центров их кластеров."""
    raise NotImplementedError("TODO")


def kmeans(X: np.ndarray, k: int, n_iter: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Алгоритм Ллойда. Начальные центры — `k` различных точек, выбранных
    `np.random.default_rng(seed).choice(len(X), k, replace=False)`.

    Чередуем «назначить ближайший центр» и «центр = среднее своих точек», пока назначения меняются
    (не больше `n_iter` итераций). Если кластер опустел, его центр остаётся прежним.
    Возвращает (центры [k, d], метки [n]).
    """
    raise NotImplementedError("TODO")


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Средний силуэт: для точки s = (b − a) / max(a, b), где a — среднее расстояние до других точек
    своего кластера, b — наименьшее среднее расстояние до точек другого кластера.
    Для точки, единственной в своём кластере, s = 0.
    """
    raise NotImplementedError("TODO")
