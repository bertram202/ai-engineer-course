"""Упражнение 4. k-means (алгоритм Ллойда) и силуэт с нуля.

Теория: раздел «Кластеризация: k-means». Урок 4. Расстояние — евклидово.
"""

import numpy as np


def assign(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Номер ближайшего центра для каждой точки."""
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1)


def inertia(X: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    """Сумма квадратов расстояний от точек до центров их кластеров."""
    return float(((X - centers[labels]) ** 2).sum())


def kmeans(X: np.ndarray, k: int, n_iter: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Алгоритм Ллойда. Начальные центры — `k` различных точек, выбранных
    `np.random.default_rng(seed).choice(len(X), k, replace=False)`.

    Чередуем «назначить ближайший центр» и «центр = среднее своих точек», пока назначения меняются
    (не больше `n_iter` итераций). Если кластер опустел, его центр остаётся прежним.
    Возвращает (центры [k, d], метки [n]).
    """
    centers = X[np.random.default_rng(seed).choice(len(X), k, replace=False)].astype(float)
    labels = assign(X, centers)
    for _ in range(n_iter):
        for c in range(k):
            if np.any(labels == c):
                centers[c] = X[labels == c].mean(axis=0)
        new_labels = assign(X, centers)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
    return centers, labels


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Средний силуэт: для точки s = (b − a) / max(a, b), где a — среднее расстояние до других точек
    своего кластера, b — наименьшее среднее расстояние до точек другого кластера.
    Для точки, единственной в своём кластере, s = 0.
    """
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    clusters = np.unique(labels)
    s = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        if own.sum() == 1:
            continue
        a = d[i, own].sum() / (own.sum() - 1)
        b = min(d[i, labels == c].mean() for c in clusters if c != labels[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())
