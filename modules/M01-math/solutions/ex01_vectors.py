"""Упражнение 1. Векторы, косинус и поиск ближайших.

Теория: разделы «Векторы и скалярное произведение», «Проекции». Урок 1.
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусная близость двух векторов. Если один из векторов нулевой — ValueError."""
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        raise ValueError("косинус с нулевым вектором не определён")
    return float(a @ b / norm)


def normalize_rows(m: np.ndarray) -> np.ndarray:
    """Каждая строка матрицы, приведённая к единичной длине (исходная матрица не меняется)."""
    # подсказка: np.linalg.norm(..., axis=1, keepdims=True)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def top_k_similar(query: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
    """Индексы `k` строк `matrix`, самых близких к `query` по косинусу, — от самой близкой.

    Строки и запрос не обязательно нормированы. Без цикла по строкам: одно матричное умножение.
    """
    scores = normalize_rows(matrix) @ (query / np.linalg.norm(query))
    return [int(i) for i in np.argsort(-scores, kind="stable")[:k]]


def project(v: np.ndarray, onto: np.ndarray) -> np.ndarray:
    """Проекция вектора `v` на подпространство, натянутое на столбцы `onto` формы [d, k].

    Столбцы линейно независимы, но не обязательно ортонормированы.
    """
    # подсказка: ортонормируйте столбцы через np.linalg.qr, затем Q (Qᵀ v)
    q, _ = np.linalg.qr(onto)
    return q @ (q.T @ v)
