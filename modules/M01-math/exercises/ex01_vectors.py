"""Упражнение 1. Векторы, косинус и поиск ближайших.

Теория: разделы «Векторы и скалярное произведение», «Проекции». Урок 1.
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусная близость двух векторов. Если один из векторов нулевой — ValueError."""
    raise NotImplementedError("TODO")


def normalize_rows(m: np.ndarray) -> np.ndarray:
    """Каждая строка матрицы, приведённая к единичной длине (исходная матрица не меняется)."""
    raise NotImplementedError("TODO: np.linalg.norm(..., axis=1, keepdims=True)")


def top_k_similar(query: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
    """Индексы `k` строк `matrix`, самых близких к `query` по косинусу, — от самой близкой.

    Строки и запрос не обязательно нормированы. Без цикла по строкам: одно матричное умножение.
    """
    raise NotImplementedError("TODO")


def project(v: np.ndarray, onto: np.ndarray) -> np.ndarray:
    """Проекция вектора `v` на подпространство, натянутое на столбцы `onto` формы [d, k].

    Столбцы линейно независимы, но не обязательно ортонормированы.
    """
    raise NotImplementedError("TODO: ортонормируйте столбцы через np.linalg.qr, затем Q (Qᵀ v)")
