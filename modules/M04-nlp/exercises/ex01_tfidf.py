"""Упражнение 1. TF-IDF с нуля.

Теория: раздел «Мешок слов и TF-IDF». Урок 1.
Формулы — как в scikit-learn по умолчанию (smooth_idf=True, norm="l2"), чтобы результат можно было сверить.
"""

import math
import re
from collections import Counter

import numpy as np

TOKEN = re.compile(r"[0-9a-zа-яё]+")


def tokenize(text: str) -> list[str]:
    """Слова в нижнем регистре: последовательности цифр, латинских и русских букв (включая ё)."""
    raise NotImplementedError("TODO")


def document_frequency(docs: list[list[str]]) -> Counter:
    """Для каждого слова — в скольких документах оно встречается (не сколько раз всего)."""
    raise NotImplementedError("TODO: считайте set(doc) каждого документа")


def idf(df: int, n_docs: int) -> float:
    """Сглаженный IDF, как в scikit-learn: ln((1 + n) / (1 + df)) + 1."""
    raise NotImplementedError("TODO")


def tfidf_matrix(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    """Матрица TF-IDF [документы, слова] и словарь (слова по алфавиту).

    TF — число вхождений слова в документ. Каждая строка нормирована до единичной длины (L2);
    нулевые строки остаются нулевыми.
    """
    raise NotImplementedError("TODO")
