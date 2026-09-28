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
    return TOKEN.findall(text.lower())


def document_frequency(docs: list[list[str]]) -> Counter:
    """Для каждого слова — в скольких документах оно встречается (не сколько раз всего)."""
    # подсказка: считайте set(doc) каждого документа
    df: Counter = Counter()
    for doc in docs:
        df.update(set(doc))
    return df


def idf(df: int, n_docs: int) -> float:
    """Сглаженный IDF, как в scikit-learn: ln((1 + n) / (1 + df)) + 1."""
    return math.log((1 + n_docs) / (1 + df)) + 1


def tfidf_matrix(texts: list[str]) -> tuple[np.ndarray, list[str]]:
    """Матрица TF-IDF [документы, слова] и словарь (слова по алфавиту).

    TF — число вхождений слова в документ. Каждая строка нормирована до единичной длины (L2);
    нулевые строки остаются нулевыми.
    """
    docs = [tokenize(t) for t in texts]
    df = document_frequency(docs)
    vocab = sorted(df)
    index = {w: j for j, w in enumerate(vocab)}
    X = np.zeros((len(docs), len(vocab)))
    for i, doc in enumerate(docs):
        for w, count in Counter(doc).items():
            X[i, index[w]] = count * idf(df[w], len(docs))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, norms, out=np.zeros_like(X), where=norms > 0), vocab
