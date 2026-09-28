import math

import numpy as np
import pytest
from ex01_tfidf import document_frequency, idf, tfidf_matrix, tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

TEXTS = [
    "Платье супер! Платье пришло быстро.",
    "Ткань тонкая, платье маломерит.",
    "Доставка 3 недели, всё OK",
    "!!!",
]


def test_tokenize():
    assert tokenize("Платье СУПЕР, ёлка-2025 ok!") == ["платье", "супер", "ёлка", "2025", "ok"]
    assert tokenize("!!!") == []


def test_document_frequency_counts_documents_not_occurrences():
    df = document_frequency([tokenize(t) for t in TEXTS])
    assert df["платье"] == 2  # в первом документе дважды, но документ один
    assert df["доставка"] == 1


def test_idf():
    assert idf(1, 3) == pytest.approx(math.log(4 / 2) + 1)
    assert idf(3, 3) == pytest.approx(1.0)  # слово во всех документах — минимальный вес


def test_tfidf_matches_sklearn():
    X, vocab = tfidf_matrix(TEXTS)
    sk = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
    expected = sk.fit_transform(TEXTS).toarray()
    assert vocab == list(sk.get_feature_names_out())
    assert np.allclose(X, expected)


def test_rows_are_unit_length_or_zero():
    X, _ = tfidf_matrix(TEXTS)
    norms = np.linalg.norm(X, axis=1)
    assert np.allclose(norms[:3], 1.0) and norms[3] == 0.0
