import numpy as np
import pytest
from ex01_vectors import cosine_similarity, normalize_rows, project, top_k_similar


def test_cosine_basic_cases():
    a = np.array([3.0, 4.0])
    assert cosine_similarity(a, np.array([4.0, 3.0])) == pytest.approx(0.96)
    assert cosine_similarity(a, -a) == pytest.approx(-1.0)
    assert cosine_similarity(a, np.array([-4.0, 3.0])) == pytest.approx(0.0)


def test_cosine_ignores_length():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=16), rng.normal(size=16)
    assert cosine_similarity(a, 7 * b) == pytest.approx(cosine_similarity(a, b))
    assert isinstance(cosine_similarity(a, b), float)


def test_cosine_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity(np.zeros(3), np.ones(3))


def test_normalize_rows():
    m = np.array([[3.0, 4.0], [0.0, 2.0]])
    out = normalize_rows(m)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert np.allclose(out[0], [0.6, 0.8])
    assert m[0, 0] == 3.0  # исходная матрица не изменилась


def test_top_k_uses_cosine_not_dot():
    docs = np.array([
        [10.0, 10.0],   # длинный, угол 45°
        [1.0, 0.1],     # короткий, почти вдоль x
        [0.0, 1.0],
    ])
    # по скалярному произведению выиграл бы длинный вектор 0, по косинусу — вектор 1
    assert top_k_similar(np.array([1.0, 0.0]), docs, 2) == [1, 0]


def test_top_k_on_random_data():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(50, 8))
    q = rng.normal(size=8)
    expected = sorted(range(50), key=lambda i: -(m[i] @ q) / np.linalg.norm(m[i]))[:5]
    result = top_k_similar(q, m, 5)
    assert result == expected
    assert all(isinstance(i, int) for i in result)


def test_project_onto_vector():
    v = np.array([2.0, 3.0])
    p = project(v, np.array([[1.0], [0.5]]))
    assert np.allclose(p, [2.8, 1.4])
    assert (v - p) @ np.array([1.0, 0.5]) == pytest.approx(0.0, abs=1e-12)


def test_project_onto_plane_with_non_orthonormal_basis():
    basis = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])  # плоскость z = 0, базис не ортогональный
    assert np.allclose(project(np.array([1.0, 2.0, 3.0]), basis), [1.0, 2.0, 0.0])


def test_projection_is_idempotent():
    rng = np.random.default_rng(3)
    basis = rng.normal(size=(6, 2))
    v = rng.normal(size=6)
    p = project(v, basis)
    assert np.allclose(project(p, basis), p)
    assert np.allclose(basis.T @ (v - p), 0)  # остаток перпендикулярен подпространству
