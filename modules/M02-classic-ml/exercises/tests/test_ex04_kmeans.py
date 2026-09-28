import numpy as np
import pytest
from ex04_kmeans import assign, inertia, kmeans, silhouette
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score


def test_assign_and_inertia():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0]])
    centers = np.array([[0.0, 0.5], [10.0, 10.0]])
    labels = assign(X, centers)
    assert labels.tolist() == [0, 0, 1]
    assert inertia(X, centers, labels) == pytest.approx(0.5)


def test_kmeans_finds_blobs():
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)
    centers, labels = kmeans(X, 3, seed=1)
    assert centers.shape == (3, 2) and labels.shape == (300,)
    assert adjusted_rand_score(y, labels) > 0.95
    # центры — средние своих точек, назначения — ближайшие центры
    assert np.array_equal(assign(X, centers), labels)
    for c in range(3):
        assert np.allclose(centers[c], X[labels == c].mean(axis=0))


def test_kmeans_is_deterministic_and_uses_seed():
    X, _ = make_blobs(n_samples=200, centers=4, random_state=3)
    a = kmeans(X, 4, seed=5)
    b = kmeans(X, 4, seed=5)
    assert np.array_equal(a[1], b[1]) and np.allclose(a[0], b[0])


def test_kmeans_does_not_modify_input():
    X, _ = make_blobs(n_samples=50, centers=2, random_state=0)
    X_copy = X.copy()
    kmeans(X, 2)
    assert np.array_equal(X, X_copy)


def test_silhouette_matches_sklearn():
    X, _ = make_blobs(n_samples=120, centers=3, cluster_std=1.5, random_state=2)
    labels = kmeans(X, 3)[1]
    assert silhouette(X, labels) == pytest.approx(silhouette_score(X, labels))


def test_silhouette_singleton_cluster_is_zero():
    X = np.array([[0.0], [0.1], [5.0]])
    labels = np.array([0, 0, 1])
    assert silhouette(X, labels) == pytest.approx(silhouette_score(X, labels))
