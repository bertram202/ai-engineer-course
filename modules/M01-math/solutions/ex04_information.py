"""Упражнение 4. Softmax, энтропия, KL и перплексия.

Теория: разделы «Softmax», «Температура и сэмплирование», «Теория информации». Урок 3.
Все логарифмы натуральные (наты).
"""

from collections.abc import Sequence

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Устойчивый softmax с температурой. Температура должна быть > 0, иначе ValueError."""
    if temperature <= 0:
        raise ValueError("температура должна быть положительной")
    z = np.asarray(logits, dtype=float) / temperature
    e = np.exp(z - z.max())
    return e / e.sum()


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Логарифм softmax без переполнения: z − logsumexp(z)."""
    z = np.asarray(logits, dtype=float)
    m = z.max()
    return z - (m + np.log(np.sum(np.exp(z - m))))


def entropy(p: np.ndarray) -> float:
    """Энтропия распределения в натах; слагаемые с p = 0 дают 0."""
    p = np.asarray(p, dtype=float)
    nz = p[p > 0]
    return float(-np.sum(nz * np.log(nz)))


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """Кросс-энтропия H(p, q) = −Σ p log q (слагаемые с p = 0 пропускаются)."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(q[mask])))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p ‖ q) = Σ p log(p / q) (слагаемые с p = 0 пропускаются)."""
    return cross_entropy(p, q) - entropy(p)


def perplexity(logprobs: Sequence[float]) -> float:
    """Перплексия по натуральным логарифмам вероятностей токенов текста."""
    # подсказка: экспонента от средней отрицательной log-вероятности
    return float(np.exp(-np.mean(logprobs)))


def top_p_filter(probs: np.ndarray, top_p: float) -> np.ndarray:
    """Nucleus-фильтр: оставить наименьший набор самых вероятных токенов с суммой ≥ `top_p`,
    остальным дать 0 и перенормировать. Форма и порядок элементов сохраняются.
    """
    probs = np.asarray(probs, dtype=float)
    order = np.argsort(-probs, kind="stable")
    n_keep = int(np.searchsorted(np.cumsum(probs[order]), top_p - 1e-12) + 1)
    kept = np.zeros_like(probs)
    kept[order[:n_keep]] = probs[order[:n_keep]]
    return kept / kept.sum()
