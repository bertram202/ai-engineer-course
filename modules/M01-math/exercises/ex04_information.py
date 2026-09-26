"""Упражнение 4. Softmax, энтропия, KL и перплексия.

Теория: разделы «Softmax», «Температура и сэмплирование», «Теория информации». Урок 3.
Все логарифмы натуральные (наты).
"""

from collections.abc import Sequence

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Устойчивый softmax с температурой. Температура должна быть > 0, иначе ValueError."""
    raise NotImplementedError("TODO")


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Логарифм softmax без переполнения: z − logsumexp(z)."""
    raise NotImplementedError("TODO")


def entropy(p: np.ndarray) -> float:
    """Энтропия распределения в натах; слагаемые с p = 0 дают 0."""
    raise NotImplementedError("TODO")


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """Кросс-энтропия H(p, q) = −Σ p log q (слагаемые с p = 0 пропускаются)."""
    raise NotImplementedError("TODO")


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p ‖ q) = Σ p log(p / q) (слагаемые с p = 0 пропускаются)."""
    raise NotImplementedError("TODO")


def perplexity(logprobs: Sequence[float]) -> float:
    """Перплексия по натуральным логарифмам вероятностей токенов текста."""
    raise NotImplementedError("TODO: экспонента от средней отрицательной log-вероятности")


def top_p_filter(probs: np.ndarray, top_p: float) -> np.ndarray:
    """Nucleus-фильтр: оставить наименьший набор самых вероятных токенов с суммой ≥ `top_p`,
    остальным дать 0 и перенормировать. Форма и порядок элементов сохраняются.
    """
    raise NotImplementedError("TODO")
