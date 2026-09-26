"""Упражнение 1. Метрики классификации, ROC-AUC, порог и калибровка — с нуля.

Теория: раздел «Метрики». Урок 1. Все функции принимают numpy-массивы; метки — 0 и 1.
"""

import numpy as np


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Матрица ошибок бинарной классификации: словарь с ключами tp, fp, fn, tn (int)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "tp": int(np.sum((y_pred == 1) & (y_true == 1))),
        "fp": int(np.sum((y_pred == 1) & (y_true == 0))),
        "fn": int(np.sum((y_pred == 0) & (y_true == 1))),
        "tn": int(np.sum((y_pred == 0) & (y_true == 0))),
    }


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall и F1 положительного класса. Если знаменатель равен нулю, метрика равна 0."""
    c = confusion_counts(y_true, y_pred)
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC как вероятность того, что случайный положительный объект получит оценку выше
    случайного отрицательного. Равные оценки считаются за половину.
    """
    # подсказка: сравните каждую пару (положительный, отрицательный) — хватит NumPy-бродкастинга
    y_true, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    pos, neg = scores[y_true == 1], scores[y_true == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def threshold_for_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    """Наибольший порог t, при котором предсказание `scores >= t` даёт recall не ниже `target_recall`.

    Кандидаты в пороги — сами значения `scores`.
    """
    y_true, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    for t in np.unique(scores)[::-1]:
        if precision_recall_f1(y_true, (scores >= t).astype(int))[1] >= target_recall:
            return float(t)
    return float(scores.min())


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """ECE: делим [0, 1] на `n_bins` равных корзин, в каждой — |средняя вероятность − доля положительных|,
    усредняем с весами по размеру корзин. Вероятность 1.0 попадает в последнюю корзину.
    """
    y_true, probs = np.asarray(y_true, dtype=float), np.asarray(probs, dtype=float)
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(probs[m].mean() - y_true[m].mean())
    return float(ece)
