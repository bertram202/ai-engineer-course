"""Упражнение 1. Метрики классификации, ROC-AUC, порог и калибровка — с нуля.

Теория: раздел «Метрики». Урок 1. Все функции принимают numpy-массивы; метки — 0 и 1.
"""

import numpy as np


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Матрица ошибок бинарной классификации: словарь с ключами tp, fp, fn, tn (int)."""
    raise NotImplementedError("TODO")


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall и F1 положительного класса. Если знаменатель равен нулю, метрика равна 0."""
    raise NotImplementedError("TODO")


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC как вероятность того, что случайный положительный объект получит оценку выше
    случайного отрицательного. Равные оценки считаются за половину.
    """
    raise NotImplementedError("TODO: сравните каждую пару (положительный, отрицательный) — хватит NumPy-бродкастинга")


def threshold_for_recall(y_true: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    """Наибольший порог t, при котором предсказание `scores >= t` даёт recall не ниже `target_recall`.

    Кандидаты в пороги — сами значения `scores`.
    """
    raise NotImplementedError("TODO")


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """ECE: делим [0, 1] на `n_bins` равных корзин, в каждой — |средняя вероятность − доля положительных|,
    усредняем с весами по размеру корзин. Вероятность 1.0 попадает в последнюю корзину.
    """
    raise NotImplementedError("TODO")
