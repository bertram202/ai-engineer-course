"""Упражнение 3. Градиентный бустинг для регрессии (MSE) с нуля.

Теория: раздел «Градиентный бустинг». Урок 3.
Деревья берём готовые — `DecisionTreeRegressor` из scikit-learn; сам бустинг пишем сами.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.tree import DecisionTreeRegressor


@dataclass
class Boosting:
    """Обученная модель: начальное предсказание, шаг и список деревьев."""

    init: float
    learning_rate: float
    trees: list[DecisionTreeRegressor] = field(default_factory=list)


def fit_boosting(
    X: np.ndarray, y: np.ndarray, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3
) -> Boosting:
    """Бустинг на MSE: старт со среднего `y`, каждое дерево (random_state=0) учится на остатках.

    После каждого дерева предсказание обновляется: F ← F + learning_rate · tree(X).
    """
    # подсказка: для MSE антиградиент по предсказаниям — это остатки y − F
    model = Boosting(init=float(np.mean(y)), learning_rate=learning_rate)
    pred = np.full(len(y), model.init)
    for _ in range(n_estimators):
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=0).fit(X, y - pred)
        pred += learning_rate * tree.predict(X)
        model.trees.append(tree)
    return model


def predict_boosting(model: Boosting, X: np.ndarray, n_trees: int | None = None) -> np.ndarray:
    """Предсказание по первым `n_trees` деревьям (все, если None)."""
    trees = model.trees if n_trees is None else model.trees[:n_trees]
    pred = np.full(len(X), model.init)
    for tree in trees:
        pred += model.learning_rate * tree.predict(X)
    return pred
