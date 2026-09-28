"""Рисунки к теории M02. Запуск из папки модуля: `uv run python theory/figures/make_figures.py`."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification, make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor

from mlcourse import repo_root

OUT = Path(__file__).parent
plt.style.use(repo_root() / "quarto" / "figures.mplstyle")
rng = np.random.default_rng(0)


def bias_variance() -> None:
    """Полиномы разной степени на зашумлённом синусе и ошибка на обучении и валидации."""
    x = np.sort(rng.uniform(0, 1, 20))
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, x.size)
    x_val = rng.uniform(0, 1, 200)
    y_val = np.sin(2 * np.pi * x_val) + rng.normal(0, 0.25, x_val.size)
    grid = np.linspace(0, 1, 300)

    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
    for ax, deg, title in zip(axes, [1, 4, 9], ["степень 1: недообучение", "степень 4: в самый раз",
                                                 "степень 9: переобучение"]):
        coef = np.polyfit(x, y, deg)
        ax.scatter(x, y, s=10, color="0.4")
        ax.plot(grid, np.sin(2 * np.pi * grid), color="0.7", ls="--", lw=1)
        ax.plot(grid, np.polyval(coef, grid))
        ax.set_ylim(-1.8, 1.8)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    degrees = range(0, 11)
    train, val = [], []
    for d in degrees:
        coef = np.polyfit(x, y, d)
        train.append(np.mean((np.polyval(coef, x) - y) ** 2))
        val.append(np.mean((np.polyval(coef, x_val) - y_val) ** 2))
    ax = axes[3]
    ax.semilogy(degrees, train, marker="o", ms=3, label="обучение")
    ax.semilogy(degrees, val, marker="o", ms=3, label="валидация")
    ax.set_xlabel("степень полинома")
    ax.set_title("ошибка (MSE)", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(OUT / "bias_variance.png")
    plt.close(fig)


def roc_pr() -> None:
    """ROC- и PR-кривые двух классификаторов на несбалансированных данных (5% положительных)."""
    x, y = make_classification(n_samples=6000, n_features=20, n_informative=6, weights=[0.95], shuffle=False,
                               random_state=0)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.5, stratify=y, random_state=0)
    models = {
        "логистическая регрессия": LogisticRegression(max_iter=1000).fit(xtr, ytr),
        "модель на 2 признаках": LogisticRegression().fit(xtr[:, :2], ytr),
    }
    scores = {
        "логистическая регрессия": models["логистическая регрессия"].predict_proba(xte)[:, 1],
        "модель на 2 признаках": models["модель на 2 признаках"].predict_proba(xte[:, :2])[:, 1],
    }
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(yte, s)
        prec, rec, _ = precision_recall_curve(yte, s)
        a1.plot(fpr, tpr, label=f"{name}: AUC {auc(fpr, tpr):.2f}")
        a2.plot(rec, prec, label=f"{name}: AUC {auc(rec, prec):.2f}")
    a1.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1)
    a2.axhline(yte.mean(), ls="--", color="0.6", lw=1, label=f"случайная: {yte.mean():.2f}")
    a1.set(xlabel="FPR (доля ложных тревог)", ylabel="TPR (recall)", title="ROC-кривая")
    a2.set(xlabel="recall", ylabel="precision", title="Precision–recall кривая")
    a1.legend(fontsize=8, loc="lower right")
    a2.legend(fontsize=8, loc="lower left")
    fig.savefig(OUT / "roc_pr.png")
    plt.close(fig)


def calibration() -> None:
    """Калибровочные кривые: логистическая регрессия против наивного Байеса и случайного леса."""
    x, y = make_classification(n_samples=40000, n_features=20, n_informative=5, n_redundant=0, random_state=42)
    xtr, xte, ytr, yte = train_test_split(x, y, train_size=5000, random_state=0)
    models = {
        "логистическая регрессия": LogisticRegression(max_iter=1000),
        "наивный Байес": GaussianNB(),
        "случайный лес": RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
    }
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1, label="идеальная калибровка")
    for name, m in models.items():
        prob = m.fit(xtr, ytr).predict_proba(xte)[:, 1]
        frac, mean_pred = calibration_curve(yte, prob, n_bins=10)
        ax.plot(mean_pred, frac, marker="o", ms=3, label=name)
    ax.set(xlabel="предсказанная вероятность", ylabel="реальная доля положительных", title="Калибровка")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "calibration.png")
    plt.close(fig)


def boosting() -> None:
    """Градиентный бустинг по шагам: каждое дерево глубины 2 подгоняет остатки предыдущих."""
    x = np.sort(rng.uniform(0, 1, 120))
    y = np.sin(3 * np.pi * x) * (1 - x) + rng.normal(0, 0.1, x.size)
    grid = np.linspace(0, 1, 400)
    lr, stages = 0.3, [1, 3, 10, 60]
    pred, pred_grid, snapshots = np.full_like(y, y.mean()), np.full_like(grid, y.mean()), {}
    for m in range(1, max(stages) + 1):
        tree = DecisionTreeRegressor(max_depth=2).fit(x[:, None], y - pred)
        pred += lr * tree.predict(x[:, None])
        pred_grid += lr * tree.predict(grid[:, None])
        if m in stages:
            snapshots[m] = pred_grid.copy()
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.6), sharey=True)
    for ax, m in zip(axes, stages):
        ax.scatter(x, y, s=8, color="0.5")
        ax.plot(grid, snapshots[m], color="#E53935")
        ax.set_title(f"{m} {'дерево' if m == 1 else 'деревьев'}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(OUT / "boosting.png")
    plt.close(fig)


def kmeans() -> None:
    """k-means: хорошо на «шарах», плохо на невыпуклых кластерах."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, (x, title) in zip(axes, [
        (make_blobs(n_samples=400, centers=3, cluster_std=1.0, random_state=2)[0], "три «шара»: k-means справляется"),
        (make_moons(n_samples=400, noise=0.06, random_state=0)[0], "два «полумесяца»: k-means режет поперёк"),
    ]):
        k = 3 if "три" in title else 2
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(x)
        ax.scatter(x[:, 0], x[:, 1], c=km.labels_, s=8, cmap="viridis")
        ax.scatter(*km.cluster_centers_.T, marker="X", s=120, color="#E53935", edgecolor="w")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(OUT / "kmeans.png")
    plt.close(fig)


if __name__ == "__main__":
    for make in (bias_variance, roc_pr, calibration, boosting, kmeans):
        make()
        print("✓", make.__name__)
