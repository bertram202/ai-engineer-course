"""Рисунки к теории M01. Запуск из папки модуля: `uv run python theory/figures/make_figures.py`."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

from mlcourse import repo_root

OUT = Path(__file__).parent
plt.style.use(repo_root() / "quarto" / "figures.mplstyle")


def transform_grid() -> None:
    """Матрица как преобразование плоскости: сетка и базисные векторы до и после."""
    matrices = {
        "Поворот на 30°": np.array([[np.cos(np.pi / 6), -np.sin(np.pi / 6)], [np.sin(np.pi / 6), np.cos(np.pi / 6)]]),
        "Растяжение": np.array([[1.8, 0.0], [0.0, 0.6]]),
        "Сдвиг": np.array([[1.0, 0.8], [0.0, 1.0]]),
        "Проекция на ось x": np.array([[1.0, 0.0], [0.0, 0.0]]),
    }
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    ticks = np.linspace(-1, 1, 5)
    for ax, (title, a) in zip(axes, matrices.items()):
        for t in ticks:
            for line in (np.array([[t, t], [-1, 1]]), np.array([[-1, 1], [t, t]])):
                ax.plot(*line, color="0.85", lw=0.8)
                ax.plot(*(a @ line), color="#3949AB", lw=0.9, alpha=0.6)
        for vec, color in ((np.array([1, 0]), "#E53935"), (np.array([0, 1]), "#43A047")):
            v = a @ vec
            ax.annotate("", xy=v, xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": color, "lw": 2})
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1.6, 1.6)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(OUT / "transform.png")
    plt.close(fig)


def svd_digits() -> None:
    """Сингулярные числа матрицы цифр и восстановление цифры низкоранговым приближением."""
    x = load_digits().data  # 1797 × 64
    x = x - x.mean(axis=0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    energy = np.cumsum(s**2) / np.sum(s**2)

    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_axes((0.05, 0.17, 0.26, 0.68))
    ax.plot(np.arange(1, len(s) + 1), s, marker="o", ms=2)
    ax.set_title("Сингулярные числа")
    ax.set_xlabel("номер $i$")
    ax.set_ylabel(r"$\sigma_i$")
    ax = fig.add_axes((0.37, 0.17, 0.26, 0.68))
    ax.plot(np.arange(1, len(s) + 1), energy)
    ax.axhline(0.9, color="#E53935", ls="--", lw=1)
    k90 = int(np.searchsorted(energy, 0.9)) + 1
    ax.annotate(f"90% «энергии»: k = {k90}", (k90, 0.9), (k90 + 8, 0.6), arrowprops={"arrowstyle": "->"})
    ax.set_title("Доля энергии $\\sum_{i\\leq k}\\sigma_i^2 / \\sum \\sigma_i^2$")
    ax.set_xlabel("ранг $k$")

    mean = load_digits().data.mean(axis=0)
    idx = 7  # эта строка матрицы — цифра 7
    ranks = [1, 3, 5, 10, 20, 64]
    grid = fig.add_gridspec(2, 3, left=0.69, right=0.99, top=0.85, bottom=0.05, hspace=0.35)
    for j, k in enumerate(ranks):
        approx = (u[idx, :k] * s[:k]) @ vt[:k] + mean
        sub = fig.add_subplot(grid[j // 3, j % 3])
        sub.imshow(approx.reshape(8, 8), cmap="gray_r", vmin=0, vmax=16)
        sub.set_title("оригинал" if k == 64 else f"ранг {k}", fontsize=9)
        sub.axis("off")
    fig.savefig(OUT / "svd_digits.png")
    plt.close(fig)


def gd_paths() -> None:
    """Градиентный спуск на вытянутой квадратичной чаше: маленький, хороший и слишком большой шаг."""
    lam = np.array([1.0, 10.0])  # собственные числа: число обусловленности 10

    def run(lr: float, beta: float = 0.0, steps: int = 30) -> np.ndarray:
        x, v, path = np.array([-4.0, 1.5]), np.zeros(2), []
        for _ in range(steps):
            path.append(x.copy())
            v = beta * v + lam * x
            x = x - lr * v
        return np.array(path)

    xs, ys = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-2.2, 2.2, 200))
    zs = 0.5 * (lam[0] * xs**2 + lam[1] * ys**2)
    cases = [
        ("η = 0.02: медленно", run(0.02)),
        ("η = 0.18: зигзаг", run(0.18)),
        (r"$\eta = 0.201 > 2/\lambda_{\max}$: расходится", run(0.201, steps=18)),
        ("η = 0.05, momentum β = 0.8", run(0.05, beta=0.8)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6))
    for ax, (title, path) in zip(axes, cases):
        ax.contour(xs, ys, zs, levels=12, colors="0.75", linewidths=0.7)
        ax.plot(path[:, 0], path[:, 1], marker="o", ms=2.5, lw=1.2)
        ax.plot(0, 0, marker="*", color="#E53935", ms=10)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-2.2, 2.2)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(OUT / "gd_paths.png")
    plt.close(fig)


def softmax_temperature() -> None:
    """Softmax одних и тех же логитов при разных температурах."""
    tokens = ["кот", "пёс", "лис", "ёж", "кит", "як"]
    logits = np.array([3.0, 2.2, 1.5, 0.8, 0.2, -0.5])
    temps = [0.3, 1.0, 3.0]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.5), sharey=True)
    for ax, t in zip(axes, temps):
        z = logits / t
        p = np.exp(z - z.max())
        p /= p.sum()
        h = -(p * np.log(p)).sum()
        ax.bar(tokens, p, color="#3949AB")
        ax.set_title(f"T = {t}: энтропия {h:.2f} нат")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("вероятность")
    fig.savefig(OUT / "temperature.png")
    plt.close(fig)


def kl_directions() -> None:
    """Прямая и обратная KL: какую одну гауссиану мы подберём к бимодальному распределению."""
    x = np.linspace(-8, 8, 2001)
    dx = x[1] - x[0]

    def normal(mu: float, sigma: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    p = 0.5 * normal(-3, 1) + 0.5 * normal(3, 1)
    eps = 1e-300

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * (np.log(a + eps) - np.log(b + eps))) * dx)

    grid = [(mu, s) for mu in np.linspace(-4, 4, 81) for s in np.linspace(0.5, 5, 46)]
    fwd = min(grid, key=lambda ms: kl(p, normal(*ms)))
    rev = min(grid, key=lambda ms: kl(normal(*ms), p))

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.6), sharey=True)
    for ax, (mu, s), title in (
        (axes[0], fwd, "Прямая KL(p‖q): «накрыть всё»"),
        (axes[1], rev, "Обратная KL(q‖p): «выбрать моду»"),
    ):
        ax.fill_between(x, p, color="0.8", label="данные p")
        ax.plot(x, normal(mu, s), color="#E53935", label=f"q: μ={mu:.1f}, σ={s:.1f}")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_yticks([])
    fig.savefig(OUT / "kl_directions.png")
    plt.close(fig)


if __name__ == "__main__":
    for make in (transform_grid, svd_digits, gd_paths, softmax_temperature, kl_directions):
        make()
        print("✓", make.__name__)
