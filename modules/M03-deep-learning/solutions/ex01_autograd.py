"""Упражнение 1. Свой autograd.

Теория: раздел «Обратное распространение и autograd». Урок 1.
Каждая операция создаёт новый `Value` и задаёт ему `backward_step` — функцию, которая раздаёт
`out.grad` родителям по цепному правилу. Градиенты **накапливаются** (`+=`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(eq=False)
class Value:
    """Число в графе вычислений: значение, градиент, родители и шаг обратного прохода."""

    data: float
    parents: tuple[Value, ...] = ()
    grad: float = 0.0
    backward_step: Callable[[], None] = field(default=lambda: None, repr=False)


def add(a: Value, b: Value) -> Value:  # готово — образец для остальных операций
    """a + b: производная по каждому слагаемому равна 1."""
    out = Value(a.data + b.data, (a, b))

    def step() -> None:
        a.grad += out.grad
        b.grad += out.grad

    out.backward_step = step
    return out


def mul(a: Value, b: Value) -> Value:
    """a · b: d/da = b, d/db = a."""
    out = Value(a.data * b.data, (a, b))

    def step() -> None:
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad

    out.backward_step = step
    return out


def tanh(a: Value) -> Value:
    """tanh(a): производная 1 − tanh²(a)."""
    t = math.tanh(a.data)
    out = Value(t, (a,))

    def step() -> None:
        a.grad += (1 - t**2) * out.grad

    out.backward_step = step
    return out


def exp(a: Value) -> Value:
    """e^a: производная равна самой e^a."""
    out = Value(math.exp(a.data), (a,))

    def step() -> None:
        a.grad += out.data * out.grad

    out.backward_step = step
    return out


def topological_order(root: Value) -> list[Value]:
    """Все вершины, от которых зависит `root` (включая его), так что родители стоят раньше детей.

    Каждая вершина — ровно один раз, даже если до неё ведут несколько путей.
    """
    # подсказка: обход в глубину; вершину добавляйте в список после всех её родителей
    order: list[Value] = []
    seen: set[int] = set()

    def visit(v: Value) -> None:
        if id(v) in seen:
            return
        seen.add(id(v))
        for p in v.parents:
            visit(p)
        order.append(v)

    visit(root)
    return order


def backward(root: Value) -> None:
    """Обратный проход: root.grad = 1, затем backward_step всех вершин от root к листьям."""
    root.grad = 1.0
    for v in reversed(topological_order(root)):
        v.backward_step()
