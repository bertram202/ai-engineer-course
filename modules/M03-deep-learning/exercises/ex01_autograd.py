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
    raise NotImplementedError("TODO")


def tanh(a: Value) -> Value:
    """tanh(a): производная 1 − tanh²(a)."""
    raise NotImplementedError("TODO")


def exp(a: Value) -> Value:
    """e^a: производная равна самой e^a."""
    raise NotImplementedError("TODO")


def topological_order(root: Value) -> list[Value]:
    """Все вершины, от которых зависит `root` (включая его), так что родители стоят раньше детей.

    Каждая вершина — ровно один раз, даже если до неё ведут несколько путей.
    """
    raise NotImplementedError("TODO: обход в глубину; вершину добавляйте в список после всех её родителей")


def backward(root: Value) -> None:
    """Обратный проход: root.grad = 1, затем backward_step всех вершин от root к листьям."""
    raise NotImplementedError("TODO")
