import math

import pytest
import torch
from ex01_autograd import Value, add, backward, exp, mul, tanh, topological_order


def test_forward_values():
    a, b = Value(2.0), Value(-3.0)
    assert mul(a, b).data == -6.0
    assert tanh(a).data == pytest.approx(math.tanh(2.0))
    assert exp(b).data == pytest.approx(math.exp(-3.0))


def test_mul_gradients():
    a, b = Value(2.0), Value(-3.0)
    out = mul(a, b)
    backward(out)
    assert (a.grad, b.grad) == (-3.0, 2.0)


def test_gradients_accumulate_for_reused_value():
    x = Value(3.0)
    y = mul(x, x)  # x используется дважды
    backward(y)
    assert x.grad == pytest.approx(6.0)


def test_topological_order_parents_first_and_unique():
    a, b = Value(1.0), Value(2.0)
    c = mul(a, b)
    d = add(c, a)  # «ромб»: до a два пути
    e = tanh(d)
    order = topological_order(e)
    assert len(order) == len({id(v) for v in order}) == 5
    pos = {id(v): i for i, v in enumerate(order)}
    for v in order:
        for p in v.parents:
            assert pos[id(p)] < pos[id(v)]
    assert order[-1] is e


def test_matches_pytorch():
    def f(a, b, c, mul_, add_, tanh_, exp_):
        return tanh_(add_(mul_(a, b), exp_(mul_(c, a))))

    vals = (0.5, -1.2, 0.8)
    ours = [Value(v) for v in vals]
    backward(f(*ours, mul, add, tanh, exp))
    theirs = [torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in vals]
    f(*theirs, torch.mul, torch.add, torch.tanh, torch.exp).backward()
    for o, t in zip(ours, theirs):
        assert o.grad == pytest.approx(t.grad.item(), rel=1e-9)
