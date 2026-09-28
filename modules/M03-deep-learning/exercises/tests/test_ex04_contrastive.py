import math

import pytest
import torch
import torch.nn.functional as F
from ex04_contrastive import info_nce, recall_at_k, zero_shot_predict


def test_info_nce_matches_manual_formula():
    torch.manual_seed(0)
    a, b = torch.randn(6, 8), torch.randn(6, 8)
    an, bn = F.normalize(a, dim=1), F.normalize(b, dim=1)
    s = an @ bn.T / 0.1
    manual = -(torch.diag(s) - torch.logsumexp(s, dim=1)).mean()
    assert info_nce(a, b, 0.1).item() == pytest.approx(manual.item(), rel=1e-5)


def test_info_nce_symmetric():
    torch.manual_seed(1)
    a, b = torch.randn(5, 4), torch.randn(5, 4)
    sym = info_nce(a, b, 0.2, symmetric=True)
    assert sym.item() == pytest.approx(((info_nce(a, b, 0.2) + info_nce(b, a, 0.2)) / 2).item(), rel=1e-5)


def test_info_nce_extremes():
    eye = torch.eye(16)
    assert info_nce(eye, eye, 0.01).item() < 1e-6            # идеальные пары
    ones = torch.ones(16, 4)
    assert info_nce(ones, ones, 0.05).item() == pytest.approx(math.log(16), rel=1e-5)  # все одинаковы


def test_info_nce_ignores_scale_and_has_gradient():
    torch.manual_seed(2)
    a = torch.randn(4, 3, requires_grad=True)
    b = torch.randn(4, 3)
    assert info_nce(a, b).item() == pytest.approx(info_nce(10 * a, b).item(), rel=1e-5)
    info_nce(a, b).backward()
    assert a.grad is not None and a.grad.abs().sum() > 0


def test_zero_shot_uses_cosine():
    texts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    images = torch.tensor([[10.0, 1.0], [0.1, 0.3], [5.0, 5.1]])
    assert zero_shot_predict(images, texts).tolist() == [0, 1, 1]
    # по скалярному произведению с ненормированным текстом ответ был бы другим
    assert zero_shot_predict(images, torch.tensor([[1.0, 0.0], [0.0, 100.0]])).tolist() == [0, 1, 1]


def test_recall_at_k():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    d = torch.tensor([[1.0, 0.1], [1.0, 0.0], [0.7, 0.7]])
    # запрос 0 ближе всего к документу 1, запрос 1 — к документу 2, запрос 2 — к своему
    assert recall_at_k(q, d, 1) == pytest.approx(1 / 3)
    assert recall_at_k(q, d, 2) == pytest.approx(2 / 3)
    assert recall_at_k(q, d, 3) == pytest.approx(1.0)
