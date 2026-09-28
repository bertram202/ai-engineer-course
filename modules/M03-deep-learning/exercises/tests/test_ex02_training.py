import torch
from ex02_training import count_parameters, evaluate, make_mlp, train_epoch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def blobs(n=600, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.5]])
    y = torch.randint(0, 3, (n,), generator=g)
    x = centers[y] + 0.5 * torch.randn(n, 2, generator=g)
    return TensorDataset(x, y)


def test_make_mlp_structure():
    m = make_mlp([4, 8, 8, 3])
    kinds = [type(layer) for layer in m]
    assert kinds == [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
    assert m[0].in_features == 4 and m[-1].out_features == 3


def test_make_mlp_dropout():
    m = make_mlp([4, 8, 3], dropout=0.2)
    assert [type(layer) for layer in m] == [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear]
    assert m[2].p == 0.2


def test_count_parameters():
    assert count_parameters(make_mlp([4, 8, 3])) == (4 * 8 + 8) + (8 * 3 + 3)
    m = make_mlp([4, 8, 3])
    m[0].weight.requires_grad_(False)
    assert count_parameters(m) == 8 + 8 * 3 + 3


def test_training_reduces_loss_and_learns():
    torch.manual_seed(0)
    train = DataLoader(blobs(), batch_size=32, shuffle=True)
    test = DataLoader(blobs(seed=1), batch_size=100)
    model = make_mlp([2, 16, 3])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = [train_epoch(model, train, opt) for _ in range(5)]
    assert isinstance(losses[0], float)
    assert losses[-1] < losses[0] * 0.5
    assert evaluate(model, test) > 0.9


def test_evaluate_uses_eval_mode_and_no_grad():
    torch.manual_seed(0)
    model = make_mlp([2, 64, 3], dropout=0.9)
    loader = DataLoader(blobs(200), batch_size=50)
    before = [p.clone() for p in model.parameters()]
    first, second = evaluate(model, loader), evaluate(model, loader)
    assert first == second  # dropout выключен — результат детерминирован
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert all(p.grad is None for p in model.parameters())
    assert 0.0 <= first <= 1.0
