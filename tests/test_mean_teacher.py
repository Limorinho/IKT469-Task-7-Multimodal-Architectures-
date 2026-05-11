import torch
import torch.nn as nn
from src.ssl.mean_teacher import MeanTeacher, consistency_loss


def _make_student():
    return nn.Sequential(nn.Linear(4, 3))


def test_teacher_initially_matches_student():
    s = _make_student()
    mt = MeanTeacher(s)
    for sp, tp in zip(s.parameters(), mt.teacher.parameters()):
        assert torch.allclose(sp, tp)


def test_teacher_ema_update():
    s = _make_student()
    mt = MeanTeacher(s, alpha=0.9)

    for p in s.parameters():
        p.data.add_(1.0)

    before = [p.detach().clone() for p in mt.teacher.parameters()]
    mt.update(s)
    after = list(mt.teacher.parameters())

    for b, a, sp in zip(before, after, s.parameters()):
        expected = 0.9 * b + 0.1 * sp.detach()
        assert torch.allclose(a, expected, atol=1e-6)


def test_consistency_loss_zero_when_equal():
    x = torch.randn(2, 3)
    assert consistency_loss(x, x).item() == 0.0


def test_consistency_loss_positive():
    a = torch.zeros(2, 3)
    b = torch.ones(2, 3) * 5.0
    assert consistency_loss(a, b).item() > 0.0
