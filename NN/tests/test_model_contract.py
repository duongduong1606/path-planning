import torch

from models import (
    lane_feasibility_loss, teacher_distillation_loss,
    terminal_heteroscedastic_loss,
)
from tcn_model import LateralPlanningTCN


def test_tcn_shape_receptive_field_and_causality():
    torch.manual_seed(1)
    model = LateralPlanningTCN(dropout=0).eval()
    assert model.receptive_field == 61
    x1 = torch.randn(2, 40, 16)
    x2 = x1.clone(); x2[:, 30:] = torch.randn_like(x2[:, 30:])
    f1 = model.tcn(x1.transpose(1, 2))
    f2 = model.tcn(x2.transpose(1, 2))
    assert torch.allclose(f1[:, :, :30], f2[:, :, :30], atol=1e-6)
    assert model(x1).shape == (2, 4)


def test_loss_is_finite():
    loss = terminal_heteroscedastic_loss(torch.zeros(3, 4), torch.ones(3, 3))
    assert torch.isfinite(loss)


def test_lane_feasibility_loss_penalizes_outside_path():
    features = torch.zeros(1, 16)
    features[:, 0] = 1.8
    features[:, 4] = -1.8
    features[:, 8:10] = 1.0
    features[:, 10] = 10.0
    features[:, 15] = 1.0
    safe = lane_feasibility_loss(torch.zeros(1, 4), features)
    unsafe_prediction = torch.tensor([[3.0, 0.0, 0.0, 0.0]])
    unsafe = lane_feasibility_loss(unsafe_prediction, features)
    assert safe == 0
    assert unsafe > 0


def test_teacher_loss_masks_invalid_and_rewards_match():
    prediction = torch.zeros(2, 4, requires_grad=True)
    teacher = torch.tensor([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])
    std = torch.ones(2, 3) * 0.1
    loss = teacher_distillation_loss(
        prediction, teacher, std, torch.tensor([True, False])
    )
    assert loss == 0
    loss.backward()
    assert prediction.grad is not None

