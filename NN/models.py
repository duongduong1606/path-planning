"""Neural baselines and losses sharing the TCN input/output contract."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FlattenedMLP(nn.Module):
    def __init__(self, history_steps: int = 40, features: int = 16) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(), nn.Linear(history_steps * features, 128), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 4),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class GRUBaseline(nn.Module):
    def __init__(self, features: int = 16, hidden_size: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(features, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x: Tensor) -> Tensor:
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


def quintic_path_torch(
    terminal: Tensor, horizon_s: float = 2.5, samples: int = 51
) -> Tensor:
    """Reconstruct the lateral quintic path for a batch of terminal states."""
    dtype, device = terminal.dtype, terminal.device
    horizon = torch.as_tensor(horizon_s, dtype=dtype, device=device)
    matrix = torch.stack((
        torch.stack((horizon**3, horizon**4, horizon**5)),
        torch.stack((3*horizon**2, 4*horizon**3, 5*horizon**4)),
        torch.stack((6*horizon, 12*horizon**2, 20*horizon**3)),
    ))
    coefficients = torch.linalg.solve(matrix, terminal[..., :3].transpose(0, 1))
    time = torch.linspace(0.0, horizon_s, samples, dtype=dtype, device=device)
    basis = torch.stack((time**3, time**4, time**5), dim=1)
    return (basis @ coefficients).transpose(0, 1)


def lane_feasibility_loss(
    prediction: Tensor,
    current_features: Tensor,
    horizon_s: float = 2.5,
    samples: int = 51,
    margin_m: float = 0.20,
) -> Tensor:
    """Confidence-weighted soft penalty outside the current lane corridor."""
    if current_features.shape[-1] != 16:
        raise ValueError("expected 16 physical input features")
    path = quintic_path_torch(prediction, horizon_s, samples)
    time = torch.linspace(
        0.0, horizon_s, samples,
        dtype=prediction.dtype, device=prediction.device,
    )
    speed = current_features[:, 10].clamp_min(0.0)
    # Lane cubics share the preprocessing contract: they are calibrated only
    # over 5..60 m. Keep training identical to teacher generation/evaluation.
    x = (speed[:, None] * time[None, :]).clamp(5.0, 60.0)
    powers = torch.stack((torch.ones_like(x), x, x.square(), x.pow(3)), dim=-1)
    left = (powers * current_features[:, None, :4]).sum(dim=-1)
    right = (powers * current_features[:, None, 4:8]).sum(dim=-1)
    corridor_valid = torch.isfinite(left) & torch.isfinite(right) & (
        left - right > 2.0 * margin_m
    )
    upper_violation = torch.relu(path - (left - margin_m))
    lower_violation = torch.relu((right + margin_m) - path)
    point_penalty = (upper_violation.square() + lower_violation.square()) * corridor_valid

    confidence = current_features[:, 8:10].clamp(0.0, 1.0).amin(dim=1)
    perception_validity = current_features[:, 15].clamp(0.0, 1.0)
    sample_weight = (0.25 + 0.75 * confidence) * perception_validity
    valid_points = corridor_valid.sum(dim=1).clamp_min(1)
    sample_penalty = point_penalty.sum(dim=1) / valid_points
    return (sample_penalty * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)


def terminal_heteroscedastic_loss(
    prediction: Tensor,
    target: Tensor,
    current_features: Tensor | None = None,
    lane_weight: float = 0.0,
    lane_margin_m: float = 0.20,
) -> Tensor:
    """MSE for derivatives plus Gaussian NLL for terminal lateral position."""
    if prediction.shape[-1] != 4 or target.shape[-1] != 3:
        raise ValueError("expected prediction [...,4] and target [...,3]")
    log_sigma = prediction[..., 3].clamp(-6.0, 3.0)
    residual = prediction[..., 0] - target[..., 0]
    position_nll = 0.5 * (residual.square() * torch.exp(-2*log_sigma) + 2*log_sigma)
    derivative_mse = (prediction[..., 1:3] - target[..., 1:3]).square()
    loss = position_nll.mean() + derivative_mse.mean()
    if lane_weight and current_features is not None:
        loss = loss + lane_weight * lane_feasibility_loss(
            prediction, current_features, margin_m=lane_margin_m
        )
    return loss


def teacher_distillation_loss(
    prediction: Tensor,
    teacher_target: Tensor,
    teacher_std: Tensor,
    valid: Tensor,
    uncertainty_scale_m: float = 0.5,
    component_weights: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> Tensor:
    """Confidence-weighted secondary loss; human pose remains primary truth."""
    if prediction.shape[-1] != 4 or teacher_target.shape[-1] != 3:
        raise ValueError("expected prediction [...,4] and teacher target [...,3]")
    valid = valid.bool() & torch.isfinite(teacher_target).all(dim=-1)
    valid &= torch.isfinite(teacher_std).all(dim=-1) & (teacher_std[..., 0] > 0)
    if not torch.any(valid):
        return prediction.sum() * 0.0
    residual = torch.nn.functional.smooth_l1_loss(
        prediction[valid, :3], teacher_target[valid], reduction="none", beta=0.25
    )
    # Position uncertainty is the common confidence contract across supported
    # offline teachers. The bound prevents overconfident frames dominating.
    weight = torch.exp(
        -teacher_std[valid, 0].clamp(0.0, 3.0) / uncertainty_scale_m
    ).clamp(0.05, 1.0)
    component_scale = prediction.new_tensor(component_weights)
    if torch.any(component_scale < 0) or component_scale.sum() <= 0:
        raise ValueError("teacher component weights must be non-negative with positive sum")
    per_sample = (residual * component_scale).sum(dim=-1) / component_scale.sum()
    return (per_sample * weight).sum() / weight.sum().clamp_min(1e-6)

