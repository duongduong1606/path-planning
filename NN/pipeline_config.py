"""Shared contracts for the comma2k19 lateral-planning pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


FEATURE_NAMES = (
    "left_c0", "left_c1", "left_c2", "left_c3",
    "right_c0", "right_c1", "right_c2", "right_c3",
    "left_confidence", "right_confidence", "speed_mps", "yaw_rate_rps",
    "lateral_accel_mps2", "steering_angle_rad", "steering_rate_rps",
    "perception_validity",
)
TARGET_NAMES = ("y_T_m", "y_dot_T_mps", "y_ddot_T_mps2")


@dataclass(frozen=True)
class PipelineConfig:
    sample_rate_hz: float = 20.0
    history_steps: int = 40
    horizon_steps: int = 50
    derivative_fit_steps: int = 9
    train_stride: int = 2
    eval_stride: int = 5
    seed: int = 2026
    train_routes: int = 26
    validation_routes: int = 5
    test_routes: int = 6
    manual_threshold: float = 0.80
    lane_width_m: float = 3.7
    lane_min_x_m: float = 5.0
    lane_max_x_m: float = 60.0
    lane_hold_seconds: float = 0.5
    fx: float = 910.0
    fy: float = 910.0
    cx: float = 582.0
    cy: float = 437.0

    @property
    def horizon_seconds(self) -> float:
        return self.horizon_steps / self.sample_rate_hz

    @property
    def lane_hold_steps(self) -> int:
        return round(self.lane_hold_seconds * self.sample_rate_hz)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

