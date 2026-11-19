"""Tests for pure-NumPy TCN runtime and 40 Hz -> 20 Hz HFSM compatibility adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tcn_model import LateralPlanningTCN
from tcn_runtime import (
    HistoryResampler20Hz,
    NumpyLateralTCN,
    TCN40HzAdapter,
    evaluate_ego_quintic,
    export_runtime_bundle,
    hfsm_inputs_to_nn_features,
)


@dataclass
class _StubLaneLine:
    c0: float
    c1: float
    c2_inv_km: float = 0.0
    c3: float = 0.0
    existence_probability: float = 1.0


def test_numpy_pytorch_parity() -> None:
    ckpt = Path("artifacts/tcn_lane_safe/tcn.pt")
    bundle = Path("artifacts/tcn_lane_safe/lateral_tcn_runtime.npz")

    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = state["model"] if "model" in state else state
    pt_model = LateralPlanningTCN()
    pt_model.load_state_dict(state)
    pt_model.eval()

    np_model = NumpyLateralTCN(bundle)
    rng = np.random.default_rng(123)
    x_norm = rng.normal(0.0, 0.8, size=(40, 16)).astype(np.float32)

    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(x_norm)[None]).numpy()[0]
    np_out = np_model.forward_normalized(x_norm)
    assert np.max(np.abs(pt_out - np_out)) < 1e-5


def test_resampler_40hz_to_20hz_exact_decimation_and_warmup() -> None:
    resampler = HistoryResampler20Hz(target_hz=20.0, history_steps=40)
    # Warm-up: 5 frames at 40 Hz (0.1 s)
    for i in range(5):
        resampler.push(i * 0.025, np.full(16, float(i), dtype=np.float32))
    warm = resampler.window()
    assert warm.shape == (40, 16)
    assert not np.isnan(warm).any()
    assert np.isclose(warm[-1, 0], 4.0)
    assert np.isclose(warm[0, 0], 0.0)

    # Full 2.0 s at 40 Hz (80 frames: 0..79)
    resampler.reset()
    for i in range(80):
        resampler.push(i * 0.025, np.full(16, float(i), dtype=np.float32))
    full = resampler.window()
    expected = np.arange(1, 80, 2, dtype=np.float32)  # 1, 3, 5, ..., 79
    assert np.allclose(full[:, 0], expected, atol=1e-5)


def test_hfsm_sign_unit_and_missing_lane_conversion() -> None:
    left = _StubLaneLine(c0=-1.85, c1=0.02, c2_inv_km=0.5, existence_probability=0.95)
    right = _StubLaneLine(c0=1.85, c1=0.02, c2_inv_km=0.5, existence_probability=0.90)
    feat = hfsm_inputs_to_nn_features(
        stamp_s=0.025,
        left_line=left,
        right_line=right,
        speed_mps=20.0,
        yaw_rate_rps=0.01,
        lateral_accel_mps2=0.2,
        front_wheel_angle_rad=0.02,
        steering_ratio=15.0,
    )
    assert np.isclose(feat[0], 1.85)   # left_c0 left-positive
    assert np.isclose(feat[4], -1.85)  # right_c0 left-positive
    assert np.isclose(feat[2], 0.0005) # c2_inv_m
    assert np.isclose(feat[13], 0.30)  # steering wheel angle = 0.02 * 15
    assert np.isclose(feat[15], 1.0)   # perception_validity


def test_40hz_adapter_and_quintic_end_to_end() -> None:
    bundle = Path("artifacts/tcn_lane_safe/lateral_tcn_runtime.npz")
    adapter = TCN40HzAdapter(bundle, steering_ratio=15.0, infer_every_n_steps=2)
    left = _StubLaneLine(c0=-1.85, c1=0.0, c2_inv_km=0.0, existence_probability=1.0)
    right = _StubLaneLine(c0=1.85, c1=0.0, c2_inv_km=0.0, existence_probability=1.0)

    pred = None
    for k in range(100):
        pred = adapter.step(
            stamp_s=k * 0.025,
            left_line=left,
            right_line=right,
            speed_mps=20.0,
            yaw_rate_rps=0.0,
            lateral_accel_mps2=0.0,
            front_wheel_angle_rad=0.0,
        )
    assert pred is not None and pred.shape == (4,)
    assert np.isfinite(pred).all()

    t_eval = np.linspace(0.0, 2.5, 26)
    y, yd, ydd = evaluate_ego_quintic(t_eval, float(pred[0]), float(pred[1]), float(pred[2]))
    assert np.isclose(y[-1], pred[0], atol=1e-5)
    assert np.isclose(yd[-1], pred[1], atol=1e-5)
    assert np.isclose(ydd[-1], pred[2], atol=1e-5)
