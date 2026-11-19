"""Pure-NumPy runtime and 40 Hz -> 20 Hz compatibility adapter for LateralPlanningTCN.

Bridges the 20 Hz left-positive comma2k19 TCN contract with 40 Hz (or arbitrary-rate)
planners such as ``HFSM_ Frenet_quintic_ LTV-MPC``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pipeline_config import FEATURE_NAMES, PipelineConfig


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _layer_norm_channels(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """ChannelLayerNorm on [channels, time] independently at every timestep."""
    mean = x.mean(axis=0, keepdims=True)
    var = x.var(axis=0, keepdims=True)
    normed = (x - mean) / np.sqrt(var + eps)
    return normed * weight[:, None] + bias[:, None]


def _causal_conv1d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    dilation: int,
) -> np.ndarray:
    """1D causal convolution on [in_channels, time] -> [out_channels, time]."""
    out_ch, in_ch, k = weight.shape
    pad = (k - 1) * dilation
    padded = np.pad(x, ((0, 0), (pad, 0)), mode="constant")
    t_out = x.shape[1]
    out = np.broadcast_to(bias[:, None], (out_ch, t_out)).copy()
    for ki in range(k):
        start = ki * dilation
        sl = padded[:, start : start + t_out]
        out += weight[:, :, ki] @ sl
    return out


class NumpyLateralTCN:
    """Dependency-free NumPy inference for LateralPlanningTCN."""

    def __init__(self, bundle_path: Path | str) -> None:
        self.bundle_path = Path(bundle_path)
        with np.load(self.bundle_path, allow_pickle=False) as data:
            self.weights = {k: data[k].astype(np.float32) for k in data.files if k.startswith("w__")}
            self.clip_low = data["clip_low"].astype(np.float32)
            self.clip_high = data["clip_high"].astype(np.float32)
            self.mean = data["mean"].astype(np.float32)
            self.std = np.maximum(data["std"].astype(np.float32), 1e-6)
            self.activation = str(data["activation"].item()) if "activation" in data else "relu"
            self.sample_rate_hz = float(data["sample_rate_hz"].item())
            self.history_steps = int(data["history_steps"].item())
            self.horizon_seconds = float(data["horizon_seconds"].item())

        self._act = _silu if self.activation in {"silu", "swish"} else _relu
        self._block_indices = sorted(
            {int(k.split("__")[1].split(".")[1]) for k in self.weights if k.startswith("w__tcn.")}
        )

    def _w(self, name: str) -> np.ndarray:
        return self.weights[f"w__{name}"]

    def normalize(self, window: np.ndarray) -> np.ndarray:
        """Clip and standardize [40, 16] raw features using training-only stats."""
        x = np.asarray(window, dtype=np.float32)
        if x.shape != (self.history_steps, len(FEATURE_NAMES)):
            raise ValueError(f"Expected shape ({self.history_steps}, {len(FEATURE_NAMES)}), got {x.shape}")
        clipped = np.clip(x, self.clip_low, self.clip_high)
        return (clipped - self.mean) / self.std

    def forward_normalized(self, normalized_window: np.ndarray) -> np.ndarray:
        """Run causal TCN on normalized [40, 16] window -> [y_T, y_dot_T, y_ddot_T, log_sigma_y]."""
        x = np.asarray(normalized_window, dtype=np.float32).T  # [16, 40]
        for idx in self._block_indices:
            p = f"tcn.{idx}"
            dilation = 2**idx
            if f"w__{p}.residual.weight" in self.weights:
                res = self._w(f"{p}.residual.weight")[:, :, 0] @ x + self._w(f"{p}.residual.bias")[:, None]
            else:
                res = x
            y = _causal_conv1d(x, self._w(f"{p}.conv1.weight"), self._w(f"{p}.conv1.bias"), dilation)
            y = _layer_norm_channels(y, self._w(f"{p}.norm1.norm.weight"), self._w(f"{p}.norm1.norm.bias"))
            y = self._act(y)
            y = _causal_conv1d(y, self._w(f"{p}.conv2.weight"), self._w(f"{p}.conv2.bias"), dilation)
            y = _layer_norm_channels(y, self._w(f"{p}.norm2.norm.weight"), self._w(f"{p}.norm2.norm.bias"))
            y = self._act(y)
            x = self._act(y + res)

        h = x[:, -1]  # final timestep [C]
        for layer_idx in (0, 3, 6):
            h = self._act(self._w(f"head.{layer_idx}.weight") @ h + self._w(f"head.{layer_idx}.bias"))
        out = self._w("head.9.weight") @ h + self._w("head.9.bias")
        return out.astype(np.float32)

    def predict(self, raw_window: np.ndarray) -> np.ndarray:
        """Normalize [40, 16] raw features and predict [y_T, y_dot_T, y_ddot_T, log_sigma_y]."""
        return self.forward_normalized(self.normalize(raw_window))


class HistoryResampler20Hz:
    """Maintains causal sensor history at any input rate (e.g. 40 Hz) and resamples to 20 Hz."""

    def __init__(self, target_hz: float = 20.0, history_steps: int = 40, max_buffer_s: float = 4.0) -> None:
        self.target_hz = float(target_hz)
        self.target_dt = 1.0 / self.target_hz
        self.history_steps = int(history_steps)
        maxlen = max(history_steps * 4, int(round(max_buffer_s * 100)))
        self._stamps: deque[float] = deque(maxlen=maxlen)
        self._frames: deque[np.ndarray] = deque(maxlen=maxlen)

    def reset(self) -> None:
        self._stamps.clear()
        self._frames.clear()

    def push(self, stamp_s: float, features: np.ndarray) -> None:
        vec = np.asarray(features, dtype=np.float32)
        if vec.shape != (len(FEATURE_NAMES),):
            raise ValueError(f"Expected ({len(FEATURE_NAMES)},) features, got {vec.shape}")
        if self._stamps and stamp_s <= self._stamps[-1]:
            if abs(stamp_s - self._stamps[-1]) < 1e-9:
                self._frames[-1] = vec
                return
            raise ValueError(f"Non-monotonic timestamp: {stamp_s} <= {self._stamps[-1]}")
        self._stamps.append(float(stamp_s))
        self._frames.append(vec)

    def window(self) -> np.ndarray:
        """Return [history_steps, 16] resampled at target_hz ending at the latest timestamp."""
        if not self._stamps:
            raise RuntimeError("History buffer is empty; push at least one frame first.")
        stamps = np.asarray(self._stamps, dtype=np.float64)
        frames = np.stack(self._frames, axis=0)  # [N, 16]
        t_end = stamps[-1]
        query_t = t_end - (self.history_steps - 1 - np.arange(self.history_steps, dtype=np.float64)) * self.target_dt
        # Linear interpolation with left-edge clamping for warm-up (< 2.0 s)
        out = np.empty((self.history_steps, frames.shape[1]), dtype=np.float32)
        for col in range(frames.shape[1]):
            out[:, col] = np.interp(query_t, stamps, frames[:, col]).astype(np.float32)
        return out


@dataclass
class LaneHoldState:
    """Tracks causal single/both-lane fallback matching NN preprocessing.py."""

    last_left: tuple[float, float, float, float] | None = None
    last_right: tuple[float, float, float, float] | None = None
    both_missing_since_s: float | None = None
    prev_steering_wheel_rad: float | None = None
    prev_stamp_s: float | None = None


def hfsm_inputs_to_nn_features(
    stamp_s: float,
    left_line: Any | None,
    right_line: Any | None,
    speed_mps: float,
    yaw_rate_rps: float,
    lateral_accel_mps2: float,
    front_wheel_angle_rad: float,
    steering_ratio: float = 15.0,
    lane_width_m: float = 3.7,
    hold_state: LaneHoldState | None = None,
    front_wheel_rate_rps: float | None = None,
    lane_hold_seconds: float = 0.5,
) -> np.ndarray:
    """Convert HFSM_Frenet_quintic_LTV-MPC inputs into one 16-feature NN frame (left-positive).

    Coordinate mapping:
    - HFSM LaneLine.c0 is right-positive (left < 0, right > 0); NN expects left-positive.
    - HFSM LaneLine.c1 is (road_yaw - ego_yaw), already left-positive.
    - HFSM LaneLine.c2_inv_km is 0.5 * kappa * 1000, already left-positive in 1/km -> /1000 for 1/m.
    - NN steering_angle_rad is steering-wheel angle (front_wheel_angle_rad * steering_ratio).
    """
    state = hold_state if hold_state is not None else LaneHoldState()

    def _convert(line: Any | None) -> tuple[tuple[float, float, float, float], float] | None:
        if line is None:
            return None
        conf = float(getattr(line, "existence_probability", getattr(line, "confidence", 1.0)))
        c0 = -float(line.c0)
        c1 = float(line.c1)
        c2 = float(getattr(line, "c2_inv_km", 0.0)) / 1000.0
        c3 = float(getattr(line, "c3", 0.0))
        return (c0, c1, c2, c3), float(np.clip(conf, 0.0, 1.0))

    left = _convert(left_line)
    right = _convert(right_line)
    validity = 1.0

    if left is not None and right is not None:
        l_poly, l_conf = left
        r_poly, r_conf = right
        state.last_left, state.last_right = l_poly, r_poly
        state.both_missing_since_s = None
    elif left is not None:
        l_poly, l_conf = left
        r_poly = (l_poly[0] - lane_width_m, l_poly[1], l_poly[2], l_poly[3])
        r_conf = 0.0
        state.last_left, state.last_right = l_poly, r_poly
        state.both_missing_since_s = None
    elif right is not None:
        r_poly, r_conf = right
        l_poly = (r_poly[0] + lane_width_m, r_poly[1], r_poly[2], r_poly[3])
        l_conf = 0.0
        state.last_left, state.last_right = l_poly, r_poly
        state.both_missing_since_s = None
    else:
        if state.both_missing_since_s is None:
            state.both_missing_since_s = stamp_s
        elapsed = max(0.0, stamp_s - state.both_missing_since_s)
        decay = max(0.0, 1.0 - elapsed / max(lane_hold_seconds, 1e-6))
        validity = decay
        l_poly = state.last_left or (0.5 * lane_width_m, 0.0, 0.0, 0.0)
        r_poly = state.last_right or (-0.5 * lane_width_m, 0.0, 0.0, 0.0)
        l_conf = 0.25 * decay
        r_conf = 0.25 * decay

    swa = float(front_wheel_angle_rad) * float(steering_ratio)
    if front_wheel_rate_rps is not None:
        sw_rate = float(front_wheel_rate_rps) * float(steering_ratio)
    elif state.prev_steering_wheel_rad is not None and state.prev_stamp_s is not None and stamp_s > state.prev_stamp_s:
        sw_rate = (swa - state.prev_steering_wheel_rad) / (stamp_s - state.prev_stamp_s)
    else:
        sw_rate = 0.0
    state.prev_steering_wheel_rad = swa
    state.prev_stamp_s = stamp_s

    return np.array(
        [
            l_poly[0], l_poly[1], l_poly[2], l_poly[3],
            r_poly[0], r_poly[1], r_poly[2], r_poly[3],
            l_conf, r_conf,
            float(speed_mps), float(yaw_rate_rps), float(lateral_accel_mps2),
            swa, sw_rate, float(validity),
        ],
        dtype=np.float32,
    )


def evaluate_ego_quintic(
    t_eval: np.ndarray,
    y_T: float,
    y_dot_T: float,
    y_ddot_T: float,
    horizon_s: float = 2.5,
    y0: float = 0.0,
    y_dot0: float = 0.0,
    y_ddot0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct smooth quintic (y, y_dot, y_ddot) in ego frame for t_eval."""
    T = float(horizon_s)
    c0, c1, c2 = float(y0), float(y_dot0), 0.5 * float(y_ddot0)
    mat = np.array(
        [
            [T**3, T**4, T**5],
            [3 * T**2, 4 * T**3, 5 * T**4],
            [6 * T, 12 * T**2, 20 * T**3],
        ],
        dtype=np.float64,
    )
    rhs = np.array([y_T - c0 - c1 * T - c2 * T**2, y_dot_T - c1 - 2 * c2 * T, y_ddot_T - 2 * c2])
    c3, c4, c5 = np.linalg.solve(mat, rhs)

    t = np.clip(np.asarray(t_eval, dtype=np.float64), 0.0, T)
    y = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5
    yd = c1 + 2 * c2 * t + 3 * c3 * t**2 + 4 * c4 * t**3 + 5 * c5 * t**4
    ydd = 2 * c2 + 6 * c3 * t + 12 * c4 * t**2 + 20 * c5 * t**3
    return y, yd, ydd


class TCN40HzAdapter:
    """Stateful 40 Hz wrapper around NumpyLateralTCN with optional 20 Hz inference cadence."""

    def __init__(
        self,
        bundle_path: Path | str,
        steering_ratio: float = 15.0,
        lane_width_m: float = 3.7,
        infer_every_n_steps: int = 2,
    ) -> None:
        self.model = NumpyLateralTCN(bundle_path)
        self.resampler = HistoryResampler20Hz(
            target_hz=self.model.sample_rate_hz,
            history_steps=self.model.history_steps,
        )
        self.hold_state = LaneHoldState()
        self.steering_ratio = float(steering_ratio)
        self.lane_width_m = float(lane_width_m)
        self.infer_every_n_steps = max(1, int(infer_every_n_steps))
        self._step_count = 0
        self._last_prediction: np.ndarray | None = None

    def reset(self) -> None:
        self.resampler.reset()
        self.hold_state = LaneHoldState()
        self._step_count = 0
        self._last_prediction = None

    def step(
        self,
        stamp_s: float,
        left_line: Any | None,
        right_line: Any | None,
        speed_mps: float,
        yaw_rate_rps: float,
        lateral_accel_mps2: float,
        front_wheel_angle_rad: float,
        front_wheel_rate_rps: float | None = None,
    ) -> np.ndarray:
        """Push one 40 Hz sample and return [y_T, y_dot_T, y_ddot_T, log_sigma_y] (left-positive)."""
        features = hfsm_inputs_to_nn_features(
            stamp_s=stamp_s,
            left_line=left_line,
            right_line=right_line,
            speed_mps=speed_mps,
            yaw_rate_rps=yaw_rate_rps,
            lateral_accel_mps2=lateral_accel_mps2,
            front_wheel_angle_rad=front_wheel_angle_rad,
            steering_ratio=self.steering_ratio,
            lane_width_m=self.lane_width_m,
            hold_state=self.hold_state,
            front_wheel_rate_rps=front_wheel_rate_rps,
        )
        self.resampler.push(stamp_s, features)
        if self._last_prediction is None or (self._step_count % self.infer_every_n_steps == 0):
            window_20hz = self.resampler.window()
            self._last_prediction = self.model.predict(window_20hz)
        self._step_count += 1
        return self._last_prediction.copy()


def export_runtime_bundle(
    checkpoint_path: Path | str,
    normalization_path: Path | str = Path("data/processed/normalization.json"),
    output_path: Path | str | None = None,
) -> Path:
    """Export a PyTorch TCN checkpoint + normalization stats into a pure-NumPy .npz bundle."""
    import torch
    from tcn_model import LateralPlanningTCN

    ckpt_path = Path(checkpoint_path)
    norm_path = Path(normalization_path)
    out_path = Path(output_path) if output_path is not None else ckpt_path.with_name("lateral_tcn_runtime.npz")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    cfg_dict: dict[str, Any] = checkpoint.get("model_config", {})
    activation = str(cfg_dict.get("activation", "relu"))

    # Verify load against LateralPlanningTCN
    model = LateralPlanningTCN(**cfg_dict) if cfg_dict else LateralPlanningTCN()
    model.load_state_dict(state)
    model.eval()

    norm = json.loads(norm_path.read_text(encoding="utf-8"))
    cfg = PipelineConfig()
    payload: dict[str, Any] = {
        f"w__{name}": tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in state.items()
    }
    payload.update(
        {
            "clip_low": np.asarray(norm["clip_low"], dtype=np.float32),
            "clip_high": np.asarray(norm["clip_high"], dtype=np.float32),
            "mean": np.asarray(norm["mean"], dtype=np.float32),
            "std": np.asarray(norm["std"], dtype=np.float32),
            "activation": np.array(activation),
            "sample_rate_hz": np.array(cfg.sample_rate_hz, dtype=np.float32),
            "history_steps": np.array(cfg.history_steps, dtype=np.int32),
            "horizon_seconds": np.array(cfg.horizon_seconds, dtype=np.float32),
            "checkpoint_sha256": np.array(hashlib.sha256(ckpt_path.read_bytes()).hexdigest()),
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return out_path
