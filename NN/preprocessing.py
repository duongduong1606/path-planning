"""Synchronization, ego-frame truth, lane completion and feature assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from pipeline_config import FEATURE_NAMES, PipelineConfig


def load_array(path: Path) -> np.ndarray:
    """Load comma2k19 arrays, whose files intentionally have no .npy suffix."""
    return np.load(path, allow_pickle=False)


def _clean_times(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    values = np.asarray(values)
    if len(times) != len(values):
        raise ValueError("timestamp and value lengths differ")
    good = np.isfinite(times)
    times, values = times[good], values[good]
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    unique = np.r_[True, np.diff(times) > 0]
    return times[unique], values[unique]


def interpolate_continuous(
    target_times: np.ndarray, signal_times: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate without silently extrapolating beyond coverage."""
    signal_times, values = _clean_times(signal_times, values)
    target_times = np.asarray(target_times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    output = np.full((len(target_times), values.shape[1]), np.nan, np.float64)
    covered = (target_times >= signal_times[0]) & (target_times <= signal_times[-1])
    for column in range(values.shape[1]):
        finite = np.isfinite(values[:, column])
        if finite.sum() >= 2:
            output[covered, column] = np.interp(
                target_times[covered], signal_times[finite], values[finite, column]
            )
    return output.squeeze(-1) if output.shape[1] == 1 else output, covered


def previous_hold(
    target_times: np.ndarray, signal_times: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Causal previous-value hold for discrete signals."""
    signal_times, values = _clean_times(signal_times, values)
    indices = np.searchsorted(signal_times, target_times, side="right") - 1
    covered = (indices >= 0) & (target_times <= signal_times[-1])
    output = np.zeros((len(target_times),) + values.shape[1:], dtype=values.dtype)
    output[covered] = values[indices[covered]]
    return output, covered


def quaternion_to_rotation(quaternions: np.ndarray) -> np.ndarray:
    """Hamilton [w,x,y,z] quaternion(s) to rotation matrix/matrices."""
    q = np.asarray(quaternions, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = np.moveaxis(q, -1, 0)
    result = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 0] = 1 - 2 * (y*y + z*z)
    result[..., 0, 1] = 2 * (x*y - z*w)
    result[..., 0, 2] = 2 * (x*z + y*w)
    result[..., 1, 0] = 2 * (x*y + z*w)
    result[..., 1, 1] = 1 - 2 * (x*x + z*z)
    result[..., 1, 2] = 2 * (y*z - x*w)
    result[..., 2, 0] = 2 * (x*z - y*w)
    result[..., 2, 1] = 2 * (y*z + x*w)
    result[..., 2, 2] = 1 - 2 * (x*x + y*y)
    return result


def build_ego_truth(
    frame_times: np.ndarray,
    positions_ecef: np.ndarray,
    orientations_ecef_from_device: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return [y,ydot,yddot] terminal labels and future ego trajectories.

    comma2k19 device coordinates are x-forward, y-right, z-down. This project
    uses y-left, hence the sign inversion after rotation into the current frame.
    """
    n = len(frame_times)
    h = config.horizon_steps
    target = np.full((n, 3), np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    future_y = np.full((n, h + 1), np.nan, dtype=np.float64)
    rotations = quaternion_to_rotation(orientations_ecef_from_device)
    fit_count = max(3, min(config.derivative_fit_steps, h + 1))

    for current in range(n - h):
        indices = np.arange(current, current + h + 1)
        delta = positions_ecef[indices] - positions_ecef[current]
        device = delta @ rotations[current]
        x = device[:, 0]
        y_left = -device[:, 1]
        relative_t = frame_times[indices] - frame_times[current]
        if not (np.isfinite(x).all() and np.isfinite(y_left).all()):
            continue
        if x[-1] <= 0.0 or x[-1] >= 120.0 or abs(y_left[-1]) >= 20.0:
            continue
        fit_t = relative_t[-fit_count:] - relative_t[-1]
        coefficients = np.polyfit(fit_t, y_left[-fit_count:], deg=2)
        target[current] = (y_left[-1], coefficients[1], 2.0 * coefficients[0])
        future_y[current] = y_left
        valid[current] = np.isfinite(target[current]).all()
    return target.astype(np.float32), valid, future_y.astype(np.float32)


@dataclass
class LaneSeries:
    left: np.ndarray
    right: np.ndarray
    left_confidence: np.ndarray
    right_confidence: np.ndarray
    validity: np.ndarray
    observed: np.ndarray


def complete_lane_series(
    left: np.ndarray,
    right: np.ndarray,
    left_confidence: np.ndarray,
    right_confidence: np.ndarray,
    config: PipelineConfig,
) -> LaneSeries:
    """Causally complete missing boundaries; never invent confidence."""
    left = np.array(left, dtype=np.float64, copy=True)
    right = np.array(right, dtype=np.float64, copy=True)
    lc = np.array(left_confidence, dtype=np.float64, copy=True)
    rc = np.array(right_confidence, dtype=np.float64, copy=True)
    n = len(left)
    observed = np.column_stack((np.isfinite(left).all(1), np.isfinite(right).all(1)))
    validity = np.zeros(n, dtype=np.float64)
    previous_left = previous_right = None
    previous_index = -10**9

    for i in range(n):
        has_left, has_right = observed[i]
        if has_left and has_right:
            validity[i] = 1.0
        elif has_left:
            right[i] = left[i]
            right[i, 0] -= config.lane_width_m
            rc[i] = 0.0
            validity[i] = 0.5
        elif has_right:
            left[i] = right[i]
            left[i, 0] += config.lane_width_m
            lc[i] = 0.0
            validity[i] = 0.5
        elif previous_left is not None and i - previous_index <= config.lane_hold_steps:
            decay = 1.0 - (i - previous_index) / (config.lane_hold_steps + 1.0)
            left[i], right[i] = previous_left, previous_right
            lc[i], rc[i] = previous_lc * decay, previous_rc * decay
            validity[i] = 0.5 * decay
        else:
            lc[i] = rc[i] = 0.0
            continue

        # A held frame must not reset the hold timer; otherwise missing lanes
        # could be propagated forever one frame at a time.
        if has_left or has_right:
            previous_left, previous_right = left[i].copy(), right[i].copy()
            previous_lc, previous_rc = lc[i], rc[i]
            previous_index = i

    return LaneSeries(
        left.astype(np.float32), right.astype(np.float32),
        lc.astype(np.float32), rc.astype(np.float32), validity.astype(np.float32), observed,
    )


def causal_rate(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    dt = np.diff(times)
    result[1:] = np.divide(
        np.diff(values), dt, out=np.zeros_like(dt), where=np.abs(dt) > 1e-9
    )
    result[0] = result[1] if len(result) > 1 else 0.0
    return result


def assemble_features(
    lanes: LaneSeries,
    speed: np.ndarray,
    gyro: np.ndarray,
    accelerometer: np.ndarray,
    steering_degrees: np.ndarray,
    frame_times: np.ndarray,
) -> np.ndarray:
    """Assemble the fixed 16-feature runtime interface."""
    steering = np.deg2rad(np.asarray(steering_degrees).reshape(-1))
    gyro = np.asarray(gyro)
    accelerometer = np.asarray(accelerometer)
    # Device axes: x forward, y right, z down; planning convention is y left.
    yaw_rate = -gyro[:, 2]
    lateral_accel = -accelerometer[:, 1]
    features = np.column_stack(
        (lanes.left, lanes.right, lanes.left_confidence, lanes.right_confidence,
         np.asarray(speed).reshape(-1), yaw_rate, lateral_accel, steering,
         causal_rate(steering, frame_times), lanes.validity)
    )
    if features.shape[1] != len(FEATURE_NAMES):
        raise AssertionError("feature contract changed unexpectedly")
    return features.astype(np.float32)


def window_mask(
    frame_valid: np.ndarray,
    target_valid: np.ndarray,
    manual: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all valid anchors and the strict full-interval manual subset."""
    n = len(frame_valid)
    all_valid = np.zeros(n, dtype=bool)
    manual_valid = np.zeros(n, dtype=bool)
    first = config.history_steps - 1
    last = n - config.horizon_steps
    for anchor in range(first, last):
        start, stop = anchor - config.history_steps + 1, anchor + config.horizon_steps + 1
        all_valid[anchor] = frame_valid[start:anchor + 1].all() and target_valid[anchor]
        manual_valid[anchor] = all_valid[anchor] and manual[start:stop].all()
    return all_valid, manual_valid

