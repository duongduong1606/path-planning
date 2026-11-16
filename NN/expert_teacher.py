"""Human-referenced constrained Frenet teacher for offline label refinement.

The expert projects a measured terminal lateral state onto sampled lane,
lateral-acceleration and jerk constraints. It is a training-label generator,
not a runtime safety shield.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExpertConfig:
    horizon_s: float = 2.5
    samples: int = 51
    lane_margin_m: float = 0.20
    minimum_lane_confidence: float = 0.50
    max_abs_lateral_velocity_mps: float = 3.0
    max_abs_lateral_acceleration_mps2: float = 2.5
    max_abs_lateral_jerk_mps3: float = 5.0
    target_weights: tuple[float, float, float] = (1.0, 0.50, 0.25)
    max_iterations: int = 250
    tolerance: float = 1e-5
    max_correction_l2: float = 4.0


def quintic_bases(config: ExpertConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Maps terminal [y, ydot, yddot] to y(t), yddot(t), jerk(t)."""
    t = np.linspace(0.0, config.horizon_s, config.samples)
    horizon = config.horizon_s
    terminal_matrix = np.array(
        [
            [horizon**3, horizon**4, horizon**5],
            [3*horizon**2, 4*horizon**3, 5*horizon**4],
            [6*horizon, 12*horizon**2, 20*horizon**3],
        ],
        np.float64,
    )
    terminal_to_coefficients = np.linalg.inv(terminal_matrix)
    position = np.column_stack((t**3, t**4, t**5)) @ terminal_to_coefficients
    acceleration = np.column_stack((6*t, 12*t**2, 20*t**3)) @ terminal_to_coefficients
    jerk = np.column_stack((np.full_like(t, 6.0), 24*t, 60*t**2)) @ terminal_to_coefficients
    return t, position, acceleration, jerk


def _maximum_violation(
    state: np.ndarray, rows: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    values = rows @ state
    below = float(np.max(lower-values, initial=0.0))
    above = float(np.max(values-upper, initial=0.0))
    return max(below, above)


def _weighted_project_to_slabs(
    reference: np.ndarray,
    rows: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: ExpertConfig,
) -> tuple[np.ndarray, bool, int]:
    """Dykstra projection onto linear slabs in the weighted target metric."""
    weights = np.asarray(config.target_weights, np.float64)
    if np.any(weights <= 0):
        raise ValueError("target_weights must be strictly positive")
    inverse_sqrt = 1.0 / np.sqrt(weights)
    transformed_rows = rows * inverse_sqrt[None, :]
    shifted_lower = lower - rows @ reference
    shifted_upper = upper - rows @ reference
    keep = np.einsum("ij,ij->i", transformed_rows, transformed_rows) > 1e-12
    transformed_rows = transformed_rows[keep]
    shifted_lower, shifted_upper = shifted_lower[keep], shifted_upper[keep]
    corrections = np.zeros_like(transformed_rows)
    state = np.zeros(3, np.float64)
    for iteration in range(1, config.max_iterations + 1):
        for index, (row, low, high) in enumerate(
            zip(transformed_rows, shifted_lower, shifted_upper)
        ):
            candidate = state + corrections[index]
            value = float(row @ candidate)
            if value > high:
                projected = candidate - (value-high) / float(row @ row) * row
            elif value < low:
                projected = candidate - (value-low) / float(row @ row) * row
            else:
                projected = candidate
            corrections[index] = candidate - projected
            state = projected
        result = reference + inverse_sqrt * state
        if _maximum_violation(result, rows, lower, upper) <= config.tolerance:
            return result, True, iteration
    result = reference + inverse_sqrt * state
    converged = _maximum_violation(result, rows, lower, upper) <= config.tolerance
    return result, converged, config.max_iterations


def build_constraints(
    feature: np.ndarray, config: ExpertConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time, position, acceleration, jerk = quintic_bases(config)
    speed = max(float(feature[10]), 0.0)
    # Lane polynomials are calibrated over 5..60 m.
    x = np.clip(speed*time, 5.0, 60.0)
    left = np.polynomial.polynomial.polyval(x, feature[:4])
    right = np.polynomial.polynomial.polyval(x, feature[4:8])
    rows = np.concatenate(
        (position, acceleration, jerk, np.array([[0.0, 1.0, 0.0]], np.float64)), axis=0
    )
    lower = np.concatenate(
        (
            right + config.lane_margin_m,
            np.full(config.samples, -config.max_abs_lateral_acceleration_mps2),
            np.full(config.samples, -config.max_abs_lateral_jerk_mps3),
            [-config.max_abs_lateral_velocity_mps],
        )
    )
    upper = np.concatenate(
        (
            left - config.lane_margin_m,
            np.full(config.samples, config.max_abs_lateral_acceleration_mps2),
            np.full(config.samples, config.max_abs_lateral_jerk_mps3),
            [config.max_abs_lateral_velocity_mps],
        )
    )
    return rows, lower, upper


def project_human_target(
    target: np.ndarray, feature: np.ndarray, config: ExpertConfig = ExpertConfig()
) -> tuple[np.ndarray, bool, bool, float, int, str]:
    """Return target, valid, changed, correction norm, iterations and reason."""
    target = np.asarray(target, np.float64)
    feature = np.asarray(feature, np.float64)
    if target.shape != (3,) or feature.shape != (16,):
        raise ValueError("expected target [3] and feature [16]")
    if not np.isfinite(target).all() or not np.isfinite(feature).all():
        return target.astype(np.float32), False, False, 0.0, 0, "non_finite"
    if min(feature[8], feature[9]) < config.minimum_lane_confidence or feature[15] <= 0:
        return target.astype(np.float32), False, False, 0.0, 0, "lane_confidence"
    rows, lower, upper = build_constraints(feature, config)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
        return target.astype(np.float32), False, False, 0.0, 0, "invalid_corridor"
    if _maximum_violation(target, rows, lower, upper) <= config.tolerance:
        # No duplicate teacher loss when human truth is already exemplary.
        return target.astype(np.float32), False, False, 0.0, 0, "already_feasible"
    projected, converged, iterations = _weighted_project_to_slabs(
        target, rows, lower, upper, config
    )
    correction = float(np.linalg.norm(projected-target))
    valid = bool(converged and correction <= config.max_correction_l2)
    reason = "projected" if valid else "projection_failed"
    return projected.astype(np.float32), valid, valid, correction, iterations, reason


def generate_expert_targets(
    targets: np.ndarray,
    features: np.ndarray,
    candidate_mask: np.ndarray,
    config: ExpertConfig = ExpertConfig(),
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    count = len(targets)
    output = np.asarray(targets, np.float32).copy()
    std = np.ones((count, 3), np.float32)
    valid = np.zeros(count, bool)
    changed = np.zeros(count, bool)
    correction = np.zeros(count, np.float32)
    iterations = np.zeros(count, np.int16)
    reason_counts: dict[str, int] = {}
    for index in np.flatnonzero(candidate_mask):
        result, is_valid, is_changed, delta, used_iterations, reason = project_human_target(
            targets[index], features[index], config
        )
        output[index] = result
        valid[index] = is_valid
        changed[index] = is_changed
        correction[index] = delta
        iterations[index] = used_iterations
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if is_valid:
            std[index] = 0.05 + min(delta, 2.0)*np.array([0.15, 0.20, 0.25], np.float32)
    arrays = {
        "target": output,
        "target_std": std,
        "valid": valid,
        "changed": changed,
        "correction_l2": correction,
        "iterations": iterations,
    }
    candidates = int(np.asarray(candidate_mask, bool).sum())
    statistics: dict[str, float | int] = {
        "frames": count,
        "candidates": candidates,
        "valid_expert_labels": int(valid.sum()),
        "expert_label_fraction_of_candidates": float(valid.sum()/max(candidates, 1)),
        "mean_correction_l2": float(correction[valid].mean()) if valid.any() else 0.0,
        "p95_correction_l2": float(np.percentile(correction[valid], 95)) if valid.any() else 0.0,
        "mean_iterations": float(iterations[valid].mean()) if valid.any() else 0.0,
    }
    statistics.update({f"reason_{key}": value for key, value in reason_counts.items()})
    return arrays, statistics

