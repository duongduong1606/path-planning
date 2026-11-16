"""Experimental post-planner safety shield, excluded from AI acceptance gates."""

from __future__ import annotations

import numpy as np


def quintic_terminal_basis(horizon_s: float = 2.5, samples: int = 51) -> np.ndarray:
    """Linear map from [y_T, ydot_T, yddot_T] to sampled lateral path."""
    matrix = np.array([
        [horizon_s**3, horizon_s**4, horizon_s**5],
        [3*horizon_s**2, 4*horizon_s**3, 5*horizon_s**4],
        [6*horizon_s, 12*horizon_s**2, 20*horizon_s**3],
    ], dtype=np.float64)
    time = np.linspace(0.0, horizon_s, samples)
    polynomial_basis = np.column_stack((time**3, time**4, time**5))
    return polynomial_basis @ np.linalg.inv(matrix)


def project_to_lane_corridor(
    predictions: np.ndarray,
    features: np.ndarray,
    speed: np.ndarray,
    horizon_s: float = 2.5,
    margin_m: float = 0.20,
    max_iterations: int = 200,
    tolerance_m: float = 1e-6,
    max_abs_lateral_velocity: float = 3.0,
    max_abs_lateral_acceleration: float = 3.0,
    max_terminal_state_delta_l2: float = 4.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Project unsafe terminal states onto sampled lane half-space constraints.

    This is a shield diagnostic, not part of the learned planner. The first
    three outputs are adjusted and log_sigma is retained. Non-convergent or
    excessive corrections are rejected for a future fallback policy to handle.
    """
    projected = np.asarray(predictions, dtype=np.float64).copy()
    original = projected.copy()
    basis = quintic_terminal_basis(horizon_s)
    time = np.linspace(0.0, horizon_s, len(basis))
    metric_inverse = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    attempted = changed = rejected = 0

    for index, (feature, velocity) in enumerate(zip(features, speed)):
        x = time * max(float(velocity), 0.0)
        left = np.polynomial.polynomial.polyval(x, feature[:4])
        right = np.polynomial.polynomial.polyval(x, feature[4:8])
        confidence = float(np.clip(np.min(feature[8:10]), 0.0, 1.0))
        effective_margin = margin_m * (0.25 + 0.75 * confidence)
        valid = np.isfinite(left) & np.isfinite(right) & (
            left - right > 2.0 * effective_margin
        ) & (np.einsum("ij,ij->i", basis, basis) > 1e-12)
        if not np.any(valid):
            continue
        lower, upper = right + effective_margin, left - effective_margin
        state = projected[index, :3].copy()
        initial_path = basis @ state
        if np.all((initial_path[valid] >= lower[valid]) & (initial_path[valid] <= upper[valid])):
            continue
        attempted += 1
        for _ in range(max_iterations):
            largest = 0.0
            for row, low, high in zip(basis[valid], lower[valid], upper[valid]):
                value = float(row @ state)
                if value > high:
                    violation = value - high
                elif value < low:
                    violation = value - low
                else:
                    continue
                direction = metric_inverse * row
                denominator = float(row @ direction)
                if denominator > 1e-12:
                    state -= violation / denominator * direction
                    largest = max(largest, abs(violation))
            state[1] = np.clip(state[1], -max_abs_lateral_velocity, max_abs_lateral_velocity)
            state[2] = np.clip(state[2], -max_abs_lateral_acceleration, max_abs_lateral_acceleration)
            if largest <= tolerance_m:
                break
        path = basis @ state
        is_feasible = bool(np.all(
            (path[valid] >= lower[valid] - tolerance_m) &
            (path[valid] <= upper[valid] + tolerance_m)
        ))
        adjustment = float(np.linalg.norm(state - original[index, :3]))
        if is_feasible and adjustment <= max_terminal_state_delta_l2:
            projected[index, :3] = state
            changed += 1
        else:
            rejected += 1

    projected = projected.astype(np.asarray(predictions).dtype, copy=False)
    delta = projected[:, :3] - original[:, :3]
    stats = {
        "attempted_fraction": attempted / max(len(projected), 1),
        "changed_fraction": changed / max(len(projected), 1),
        "rejected_fraction": rejected / max(len(projected), 1),
        "converged_fraction_of_attempted": changed / max(attempted, 1),
        "mean_terminal_state_delta_l2": float(np.linalg.norm(delta, axis=1).mean()),
        "max_terminal_state_delta_l2": float(np.linalg.norm(delta, axis=1).max(initial=0.0)),
    }
    return projected, stats
