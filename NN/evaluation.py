"""Offline terminal-state, reconstructed-path, safety and uncertainty metrics."""

from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np


def quintic_trajectory(
    terminal: np.ndarray, horizon_s: float = 2.5, samples: int = 51
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Minimum-degree path satisfying zero initial and predicted final state."""
    y, velocity, acceleration = np.asarray(terminal)[..., :3].T
    t = np.linspace(0.0, horizon_s, samples)
    matrix = np.array([
        [horizon_s**3, horizon_s**4, horizon_s**5],
        [3*horizon_s**2, 4*horizon_s**3, 5*horizon_s**4],
        [6*horizon_s, 12*horizon_s**2, 20*horizon_s**3],
    ])
    coefficients = np.linalg.solve(matrix, np.stack((y, velocity, acceleration)))
    a3, a4, a5 = coefficients
    path = a3*t**3 + a4*t**4 + a5*t**5
    dy = 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
    ddy = 6*a3*t + 12*a4*t**2 + 20*a5*t**3
    jerk = 6*a3 + 24*a4*t + 60*a5*t**2
    return path, dy, ddy, jerk


def _errors(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction) - np.asarray(truth)
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2))),
            "p95_abs": float(np.percentile(np.abs(error), 95))}


def terminal_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    names = ("y_T", "y_dot_T", "y_ddot_T")
    result = {}
    for i, name in enumerate(names):
        result.update({f"{name}_{key}": value for key, value in _errors(predictions[:, i], targets[:, i]).items()})
    return result


def uncertainty_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    residual = predictions[:, 0] - targets[:, 0]
    sigma = np.exp(np.clip(predictions[:, 3], -6, 3))
    nll = .5*np.log(2*np.pi) + np.log(sigma) + .5*(residual/sigma)**2
    return {"gaussian_nll": float(nll.mean()), "coverage_68": float((np.abs(residual) <= sigma).mean()),
            "coverage_90": float((np.abs(residual) <= 1.644854*sigma).mean()),
            "coverage_95": float((np.abs(residual) <= 1.959964*sigma).mean()),
            "sharpness_mean_sigma": float(sigma.mean())}


def path_metrics(
    predictions: np.ndarray, future_y: np.ndarray, speed: np.ndarray, horizon_s: float = 2.5
) -> dict[str, float]:
    paths, headings, accelerations, jerks = [], [], [], []
    for prediction in predictions:
        path, dy, ddy, jerk = quintic_trajectory(prediction, horizon_s, future_y.shape[1])
        paths.append(path); headings.append(dy); accelerations.append(ddy); jerks.append(jerk)
    paths = np.asarray(paths)
    errors = paths - future_y
    speed = np.maximum(np.asarray(speed), 0.5)[:, None]
    heading_error = np.arctan(headings/speed) - np.arctan(np.gradient(future_y, horizon_s/(future_y.shape[1]-1), axis=1)/speed)
    return {"lateral_ADE": float(np.mean(np.abs(errors))), "lateral_FDE": float(np.mean(np.abs(errors[:, -1]))),
            "max_lateral_error_p95": float(np.percentile(np.max(np.abs(errors), axis=1), 95)),
            "heading_error_mae_rad": float(np.mean(np.abs(heading_error[:, -1]))),
            "max_lateral_accel_p95": float(np.percentile(np.max(np.abs(accelerations), axis=1), 95)),
            "max_lateral_jerk_p95": float(np.percentile(np.max(np.abs(jerks), axis=1), 95))}


def lane_boundary_violation(
    predictions: np.ndarray, features: np.ndarray, speed: np.ndarray, horizon_s: float = 2.5
) -> float:
    return float(lane_boundary_violation_mask(predictions, features, speed, horizon_s).mean())


def lane_boundary_violation_mask(
    predictions: np.ndarray, features: np.ndarray, speed: np.ndarray, horizon_s: float = 2.5
) -> np.ndarray:
    violations = []
    for prediction, feature, velocity in zip(predictions, features, speed):
        path, _, _, _ = quintic_trajectory(prediction, horizon_s)
        # Lane cubics are calibrated only in the 5..60 m forward interval.
        # Evaluate against the nearest supported station outside that range;
        # extrapolating the cubic back to x=0 can create false violations.
        x = np.clip(
            np.linspace(0, max(float(velocity), 0)*horizon_s, len(path)),
            5.0, 60.0,
        )
        left = np.polynomial.polynomial.polyval(x, feature[:4])
        right = np.polynomial.polynomial.polyval(x, feature[4:8])
        violations.append(np.any((path > left) | (path < right)))
    return np.asarray(violations, dtype=bool)


def lane_behavior_metrics(
    predictions: np.ndarray, targets: np.ndarray, features: np.ndarray,
    speed: np.ndarray, horizon_s: float = 2.5,
) -> dict[str, float]:
    """Separate unsafe stay-lane exits from demonstrated lane crossings.

    A crossing of the current-lane boundary is not automatically an error when
    the measured human future performs the same behavior. This does not claim
    that every demonstrated crossing is safe; it only prevents lane changes
    from being counted as lane-keeping violations.
    """
    predicted_crossing = lane_boundary_violation_mask(
        predictions, features, speed, horizon_s
    )
    truth_terminal = np.column_stack(
        (np.asarray(targets)[:, :3], np.zeros(len(targets), np.float32))
    )
    truth_crossing = lane_boundary_violation_mask(
        truth_terminal, features, speed, horizon_s
    )
    stay = ~truth_crossing
    terminal_error = np.abs(np.asarray(predictions)[:, 0] - np.asarray(targets)[:, 0])

    def conditional_mean(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(values[mask])) if np.any(mask) else float("nan")

    return {
        "truth_current_lane_crossing_fraction": float(truth_crossing.mean()),
        "stay_lane_boundary_violation": conditional_mean(predicted_crossing, stay),
        "truth_crossing_recall": conditional_mean(predicted_crossing, truth_crossing),
        "lateral_FDE_stay_lane": conditional_mean(terminal_error, stay),
        "lateral_FDE_truth_crossing": conditional_mean(terminal_error, truth_crossing),
    }


def clustered_bootstrap(
    values: np.ndarray, routes: np.ndarray, statistic: Callable[[np.ndarray], float] = np.mean,
    repetitions: int = 1000, seed: int = 2026,
) -> tuple[float, float]:
    """95% CI by resampling whole routes, not overlapping windows."""
    unique = np.unique(routes)
    rng = np.random.default_rng(seed)
    estimates = []
    groups = {route: values[routes == route] for route in unique}
    for _ in range(repetitions):
        selected = rng.choice(unique, len(unique), replace=True)
        estimates.append(statistic(np.concatenate([groups[route] for route in selected])))
    return tuple(np.percentile(estimates, [2.5, 97.5]))


def geometric_baselines(current_features: np.ndarray, horizon_s: float = 2.5) -> dict[str, np.ndarray]:
    speed = np.maximum(current_features[:, 10], 0.0)
    yaw_rate = current_features[:, 11]
    angle = yaw_rate*horizon_s
    radius = np.divide(speed, yaw_rate, out=np.full_like(speed, np.inf), where=np.abs(yaw_rate)>1e-4)
    y = np.zeros_like(speed)
    curved = np.isfinite(radius)
    y[curved] = radius[curved] * (1-np.cos(angle[curved]))
    persistence = np.column_stack((y, speed*np.sin(angle), speed*yaw_rate*np.cos(angle), np.zeros_like(y)))
    x_terminal = np.clip(speed*horizon_s, 5.0, 60.0)
    left = np.array([np.polynomial.polynomial.polyval(x, coeff) for x, coeff in zip(x_terminal, current_features[:, :4])])
    right = np.array([np.polynomial.polynomial.polyval(x, coeff) for x, coeff in zip(x_terminal, current_features[:, 4:8])])
    center = .5*(left+right)
    lane_center = np.column_stack((center, np.zeros_like(center), np.zeros_like(center), np.zeros_like(center)))
    return {"constant_curvature": persistence.astype(np.float32), "lane_center_quintic": lane_center.astype(np.float32)}

