"""Paired offline A/B comparison for two saved planner prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation import geometric_baselines, lane_boundary_violation_mask
from run_evaluation import load_truth, metrics


def align_candidate(
    baseline_keys: np.ndarray, candidate_keys: np.ndarray, candidate: np.ndarray
) -> np.ndarray:
    positions = {str(key): index for index, key in enumerate(candidate_keys)}
    missing = [str(key) for key in baseline_keys if str(key) not in positions]
    if missing:
        raise ValueError(f"candidate is missing {len(missing)} baseline keys")
    if len(positions) != len(candidate_keys):
        raise ValueError("candidate contains duplicate keys")
    return np.asarray([candidate[positions[str(key)]] for key in baseline_keys])


def route_bootstrap_delta(
    baseline: np.ndarray, candidate: np.ndarray, routes: np.ndarray,
    repetitions: int = 1000, seed: int = 2026,
) -> list[float]:
    unique = np.unique(routes)
    groups = {route: np.flatnonzero(routes == route) for route in unique}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        selected = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([groups[route] for route in selected])
        estimates.append(float(np.mean(candidate[indices] - baseline[indices])))
    return np.percentile(estimates, [2.5, 97.5]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.baseline, allow_pickle=False) as values:
        keys = values["keys"]
        baseline = values["prediction"]
    with np.load(args.candidate, allow_pickle=False) as values:
        candidate = align_candidate(keys, values["keys"], values["prediction"])
    if baseline.shape != candidate.shape or baseline.shape != (len(keys), 4):
        raise ValueError("both prediction arrays must have matching [N,4] shapes")

    truth = load_truth(args.processed, keys)
    baseline_metrics = metrics(baseline, truth)
    candidate_metrics = metrics(candidate, truth)
    baseline_fde = np.abs(baseline[:, 0] - truth["target"][:, 0])
    candidate_fde = np.abs(candidate[:, 0] - truth["target"][:, 0])
    baseline_boundary = lane_boundary_violation_mask(
        baseline, truth["feature"], truth["speed"]
    ).astype(float)
    candidate_boundary = lane_boundary_violation_mask(
        candidate, truth["feature"], truth["speed"]
    ).astype(float)
    truth_terminal = np.column_stack(
        (truth["target"], np.zeros(len(truth["target"]), np.float32))
    )
    stay_lane = ~lane_boundary_violation_mask(
        truth_terminal, truth["feature"], truth["speed"]
    )

    baselines = {
        name: metrics(prediction, truth)
        for name, prediction in geometric_baselines(truth["feature"]).items()
    }
    best_geometric_fde = min(value["lateral_FDE"] for value in baselines.values())
    best_geometric_boundary = min(
        value["lane_boundary_violation"] for value in baselines.values()
    )
    ab_checks = {
        "fde_not_worse": candidate_metrics["lateral_FDE"] <= baseline_metrics["lateral_FDE"],
        "ade_not_worse": candidate_metrics["lateral_ADE"] <= baseline_metrics["lateral_ADE"],
        "stay_lane_boundary_violation_not_worse": (
            candidate_metrics["stay_lane_boundary_violation"]
            <= baseline_metrics["stay_lane_boundary_violation"]
        ),
        "coverage_90_in_85_to_95_percent": 0.85 <= candidate_metrics["coverage_90"] <= 0.95,
        "lateral_acceleration_p95_at_most_2_5": (
            candidate_metrics["max_lateral_accel_p95"] <= 2.5
        ),
        "lateral_jerk_p95_at_most_5": candidate_metrics["max_lateral_jerk_p95"] <= 5.0,
    }
    project_checks = {
        "fde_15_percent_better_than_best_geometric": (
            candidate_metrics["lateral_FDE"] <= 0.85 * best_geometric_fde
        ),
        "stay_lane_boundary_no_worse_than_best_geometric": (
            candidate_metrics["stay_lane_boundary_violation"] <= min(
                value["stay_lane_boundary_violation"] for value in baselines.values()
            )
        ),
    }
    report = {
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta_candidate_minus_baseline": {
            key: candidate_metrics[key] - baseline_metrics[key]
            for key in (
                "lateral_ADE", "lateral_FDE", "lane_boundary_violation",
                "stay_lane_boundary_violation", "truth_crossing_recall",
                "max_lateral_accel_p95", "max_lateral_jerk_p95", "coverage_90",
            )
        },
        "paired_route_bootstrap_ci95": {
            "lateral_FDE_delta": route_bootstrap_delta(
                baseline_fde, candidate_fde, truth["route"]
            ),
            "boundary_violation_delta": route_bootstrap_delta(
                baseline_boundary, candidate_boundary, truth["route"]
            ),
            "stay_lane_boundary_violation_delta": route_bootstrap_delta(
                baseline_boundary[stay_lane], candidate_boundary[stay_lane],
                truth["route"][stay_lane],
            ),
        },
        "ab_promotion": {"pass": all(ab_checks.values()), "checks": ab_checks},
        "project_acceptance": {
            "pass": all(project_checks.values()), "checks": project_checks,
            "best_geometric_FDE": best_geometric_fde,
            "best_geometric_boundary_violation": best_geometric_boundary,
            "best_geometric_stay_lane_boundary_violation": min(
                value["stay_lane_boundary_violation"] for value in baselines.values()
            ),
        },
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
