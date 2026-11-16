"""Evaluate saved predictions against immutable HDF5 truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation import (
    clustered_bootstrap, geometric_baselines, lane_behavior_metrics, lane_boundary_violation,
    path_metrics, terminal_metrics, uncertainty_metrics,
)
from safety_shield import project_to_lane_corridor


def load_truth(processed: Path, keys: np.ndarray) -> dict[str, np.ndarray]:
    import h5py
    collected = {name: [] for name in ("target", "future_y", "feature", "speed", "route", "manual",
                                                    "brightness", "lane_observed")}
    open_files = {}
    try:
        for key in keys.astype(str):
            relative, anchor_text = key.rsplit("::", 1)
            file = open_files.setdefault(relative, h5py.File(processed / relative, "r"))
            anchor = int(anchor_text)
            collected["target"].append(file["target"][anchor])
            collected["future_y"].append(file["future_y"][anchor])
            collected["feature"].append(file["features"][anchor])
            collected["speed"].append(file["features"][anchor, 10])
            collected["route"].append(file.attrs["route"])
            collected["manual"].append(file["control_source"][anchor] == 0)
            collected["brightness"].append(file["brightness"][anchor])
            collected["lane_observed"].append(file["lane_observed"][anchor])
    finally:
        for file in open_files.values():
            file.close()
    return {name: np.asarray(values) for name, values in collected.items()}


def metrics(prediction: np.ndarray, truth: dict[str, np.ndarray]) -> dict:
    result = terminal_metrics(prediction, truth["target"])
    result.update(path_metrics(prediction, truth["future_y"], truth["speed"]))
    result.update(uncertainty_metrics(prediction, truth["target"]))
    result["lane_boundary_violation"] = lane_boundary_violation(
        prediction, truth["feature"], truth["speed"]
    )
    result.update(lane_behavior_metrics(
        prediction, truth["target"], truth["feature"], truth["speed"]
    ))
    fde = np.abs(prediction[:, 0] - truth["target"][:, 0])
    result["lateral_FDE_ci95_route_bootstrap"] = clustered_bootstrap(fde, truth["route"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, help="NPZ containing keys and prediction [N,4]")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--safety-margin", type=float, default=0.20)
    parser.add_argument("--projected-output", type=Path,
                        help="optional NPZ for safety-projected predictions")
    args = parser.parse_args()
    with np.load(args.predictions, allow_pickle=False) as data:
        keys, prediction = data["keys"], data["prediction"]
    if prediction.shape != (len(keys), 4):
        raise ValueError("prediction must have shape [len(keys),4]")
    truth = load_truth(args.processed, keys)
    projected, projection_stats = project_to_lane_corridor(
        prediction, truth["feature"], truth["speed"], margin_m=args.safety_margin
    )
    report = {
        "model": metrics(prediction, truth),
        "safety_shield_diagnostic": {
            "shielded_system_metrics": metrics(projected, truth),
            "intervention": projection_stats,
            "excluded_from_model_acceptance": True,
        },
        "slices": {},
    }
    baselines = geometric_baselines(truth["feature"])
    report["baselines"] = {name: metrics(values, truth) for name, values in baselines.items()}
    center_curvature = np.abs(truth["feature"][:, 2] + truth["feature"][:, 6] +
                              90*(truth["feature"][:, 3] + truth["feature"][:, 7]))
    slice_masks = {"manual": truth["manual"], "assisted_or_unknown": ~truth["manual"],
                   "low_lane_confidence": np.min(truth["feature"][:, 8:10], axis=1) < .5,
                   "one_lane_missing": truth["lane_observed"].sum(1) == 1,
                   "both_lanes_missing": truth["lane_observed"].sum(1) == 0,
                   "high_speed": truth["speed"] >= 25, "low_speed": truth["speed"] < 15,
                   "high_curvature": center_curvature >= np.nanquantile(center_curvature, .75),
                   "dark": truth["brightness"] < 70, "bright": truth["brightness"] >= 140}
    for name, mask in slice_masks.items():
        if mask.sum() >= 2:
            subset = {key: value[mask] for key, value in truth.items()}
            report["slices"][name] = metrics(prediction[mask], subset)
    best = min(value["lateral_FDE"] for value in report["baselines"].values())
    model_fde = report["model"]["lateral_FDE"]
    acceptance_checks = {
        "fde_improvement_fraction": (best-model_fde)/best if best > 0 else 0.0,
        "improves_fde_by_15_percent": bool(model_fde <= .85*best),
        "coverage_90_in_85_to_95_percent": bool(.85 <= report["model"]["coverage_90"] <= .95),
        "no_stay_lane_boundary_violation_regression": bool(
            report["model"]["stay_lane_boundary_violation"] <= min(
                value["stay_lane_boundary_violation"]
                for value in report["baselines"].values()
            )
        ),
    }
    report["model_acceptance"] = {
        "pass": all(value for key, value in acceptance_checks.items()
                    if key != "fde_improvement_fraction"),
        "checks": acceptance_checks,
        "safety_shield_excluded": True,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if args.projected_output:
        args.projected_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.projected_output, keys=keys, prediction=projected)
    print(text)


if __name__ == "__main__":
    main()

