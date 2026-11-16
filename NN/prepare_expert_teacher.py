"""Generate and audit constrained Frenet expert labels from processed HDF5."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import csv
import json
from pathlib import Path

import numpy as np

from expert_teacher import ExpertConfig, build_constraints, generate_expert_targets, quintic_bases
from openpilot_teacher import save_teacher_cache


def load_rows(processed: Path) -> list[dict[str, str]]:
    with (processed / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def cache_complete(path: Path, expected_frames: int, config: ExpertConfig) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as values:
            metadata = json.loads(str(values["metadata"]))
            return (
                len(values["target"]) == expected_frames
                # JSON stores tuples as lists, so compare canonical JSON values.
                and metadata["config"] == json.loads(json.dumps(asdict(config)))
                and metadata["teacher_type"] == "human_referenced_constrained_frenet"
            )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def command_generate(args: argparse.Namespace) -> None:
    import h5py

    config = ExpertConfig(
        lane_margin_m=args.lane_margin,
        minimum_lane_confidence=args.minimum_lane_confidence,
        max_abs_lateral_acceleration_mps2=args.max_lateral_acceleration,
        max_abs_lateral_jerk_mps3=args.max_lateral_jerk,
    )
    rows = load_rows(args.processed)
    if args.split != "all":
        rows = [row for row in rows if row["split"] == args.split]
    if args.limit_segments is not None:
        rows = rows[:args.limit_segments]
    summaries = []
    for number, row in enumerate(rows, 1):
        source = args.processed / row["hdf5"]
        destination = (args.output / Path(row["hdf5"])).with_suffix(".npz")
        with h5py.File(source, "r") as file:
            frames = len(file["target"])
            if not args.overwrite and cache_complete(destination, frames, config):
                with np.load(destination, allow_pickle=False) as values:
                    summaries.append(json.loads(str(values["metadata"]))["statistics"])
                print(f"[{number}/{len(rows)}] {row['hdf5']}: cached")
                continue
            targets = file["target"][:]
            features = file["features"][:]
            candidate = file["target_valid"][:].astype(bool) & file["frame_valid"][:].astype(bool)
        arrays, statistics = generate_expert_targets(targets, features, candidate, config)
        metadata = {
            "teacher_type": "human_referenced_constrained_frenet",
            "source_hdf5": row["hdf5"],
            "split": row["split"],
            "config": asdict(config),
            "statistics": statistics,
            "note": "offline training labels only; never used as runtime shield",
        }
        save_teacher_cache(destination, arrays, metadata)
        summaries.append(statistics)
        print(
            f"[{number}/{len(rows)}] {row['hdf5']}: "
            f"expert={statistics['valid_expert_labels']}/{statistics['candidates']} "
            f"mean_delta={statistics['mean_correction_l2']:.3f}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "segments_generated": len(summaries),
        "valid_expert_labels": int(sum(item["valid_expert_labels"] for item in summaries)),
        "candidates": int(sum(item["candidates"] for item in summaries)),
        "config": asdict(config),
    }
    aggregate["expert_label_fraction"] = aggregate["valid_expert_labels"] / max(aggregate["candidates"], 1)
    (args.output / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


def command_audit(args: argparse.Namespace) -> None:
    import h5py
    rows = load_rows(args.processed)
    total_candidates = total_valid = segments = 0
    human_violations = expert_violations = 0
    cause_counts = {"lane": 0, "velocity": 0, "acceleration": 0, "jerk": 0}
    corrections, expert_acceleration, expert_jerk = [], [], []
    split_counts = defaultdict(lambda: {"candidates": 0, "valid_expert_labels": 0})
    route_counts = defaultdict(lambda: {"candidates": 0, "valid_expert_labels": 0})
    for row in rows:
        cache_path = (args.teacher_cache / Path(row["hdf5"])).with_suffix(".npz")
        if not cache_path.exists():
            continue
        with np.load(cache_path, allow_pickle=False) as cache, h5py.File(
            args.processed / row["hdf5"], "r"
        ) as file:
            valid = cache["valid"].astype(bool)
            candidate = file["target_valid"][:].astype(bool) & file["frame_valid"][:].astype(bool)
            total_candidates += int(candidate.sum())
            total_valid += int(valid.sum())
            split_counts[row["split"]]["candidates"] += int(candidate.sum())
            split_counts[row["split"]]["valid_expert_labels"] += int(valid.sum())
            route_counts[row["route"]]["candidates"] += int(candidate.sum())
            route_counts[row["route"]]["valid_expert_labels"] += int(valid.sum())
            segments += 1
            if not valid.any():
                continue
            features = file["features"][:][valid]
            speed = features[:, 10]
            human = file["target"][:][valid]
            expert = cache["target"][valid]
            corrections.append(cache["correction_l2"][valid])
            _, _, acceleration_basis, jerk_basis = quintic_bases(ExpertConfig())
            for human_target, target, feature in zip(human, expert, features):
                constraint_rows, lower, upper = build_constraints(feature, ExpertConfig())
                human_values = constraint_rows @ human_target
                expert_values = constraint_rows @ target
                human_bad = (human_values < lower-1e-5) | (human_values > upper+1e-5)
                # Cache targets are float32, so allow their serialization roundoff.
                expert_bad = (expert_values < lower-1e-4) | (expert_values > upper+1e-4)
                # Constraint order is position[51], acceleration[51], jerk[51], velocity[1].
                human_lane_bad = bool(human_bad[:51].any())
                expert_lane_bad = bool(expert_bad[:51].any())
                human_violations += int(human_lane_bad)
                expert_violations += int(expert_lane_bad)
                cause_counts["lane"] += int(human_lane_bad)
                cause_counts["acceleration"] += int(human_bad[51:102].any())
                cause_counts["jerk"] += int(human_bad[102:153].any())
                cause_counts["velocity"] += int(human_bad[153:].any())
                acceleration = acceleration_basis @ target
                jerk = jerk_basis @ target
                expert_acceleration.append(np.max(np.abs(acceleration)))
                expert_jerk.append(np.max(np.abs(jerk)))
    if not segments:
        raise SystemExit("no expert caches found")
    correction_values = np.concatenate(corrections) if corrections else np.empty(0)
    split_report = {
        key: {
            **value,
            "expert_label_fraction": value["valid_expert_labels"]/max(value["candidates"], 1),
        }
        for key, value in sorted(split_counts.items())
    }
    route_report = sorted(
        (
            {
                "route": key,
                **value,
                "expert_label_fraction": value["valid_expert_labels"]/max(value["candidates"], 1),
            }
            for key, value in route_counts.items()
        ),
        key=lambda value: value["expert_label_fraction"],
        reverse=True,
    )[:5]
    report = {
        "segments": segments,
        "manifest_segments": len(rows),
        "segment_coverage": segments/max(len(rows), 1),
        "candidates": total_candidates,
        "valid_expert_labels": total_valid,
        "expert_label_fraction": total_valid/max(total_candidates, 1),
        "by_split": split_report,
        "highest_intervention_routes": route_report,
        "human_boundary_violation_on_corrected_subset": human_violations/max(total_valid, 1),
        "expert_boundary_violation": expert_violations/max(total_valid, 1),
        "correction_cause_fraction_on_corrected_subset": {
            key: value/max(total_valid, 1) for key, value in cause_counts.items()
        },
        "mean_correction_l2": float(correction_values.mean()) if len(correction_values) else 0.0,
        "p95_correction_l2": float(np.percentile(correction_values, 95)) if len(correction_values) else 0.0,
        "expert_max_lateral_accel_p95": float(np.percentile(expert_acceleration, 95)) if expert_acceleration else 0.0,
        "expert_max_lateral_jerk_p95": float(np.percentile(expert_jerk, 95)) if expert_jerk else 0.0,
    }
    print(json.dumps(report, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--processed", type=Path, default=Path("data/processed"))
    generate.add_argument("--output", type=Path, default=Path("data/teacher_frenet_expert"))
    generate.add_argument("--split", choices=("all", "train", "validation", "test"), default="all")
    generate.add_argument("--limit-segments", type=int)
    generate.add_argument("--lane-margin", type=float, default=0.20)
    generate.add_argument("--minimum-lane-confidence", type=float, default=0.50)
    generate.add_argument("--max-lateral-acceleration", type=float, default=2.5)
    generate.add_argument("--max-lateral-jerk", type=float, default=5.0)
    generate.add_argument("--overwrite", action="store_true")
    generate.set_defaults(function=command_generate)
    audit = commands.add_parser("audit")
    audit.add_argument("--processed", type=Path, default=Path("data/processed"))
    audit.add_argument("--teacher-cache", type=Path, default=Path("data/teacher_frenet_expert"))
    audit.add_argument("--output-json", type=Path, default=Path("artifacts/frenet_expert_audit.json"))
    audit.set_defaults(function=command_audit)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)

