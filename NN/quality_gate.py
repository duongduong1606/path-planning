"""Fail fast on data quality and route leakage before model training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pipeline_config import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--lane-audit", type=Path, help="JSON produced by lane_audit.py evaluate")
    args = parser.parse_args()
    config = PipelineConfig()
    with (args.processed / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    route_splits = {}
    for row in rows:
        route_splits.setdefault(row["route"], set()).add(row["split"])
    leakage = {route: list(splits) for route, splits in route_splits.items() if len(splits) != 1}
    frame_counts = [int(row["frames"]) for row in rows]
    target_counts = [max(frames - config.horizon_steps, 0) for frames in frame_counts]
    timestamp_coverage = sum(
        frames * float(row["timestamp_coverage"])
        for frames, row in zip(frame_counts, rows)
    ) / max(sum(frame_counts), 1)
    target_valid = sum(
        count * float(row["target_valid_fraction"])
        for count, row in zip(target_counts, rows)
    ) / max(sum(target_counts), 1)
    minimum_timestamp_coverage = min(float(row["timestamp_coverage"]) for row in rows)
    minimum_target_valid_fraction = min(float(row["target_valid_fraction"]) for row in rows)
    checks = {"no_route_leakage": not leakage, "timestamp_coverage_ge_99pct": timestamp_coverage >= .99,
              "target_valid_ge_95pct": target_valid >= .95}
    if args.lane_audit:
        audit = json.loads(args.lane_audit.read_text())
        checks.update({"both_lane_recall_ge_85pct": audit["both_lane_recall"] >= .85,
                       "median_pixel_error_le_10": audit["median_pixel_error"] <= 10,
                       "p95_pixel_error_le_25": audit["p95_pixel_error"] <= 25})
    report = {"pass": all(checks.values()), "checks": checks, "route_leakage": leakage,
              "timestamp_coverage": timestamp_coverage,
              "target_valid_fraction": target_valid,
              "minimum_segment_timestamp_coverage": minimum_timestamp_coverage,
              "minimum_segment_target_valid_fraction": minimum_target_valid_fraction}
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] else 2)


if __name__ == "__main__":
    main()

