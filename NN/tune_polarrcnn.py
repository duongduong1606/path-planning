"""Tune Polar R-CNN confidence and ego-lane pairing on the manual audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from build_dataset import route_split
from lane_audit import _curve_at_y
from lane_pipeline import PolarRCNNCULaneDetector
from pipeline_config import PipelineConfig
from prepare_lanes import segments


def score(rows: list[dict[str, str]], candidates: list[list[tuple[np.ndarray, float]]],
          threshold: float, reference: float, extrapolation: float) -> dict:
    errors: list[float] = []
    true_points = predicted_points = matched_true = matched_predicted = 0
    both = 0
    for row, frame_candidates in zip(rows, candidates):
        truth_sides = [np.asarray(json.loads(row[key]), np.float32)
                       for key in ("left_points_json", "right_points_json")]
        selected = PolarRCNNCULaneDetector.select_pair(
            frame_candidates, 1164, 874, threshold, reference, extrapolation
        )
        predicted_sides = (selected.left, selected.right)
        both += int(all(len(side) >= 2 for side in predicted_sides))
        for truth, predicted in zip(truth_sides, predicted_sides):
            recall_distance = np.abs(truth[:, 0]-_curve_at_y(predicted, truth[:, 1]))
            precision_distance = np.abs(predicted[:, 0]-_curve_at_y(truth, predicted[:, 1]))
            errors.extend(recall_distance[np.isfinite(recall_distance)].tolist())
            true_points += len(truth)
            predicted_points += len(predicted)
            matched_true += int(np.nansum(recall_distance <= 20))
            matched_predicted += int(np.nansum(precision_distance <= 20))
    precision = matched_predicted/max(predicted_points, 1)
    recall = matched_true/max(true_points, 1)
    return {
        "confidence_threshold": threshold,
        "reference_y_ratio": reference,
        "max_extrapolation_px": extrapolation,
        "both_lane_recall": both/len(rows),
        "f1_at_20px": 2*precision*recall/max(precision+recall, 1e-12),
        "median_pixel_error": float(np.median(errors)) if errors else None,
        "p95_pixel_error": float(np.percentile(errors, 95)) if errors else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path, default=Path("data/lane_audit/annotations.csv"), nargs="?")
    parser.add_argument("--model", type=Path, default=Path("models/polarrcnn_culane_r18.pth"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/polarrcnn_tuning.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/comma2k19/extracted"))
    parser.add_argument("--frame-source", choices=("video", "jpeg"), default="video")
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="all")
    args = parser.parse_args()

    with args.annotations.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["left_points_json"] and row["right_points_json"]]
    if args.split != "all":
        split_by_route = route_split(segments(args.data_root), PipelineConfig())
        split_by_key = {f"{route.parents[0].name}/{route.name}": name
                        for route, name in split_by_route.items()}
        rows = [row for row in rows if split_by_key["/".join(Path(row["segment"]).parts[:2])]
                == args.split]
    detector = PolarRCNNCULaneDetector(args.model, confidence_threshold=0.10)
    all_candidates: list[list[tuple[np.ndarray, float]] | None] = [None]*len(rows)
    if args.frame_source == "jpeg":
        for number, row in enumerate(rows, 1):
            frame = cv2.imread(str(args.annotations.parent/row["image"]))
            if frame is None:
                raise RuntimeError(f"could not read {args.annotations.parent/row['image']}")
            all_candidates[number-1] = detector.detect_candidates(frame)
            if number % 50 == 0 or number == len(rows):
                print(f"inference {number}/{len(rows)}")
    else:
        grouped: dict[str, list[tuple[int, dict[str, str]]]] = {}
        for index, row in enumerate(rows):
            grouped.setdefault(row["segment"], []).append((index, row))
        completed = 0
        for segment, items in grouped.items():
            wanted: dict[int, list[int]] = {}
            for index, row in items:
                wanted.setdefault(int(row["frame"]), []).append(index)
            capture = cv2.VideoCapture(str(args.data_root/segment/"video.hevc"))
            frame_index = 0
            last_frame = max(wanted)
            while frame_index <= last_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                for index in wanted.get(frame_index, []):
                    all_candidates[index] = detector.detect_candidates(frame)
                    completed += 1
                    if completed % 50 == 0 or completed == len(rows):
                        print(f"inference {completed}/{len(rows)}")
                frame_index += 1
            capture.release()
            missing = [index for index, _ in items if all_candidates[index] is None]
            if missing:
                raise RuntimeError(f"could not sequentially decode {segment}: {len(missing)} audit frames")

    candidates_complete = [item for item in all_candidates if item is not None]

    results = []
    for threshold in np.arange(0.10, 0.51, 0.05):
        for reference in (0.65, 0.70, 0.75, 0.80, 0.85):
            for extrapolation in (30.0, 80.0, 140.0):
                results.append(score(rows, candidates_complete, round(float(threshold), 2),
                                     reference, extrapolation))
    eligible = [item for item in results if item["median_pixel_error"] <= 10.0
                and item["p95_pixel_error"] <= 25.0]
    ranking = sorted(
        eligible,
        key=lambda item: (item["both_lane_recall"], item["f1_at_20px"],
                          -abs(item["confidence_threshold"]-0.30)),
        reverse=True,
    )
    recall_gate = [item for item in results if item["both_lane_recall"] >= 0.85]
    error_gate = [item for item in results if item["median_pixel_error"] <= 10.0
                  and item["p95_pixel_error"] <= 25.0]
    report = {
        "annotated_frames": len(rows),
        "frame_source": args.frame_source,
        "split": args.split,
        "fixed_default": next(
            item for item in results
            if item["confidence_threshold"] == 0.30
            and item["reference_y_ratio"] == 0.65
            and item["max_extrapolation_px"] == 80.0
        ),
        "best": ranking[0] if ranking else None,
        "top_10": ranking[:10],
        "lowest_p95_with_recall_gate": min(
            recall_gate, key=lambda item: item["p95_pixel_error"], default=None
        ),
        "highest_recall_with_error_gate": max(
            error_gate, key=lambda item: item["both_lane_recall"], default=None
        ),
        "evaluated_configurations": len(results),
        "all_results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "all_results"}, indent=2))


if __name__ == "__main__":
    main()
