"""Create and score a reproducible 300-frame manual lane audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from build_dataset import route_split
from lane_pipeline import CameraCalibration, GroundProjector
from pipeline_config import PipelineConfig
from prepare_lanes import segments, unpack_observations


def sample(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(2026)
    candidates = []
    for segment in segments(args.data_root):
        relative = segment.relative_to(args.data_root)
        cache = (args.lane_cache / relative).with_suffix(".npz")
        if not cache.exists():
            continue
        observations = unpack_observations(cache)
        for index, observation in enumerate(observations):
            if index % 10 == 0:
                confidence = min(observation.left_confidence, observation.right_confidence)
                candidates.append((segment, relative, index, confidence))
    if len(candidates) < args.count:
        raise ValueError(f"only {len(candidates)} candidate frames")
    # Equal sampling over confidence terciles; route diversity comes from round-robin sorting.
    candidates.sort(key=lambda item: item[3])
    bins = np.array_split(np.arange(len(candidates)), 3)
    selected = []
    for bin_indices in bins:
        selected.extend(rng.choice(bin_indices, args.count//3, replace=False))
    selected = selected[:args.count]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    grouped = {}
    for selected_index in selected:
        item = candidates[int(selected_index)]
        grouped.setdefault(item[0], []).append(item)
    for segment, items in grouped.items():
        wanted = {item[2]: item for item in items}
        capture = cv2.VideoCapture(str(segment / "video.hevc"))
        frame_index = 0
        while wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                _, relative, _, confidence = wanted.pop(frame_index)
                name = f"{relative.parts[0]}__{relative.parent.name}__{relative.name}__{frame_index:04d}.jpg"
                cv2.imwrite(str(args.output / name), frame)
                rows.append({"image": name, "segment": relative.as_posix(), "frame": frame_index,
                             "confidence": confidence, "left_points_json": "", "right_points_json": ""})
            frame_index += 1
        capture.release()
    with (args.output / "annotations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f"Created {len(rows)} audit images and annotations.csv")


def _curve_at_y(points: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.full(len(y_values), np.nan)
    order = np.argsort(points[:, 1])
    return np.interp(y_values, points[order, 1], points[order, 0], left=np.nan, right=np.nan)


def annotate(args: argparse.Namespace) -> None:
    """Interactive point annotator that checkpoints the CSV after every frame."""
    with args.annotations.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fields = rows[0].keys()
    audit_directory = args.annotations.parent
    window = "Lane audit: LEFT"

    def save() -> None:
        temporary = args.annotations.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        temporary.replace(args.annotations)

    pending = [i for i, row in enumerate(rows)
               if not row["left_points_json"] or not row["right_points_json"]]
    position = 0
    while position < len(pending):
        row_index = pending[position]
        row = rows[row_index]
        image = cv2.imread(str(audit_directory / row["image"]))
        if image is None:
            print(f"Skipping missing image: {row['image']}")
            position += 1
            continue
        points: dict[str, list[list[int]]] = {
            "left": json.loads(row["left_points_json"]) if row["left_points_json"] else [],
            "right": json.loads(row["right_points_json"]) if row["right_points_json"] else [],
        }
        side = "left"
        action = "stay"

        def mouse(event, x, y, flags, parameter):
            if event == cv2.EVENT_LBUTTONDOWN:
                points[side].append([x, y])

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 1164, 874)
        cv2.setMouseCallback(window, mouse)
        while True:
            canvas = image.copy()
            for name, color in (("left", (0, 255, 0)), ("right", (0, 0, 255))):
                ordered = sorted(points[name], key=lambda point: point[1])
                for point in ordered:
                    cv2.circle(canvas, point, 5, color, -1)
                if len(ordered) >= 2:
                    cv2.polylines(canvas, [np.asarray(ordered, np.int32)], False, color, 2)
            message = (f"{row_index + 1}/{len(rows)} pending {position + 1}/{len(pending)} | "
                       f"{side.upper()}: click >=4 | ENTER | U undo | R reset | B back | S skip | Q quit")
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (0, 0, 0), -1)
            cv2.putText(canvas, message, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("u"), 8):
                if points[side]: points[side].pop()
            elif key in (13, 10):
                if len(points[side]) < 4:
                    print(f"Need at least 4 {side} points")
                elif side == "left":
                    side = "right"
                else:
                    row["left_points_json"] = json.dumps(points["left"], separators=(",", ":"))
                    row["right_points_json"] = json.dumps(points["right"], separators=(",", ":"))
                    save()
                    action = "next"
                    break
            elif key == ord("r"):
                points = {"left": [], "right": []}
                side = "left"
                row["left_points_json"] = ""
                row["right_points_json"] = ""
            elif key == ord("b"):
                if position > 0:
                    action = "back"
                    break
                print("Already at the first pending frame")
            elif key == ord("s"):
                action = "next"
                break
            elif key in (ord("q"), 27):
                save(); cv2.destroyAllWindows()
                print("Annotation progress saved")
                return
        if action == "back":
            position -= 1
        else:
            position += 1
    cv2.destroyAllWindows()
    save()
    print("All audit frames annotated")


def evaluate(args: argparse.Namespace) -> None:
    config = PipelineConfig()
    errors = []
    true_points = predicted_points = matched_true = matched_predicted = 0
    both_detected = total = 0
    metric_errors = {10: [], 30: [], 50: []}
    calibration_cache: dict[Path, CameraCalibration] = {}
    with args.annotations.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if args.split != "all":
        split_by_route = route_split(segments(args.data_root), config)
        split_by_key = {f"{route.parents[0].name}/{route.name}": name
                        for route, name in split_by_route.items()}
        rows = [row for row in rows if split_by_key["/".join(Path(row["segment"]).parts[:2])]
                == args.split]
    grouped_rows: dict[Path, list[dict[str, str]]] = {}
    for row in rows:
        if row["left_points_json"] and row["right_points_json"]:
            grouped_rows.setdefault(Path(row["segment"]), []).append(row)
    for relative, segment_rows in grouped_rows.items():
        cache_path = (args.lane_cache / relative).with_suffix(".npz")
        observations = unpack_observations(cache_path)
        calibration_path = args.lane_cache / relative.parent / "calibration.json"
        if calibration_path not in calibration_cache:
            calibration_cache[calibration_path] = CameraCalibration(**json.loads(
                calibration_path.read_text()
            ))
        calibration = calibration_cache[calibration_path]
        projector = GroundProjector(calibration, config)
        for row in segment_rows:
            observation = observations[int(row["frame"])]
            truth_sides = [np.asarray(json.loads(row[key]), np.float32)
                           for key in ("left_points_json", "right_points_json")]
            predicted_sides = [observation.left, observation.right]
            total += 1
            both_detected += int(all(len(side) >= 2 for side in predicted_sides))
            for truth, predicted in zip(truth_sides, predicted_sides):
                recall_distances = np.abs(truth[:, 0] - _curve_at_y(predicted, truth[:, 1]))
                precision_distances = np.abs(predicted[:, 0] - _curve_at_y(truth, predicted[:, 1]))
                errors.extend(recall_distances[np.isfinite(recall_distances)])
                true_points += len(truth); predicted_points += len(predicted)
                matched_true += int(np.nansum(recall_distances <= 20))
                matched_predicted += int(np.nansum(precision_distances <= 20))
                truth_fit, predicted_fit = projector.fit_cubic(truth), projector.fit_cubic(predicted)
                if truth_fit is not None and predicted_fit is not None:
                    for x in metric_errors:
                        metric_errors[x].append(abs(np.polynomial.polynomial.polyval(x, truth_fit) -
                                                    np.polynomial.polynomial.polyval(x, predicted_fit)))
    if total == 0:
        raise ValueError("no completed annotations")
    precision = matched_predicted/max(predicted_points, 1)
    recall = matched_true/max(true_points, 1)
    report = {"annotated_frames": total, "both_lane_recall": both_detected/total,
              "f1_at_20px": 2*precision*recall/max(precision+recall, 1e-12),
              "median_pixel_error": float(np.median(errors)), "p95_pixel_error": float(np.percentile(errors, 95)),
              "metric_mae_m": {str(x): float(np.mean(values)) if values else None for x, values in metric_errors.items()}}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    make = subparsers.add_parser("sample")
    make.add_argument("--data-root", type=Path, default=Path("data/comma2k19/extracted"))
    make.add_argument("--lane-cache", type=Path, default=Path("data/lane_cache_polarrcnn_ft_e15"))
    make.add_argument("--output", type=Path, default=Path("data/lane_audit"))
    make.add_argument("--count", type=int, default=300); make.set_defaults(function=sample)
    label = subparsers.add_parser("annotate")
    label.add_argument("annotations", type=Path)
    label.set_defaults(function=annotate)
    score = subparsers.add_parser("evaluate")
    score.add_argument("annotations", type=Path)
    score.add_argument("--data-root", type=Path, default=Path("data/comma2k19/extracted"))
    score.add_argument("--split", choices=("all", "train", "validation", "test"), default="all")
    score.add_argument("--lane-cache", type=Path, default=Path("data/lane_cache_polarrcnn_ft_e15"))
    score.add_argument("--output", type=Path); score.set_defaults(function=evaluate)
    args = parser.parse_args(); args.function(args)


if __name__ == "__main__":
    main()

