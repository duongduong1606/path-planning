"""Convert the manual comma2k19 lane audit into a Polar R-CNN train set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from build_dataset import route_split
from prepare_lanes import segments
from pipeline_config import PipelineConfig


def densify(points: list[list[float]], count: int = 24) -> np.ndarray:
    values = np.asarray(points, np.float64)
    order = np.argsort(values[:, 1])
    y = values[order, 1]
    x = values[order, 0]
    y_dense = np.linspace(y[0], y[-1], count)
    x_dense = np.interp(y_dense, y, x)
    # CULane label files conventionally run from the image bottom upward;
    # Polar R-CNN reverses them when loading.
    return np.column_stack((x_dense, y_dense))[::-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=Path("data/lane_audit/annotations.csv"))
    parser.add_argument("--data-root", type=Path, default=Path("data/comma2k19/extracted"))
    parser.add_argument("--output", type=Path, default=Path("data/polarrcnn_finetune"))
    args = parser.parse_args()

    split_by_route = route_split(segments(args.data_root), PipelineConfig())
    split_by_key = {
        f"{route.parents[0].name}/{route.name}": split_name
        for route, split_name in split_by_route.items()
    }
    image_dir = args.output/"images"
    label_dir = args.output/"labels"
    list_dir = args.output/"list"
    for directory in (image_dir, label_dir, list_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, list[str]] = {name: [] for name in ("train", "validation", "test")}
    manifest_rows = []
    with args.annotations.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if not row["left_points_json"] or not row["right_points_json"]:
            continue
        relative = Path(row["segment"])
        route_key = f"{relative.parts[0]}/{relative.parts[1]}"
        split = split_by_key[route_key]
        source = args.annotations.parent/row["image"]
        frame = cv2.imread(str(source))
        if frame is None:
            raise FileNotFoundError(source)
        source_h, source_w = frame.shape[:2]
        destination_name = Path(row["image"]).stem + ".jpg"
        destination = image_dir/destination_name
        resized = cv2.resize(frame, (1640, 590), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(destination), resized, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"could not write {destination}")

        scale = np.asarray([1640/source_w, 590/source_h], np.float64)
        label_path = label_dir/(Path(destination_name).stem + ".lines.txt")
        lane_lines = []
        for key in ("left_points_json", "right_points_json"):
            lane = densify(json.loads(row[key]))*scale
            lane_lines.append(" ".join(f"{coordinate:.3f}" for coordinate in lane.reshape(-1)))
        label_path.write_text("\n".join(lane_lines) + "\n", encoding="utf-8")

        # Polar's CULane loader changes .jpg to .lines.txt in the same folder.
        colocated_label = image_dir/(Path(destination_name).stem + ".lines.txt")
        colocated_label.write_text(label_path.read_text(encoding="utf-8"), encoding="utf-8")
        list_line = f"/images/{destination_name} 1 1 0 0"
        manifests[split].append(list_line)
        manifest_rows.append({"image": destination_name, "segment": row["segment"],
                              "frame": row["frame"], "split": split})

    # The upstream loader has a fixed train_gt_new.txt path.
    (list_dir/"train_gt_new.txt").write_text("\n".join(manifests["train"]) + "\n", encoding="utf-8")
    for split, lines in manifests.items():
        (list_dir/f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (args.output/"manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image", "segment", "frame", "split"))
        writer.writeheader(); writer.writerows(manifest_rows)
    summary = {split: len(lines) for split, lines in manifests.items()}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
