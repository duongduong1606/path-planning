"""Build deterministic per-segment HDF5 files and a manifest."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from lane_pipeline import CameraCalibration, GroundProjector
from pipeline_config import FEATURE_NAMES, TARGET_NAMES, PipelineConfig
from prepare_lanes import segments, unpack_observations
from preprocessing import (
    assemble_features, build_ego_truth, complete_lane_series,
    interpolate_continuous, load_array, window_mask,
)


def route_split(all_segments: list[Path], config: PipelineConfig) -> dict[Path, str]:
    """Deterministic chunk-stratified ~70/14/16 split with zero route leakage."""
    routes_by_chunk: dict[str, list[Path]] = {}
    for segment in all_segments:
        routes_by_chunk.setdefault(segment.parents[1].name, []).append(segment.parent)
    routes_by_chunk = {key: sorted(set(value)) for key, value in routes_by_chunk.items()}
    if not routes_by_chunk:
        raise ValueError("no comma2k19 routes found")
    rng = np.random.default_rng(config.seed)
    shuffled = {}
    for chunk, routes in sorted(routes_by_chunk.items()):
        order = rng.permutation(len(routes))
        shuffled[chunk] = [routes[i] for i in order]
    allocations = {}
    for chunk, routes in shuffled.items():
        n = len(routes)
        if n < 3:
            raise ValueError(f"need at least 3 routes in {chunk} for split, got {n}")
        val_n = max(1, round(n * 5 / 37))
        test_n = max(1, round(n * 6 / 37))
        train_n = n - val_n - test_n
        if train_n < 1:
            raise ValueError(f"insufficient routes in {chunk} for split: {n}")
        allocations[chunk] = (train_n, val_n, test_n)
    result: dict[Path, str] = {}
    for chunk, routes in shuffled.items():
        train_n, val_n, _ = allocations[chunk]
        for i, route in enumerate(routes):
            result[route] = "train" if i < train_n else "validation" if i < train_n + val_n else "test"
    return result


def signal(segment: Path, group: str) -> tuple[np.ndarray, np.ndarray]:
    base = segment / "processed_log" / group
    return load_array(base / "t"), load_array(base / "value")


def synchronized_signals(segment: Path, times: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    result = {}
    coverage = np.ones(len(times), dtype=bool)
    for name, group in {
        "speed": "CAN/speed", "steering": "CAN/steering_angle",
        "gyro": "IMU/gyro", "accelerometer": "IMU/accelerometer",
    }.items():
        source_t, source_v = signal(segment, group)
        result[name], covered = interpolate_continuous(times, source_t, source_v)
        coverage &= covered
    return result, coverage


def load_control_source(segment: Path, times: np.ndarray) -> tuple[np.ndarray, str]:
    """Load optional decoded engagement cache; unknown is never called manual."""
    path = segment / "controls_enabled.npz"
    if not path.exists():
        return np.full(len(times), -1, np.int8), "unknown"
    with np.load(path, allow_pickle=False) as data:
        source_t, enabled = data["t"], data["enabled"].astype(bool)
    indices = np.searchsorted(source_t, times, side="right") - 1
    valid = (indices >= 0) & (times <= source_t[-1])
    output = np.full(len(times), -1, np.int8)
    output[valid] = enabled[indices[valid]].astype(np.int8)
    return output, "decoded_controlsState.enabled"


def write_hdf5(path: Path, arrays: dict[str, np.ndarray], attributes: dict) -> None:
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("h5py is required: python -m pip install -r requirements.txt") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".h5.tmp")
    with h5py.File(temporary, "w") as file:
        for name, value in arrays.items():
            file.create_dataset(name, data=value, compression="gzip", shuffle=True)
        for name, value in attributes.items():
            file.attrs[name] = value
    temporary.replace(path)


def compute_normalization(processed_root: Path, manifest_rows: list[dict]) -> dict:
    import h5py
    samples = []
    for row in manifest_rows:
        if row["split"] != "train":
            continue
        with h5py.File(processed_root / row["hdf5"], "r") as file:
            values = file["features"][:]
            valid = file["frame_valid"][:].astype(bool)
            samples.append(values[valid])
    values = np.concatenate(samples)
    low, high = np.nanquantile(values, [0.005, 0.995], axis=0)
    clipped = np.clip(values, low, high)
    mean, std = np.nanmean(clipped, axis=0), np.nanstd(clipped, axis=0)
    std = np.maximum(std, 1e-6)
    return {"feature_names": FEATURE_NAMES, "clip_low": low.tolist(), "clip_high": high.tolist(),
            "mean": mean.tolist(), "std": std.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/comma2k19/extracted"))
    parser.add_argument("--lane-cache", type=Path, default=Path("data/lane_cache_polarrcnn_ft_e15"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = PipelineConfig()
    all_segments = segments(args.data_root)
    splits = route_split(all_segments, config)
    if args.limit is not None:
        all_segments = all_segments[:args.limit]
    rows = []

    for number, segment in enumerate(all_segments, 1):
        relative = segment.relative_to(args.data_root)
        times = load_array(segment / "global_pose/frame_times")
        positions = load_array(segment / "global_pose/frame_positions")
        orientations = load_array(segment / "global_pose/frame_orientations")
        signals, signal_coverage = synchronized_signals(segment, times)
        observations = unpack_observations((args.lane_cache / relative).with_suffix(".npz"))
        with np.load((args.lane_cache / relative).with_suffix(".npz"), allow_pickle=False) as lane_data:
            brightness = lane_data["brightness"] if "brightness" in lane_data else np.full(len(times), np.nan)
        calibration_data = json.loads((args.lane_cache / relative.parent / "calibration.json").read_text())
        projector = GroundProjector(CameraCalibration(**calibration_data), config)
        left = np.full((len(times), 4), np.nan, np.float32)
        right = np.full_like(left, np.nan)
        left_conf = np.zeros(len(times), np.float32)
        right_conf = np.zeros(len(times), np.float32)
        for i, observation in enumerate(observations[:len(times)]):
            l_fit, r_fit = projector.fit_cubic(observation.left), projector.fit_cubic(observation.right)
            if l_fit is not None:
                left[i], left_conf[i] = l_fit, observation.left_confidence
            if r_fit is not None:
                right[i], right_conf[i] = r_fit, observation.right_confidence
        lanes = complete_lane_series(left, right, left_conf, right_conf, config)
        features = assemble_features(lanes, signals["speed"], signals["gyro"],
                                     signals["accelerometer"], signals["steering"], times)
        target, target_valid, future_y = build_ego_truth(times, positions, orientations, config)
        control_source, control_note = load_control_source(segment, times)
        manual = control_source == 0
        finite = np.isfinite(features).all(1)
        frame_valid = signal_coverage & finite & (lanes.validity > 0)
        all_windows, manual_windows = window_mask(frame_valid, target_valid, manual, config)
        split = splits[segment.parent]
        stride = config.train_stride if split == "train" else config.eval_stride
        anchors = np.flatnonzero(all_windows)[::stride]
        manual_anchors = np.flatnonzero(manual_windows)[::stride]
        output_relative = relative.with_suffix(".h5")
        write_hdf5(args.output / output_relative, {
            "timestamps": times, "features": features, "target": target,
            "target_valid": target_valid, "future_y": future_y,
            "frame_valid": frame_valid, "lane_observed": lanes.observed,
            "brightness": brightness,
            "control_source": control_source, "window_anchors": anchors,
            "manual_window_anchors": manual_anchors,
        }, {
            "route": relative.parent.as_posix(), "segment": relative.name, "chunk": relative.parts[0],
            "split": split, "feature_names": json.dumps(FEATURE_NAMES),
            "target_names": json.dumps(TARGET_NAMES), "control_note": control_note,
            "calibration": json.dumps(calibration_data), "sample_rate_hz": config.sample_rate_hz,
            "history_steps": config.history_steps, "horizon_steps": config.horizon_steps,
        })
        rows.append({"hdf5": output_relative.as_posix(), "chunk": relative.parts[0],
                     "route": relative.parent.name, "segment": relative.name, "split": split,
                     "frames": len(times), "valid_windows": len(anchors),
                     "manual_windows": len(manual_anchors),
                     "timestamp_coverage": float(signal_coverage.mean()),
                     "target_valid_fraction": float(target_valid[:-config.horizon_steps].mean()),
                     "both_lane_fraction": float(lanes.observed.all(1).mean()),
                     "control_note": control_note})
        print(f"[{number}/{len(all_segments)}] {relative}: {len(anchors)} windows")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    (args.output / "pipeline_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    if args.limit is None:
        normalization = compute_normalization(args.output, rows)
        (args.output / "normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

