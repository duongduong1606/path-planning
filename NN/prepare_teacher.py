"""Generate, audit and optionally attach Openpilot teacher predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from openpilot_teacher import OpenpilotTeacher, PLAN_STEPS, PLAN_WIDTH, save_teacher_cache
from prepare_lanes import segments
from preprocessing import load_array


DEFAULT_MODEL = Path("models/openpilot_big_driving_supercombo.onnx")


def cache_is_complete(path: Path, requested_frames: int | None) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as values:
            metadata = json.loads(str(values["metadata"]))
            decoded = int(metadata["decoded_frames"])
            expected = int(metadata["expected_frames"])
            stored = len(values["timestamps"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False
    required = min(expected, requested_frames) if requested_frames is not None else expected
    return decoded >= required and stored >= required


def calibration_for(segment: Path, data_root: Path, lane_cache: Path) -> tuple[np.ndarray, Path]:
    relative = segment.relative_to(data_root)
    path = lane_cache / relative.parent / "calibration.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    # Openpilot and lane_pipeline use the same positive pitch convention.
    return np.array(
        [values.get("roll_rad", 0.0), values["pitch_rad"], values.get("yaw_rad", 0.0)],
        np.float32,
    ), path


def generate_segment(
    args: argparse.Namespace, segment: Path, teacher: OpenpilotTeacher | None
) -> tuple[dict, OpenpilotTeacher | None]:
    relative = segment.relative_to(args.data_root)
    destination = (args.output / relative).with_suffix(".npz")
    if not args.overwrite and cache_is_complete(destination, args.max_frames):
        return {"segment": relative.as_posix(), "status": "cached"}, teacher
    timestamps = load_array(segment / "global_pose/frame_times")
    capture = cv2.VideoCapture(str(segment / "video.hevc"))
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"could not decode first frame: {segment / 'video.hevc'}")
    height, width = frame.shape[:2]
    calibration, calibration_path = calibration_for(
        segment, args.data_root, args.lane_cache
    )
    if teacher is None:
        teacher = OpenpilotTeacher(
            args.model, calibration, (width, height), args.provider,
            args.horizon_seconds, args.lateral_sign,
        )
    else:
        teacher.configure_camera(calibration, (width, height))
    teacher.reset()
    n = min(len(timestamps), args.max_frames or len(timestamps))
    plans = np.full((n, PLAN_STEPS, PLAN_WIDTH), np.nan, np.float32)
    plan_stds = np.full_like(plans, np.nan)
    targets = np.full((n, 3), np.nan, np.float32)
    target_stds = np.full_like(targets, np.nan)
    desire_probabilities = np.full((n, 8), np.nan, np.float32)
    actions = np.full((n, 4), np.nan, np.float32)
    inference_ms = np.full(n, np.nan, np.float32)
    decoded = 0
    for index in range(n):
        if index:
            ok, frame = capture.read()
        if not ok:
            break
        prediction = teacher.infer(frame)
        plans[index] = prediction.plan
        plan_stds[index] = prediction.plan_std
        targets[index] = prediction.target
        target_stds[index] = prediction.target_std
        desire_probabilities[index] = prediction.desire_probabilities
        actions[index] = prediction.action
        inference_ms[index] = prediction.inference_ms
        decoded += 1
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"  {relative}: {index + 1}/{n}, {np.nanmedian(inference_ms[:index+1]):.1f} ms/frame")
    capture.release()
    warmup = min(args.warmup_frames, decoded)
    valid = np.zeros(n, bool)
    valid[warmup:decoded] = (
        np.isfinite(targets[warmup:decoded]).all(1)
        & np.isfinite(target_stds[warmup:decoded]).all(1)
        & (target_stds[warmup:decoded, 0] > 0)
    )
    metadata = teacher.metadata() | {
        "segment": relative.as_posix(),
        "calibration_path": calibration_path.relative_to(args.lane_cache).as_posix(),
        "calibration_rpy": calibration.tolist(),
        "source_frame_size": [width, height],
        "decoded_frames": decoded,
        "expected_frames": len(timestamps),
        "warmup_frames": args.warmup_frames,
    }
    save_teacher_cache(
        destination,
        {
            "timestamps": timestamps[:n], "plan": plans, "plan_std": plan_stds,
            "target": targets, "target_std": target_stds,
            "desire_probabilities": desire_probabilities, "action": actions,
            "valid": valid, "inference_ms": inference_ms,
        },
        metadata,
    )
    return {
        "segment": relative.as_posix(), "status": "generated", "frames": decoded,
        "valid": int(valid.sum()), "median_inference_ms": float(np.nanmedian(inference_ms)),
        "cache": destination.as_posix(),
    }, teacher


def command_generate(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise SystemExit(f"teacher model not found: {args.model}")
    selected = segments(args.data_root)
    if args.segment:
        needle = args.segment.replace("\\", "/")
        selected = [item for item in selected if needle in item.as_posix()]
    if args.limit_segments is not None:
        selected = selected[:args.limit_segments]
    if not selected:
        raise SystemExit("no matching segments")
    results = []
    teacher = None
    for index, segment in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {segment.relative_to(args.data_root)}")
        result, teacher = generate_segment(args, segment, teacher)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))
    generated = [item for item in results if item["status"] == "generated"]
    summary = {
        "segments": len(results), "generated": len(generated),
        "median_inference_ms": float(np.median([x["median_inference_ms"] for x in generated])) if generated else None,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def command_audit(args: argparse.Namespace) -> None:
    import csv
    import h5py

    with (args.processed / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    direct, mirrored, stds, inference = [], [], [], []
    used = 0
    total_truth_frames = covered_truth_frames = 0
    for row in rows:
        cache = (args.teacher_cache / Path(row["hdf5"])).with_suffix(".npz")
        with h5py.File(args.processed / row["hdf5"], "r") as truth:
            total_truth_frames += int(truth["target_valid"][:].astype(bool).sum())
        if not cache.exists():
            continue
        with np.load(cache, allow_pickle=False) as teacher, h5py.File(args.processed / row["hdf5"], "r") as truth:
            n = min(len(teacher["target"]), len(truth["target"]))
            valid = teacher["valid"][:n].astype(bool) & truth["target_valid"][:n].astype(bool)
            if not valid.any():
                continue
            covered_truth_frames += int(valid.sum())
            prediction = teacher["target"][:n][valid, 0]
            target = truth["target"][:n][valid, 0]
            direct.append(np.abs(prediction - target))
            mirrored.append(np.abs(-prediction - target))
            stds.append(teacher["target_std"][:n][valid, 0])
            inference.append(teacher["inference_ms"][:n][valid])
            used += 1
    if not direct:
        raise SystemExit("no overlapping teacher caches and processed truth")
    direct_values, mirrored_values = np.concatenate(direct), np.concatenate(mirrored)
    result = {
        "segments": used, "manifest_segments": len(rows), "frames": len(direct_values),
        "teacher_segment_coverage": used / max(len(rows), 1),
        "teacher_valid_frame_coverage": covered_truth_frames / max(total_truth_frames, 1),
        "direct_y_mae_m": float(np.mean(direct_values)),
        "mirrored_y_mae_m": float(np.mean(mirrored_values)),
        "recommended_lateral_sign": 1 if np.mean(direct_values) <= np.mean(mirrored_values) else -1,
        "median_teacher_y_std_m": float(np.median(np.concatenate(stds))),
        "median_inference_ms": float(np.median(np.concatenate(inference))),
    }
    print(json.dumps(result, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="run the recurrent teacher over video")
    generate.add_argument("--data-root", type=Path, default=Path("data/comma2k19/extracted"))
    generate.add_argument("--lane-cache", type=Path, default=Path("data/lane_cache_polarrcnn_ft_e15"))
    generate.add_argument("--output", type=Path, default=Path("data/teacher_openpilot"))
    generate.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    generate.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    generate.add_argument("--horizon-seconds", type=float, default=2.5)
    generate.add_argument("--lateral-sign", type=float, choices=(-1.0, 1.0), default=-1.0,
                          help="-1 maps Openpilot right-positive y to project left-positive y")
    generate.add_argument("--warmup-frames", type=int, default=40)
    generate.add_argument("--max-frames", type=int)
    generate.add_argument("--limit-segments", type=int)
    generate.add_argument("--segment", help="substring of route/segment path")
    generate.add_argument("--progress-every", type=int, default=100)
    generate.add_argument("--overwrite", action="store_true")
    generate.set_defaults(function=command_generate)

    audit = commands.add_parser("audit", help="compare teacher lateral sign/scale with pose truth")
    audit.add_argument("--processed", type=Path, default=Path("data/processed"))
    audit.add_argument("--teacher-cache", type=Path, default=Path("data/teacher_openpilot"))
    audit.add_argument("--output-json", type=Path, default=Path("artifacts/openpilot_teacher_audit.json"))
    audit.set_defaults(function=command_audit)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)

