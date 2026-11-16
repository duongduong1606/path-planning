"""Run lane perception once, cache pixel lanes, and calibrate each route."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from lane_pipeline import (
    CameraCalibration, ClassicalLaneDetector, PixelLanes, PolarRCNNCULaneDetector,
    UFLDv2OnnxDetector, estimate_calibration,
)
from pipeline_config import PipelineConfig
from preprocessing import load_array


def segments(root: Path) -> list[Path]:
    return sorted(path.parents[1] for path in root.rglob("global_pose/frame_times"))


def pack_observations(
    observations: list[PixelLanes], path: Path, backend: str,
    brightness: list[float] | None = None,
) -> None:
    def pack(side: str) -> tuple[np.ndarray, np.ndarray]:
        arrays = [getattr(item, side).astype(np.float32) for item in observations]
        offsets = np.cumsum([0] + [len(item) for item in arrays], dtype=np.int64)
        points = np.concatenate(arrays) if offsets[-1] else np.empty((0, 2), np.float32)
        return points, offsets
    left, left_offsets = pack("left")
    right, right_offsets = pack("right")
    np.savez_compressed(
        path, left_points=left, left_offsets=left_offsets,
        right_points=right, right_offsets=right_offsets,
        left_confidence=np.asarray([item.left_confidence for item in observations], np.float32),
        right_confidence=np.asarray([item.right_confidence for item in observations], np.float32),
        brightness=np.asarray(brightness if brightness is not None else
                              np.full(len(observations), np.nan), np.float32),
        backend=np.asarray(backend),
    )


def unpack_observations(path: Path, stride: int = 1) -> list[PixelLanes]:
    if stride < 1:
        raise ValueError("stride must be at least 1")
    with np.load(path, allow_pickle=False) as data:
        # NPZ members are compressed independently.  Materialize each member
        # once; indexing ``data[name]`` inside the loop would decompress the
        # complete array again for every frame.
        left_points = data["left_points"]
        left_offsets = data["left_offsets"]
        right_points = data["right_points"]
        right_offsets = data["right_offsets"]
        left_confidence = data["left_confidence"]
        right_confidence = data["right_confidence"]
        result = []
        for i in range(0, len(left_confidence), stride):
            ls = slice(left_offsets[i], left_offsets[i+1])
            rs = slice(right_offsets[i], right_offsets[i+1])
            result.append(PixelLanes(
                left_points[ls], right_points[rs],
                float(left_confidence[i]), float(right_confidence[i]),
            ))
        return result


def cached_backend(path: Path) -> str | None:
    with np.load(path, allow_pickle=False) as data:
        return str(data["backend"].item()) if "backend" in data else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/comma2k19/extracted"))
    parser.add_argument("--cache-root", type=Path,
                        help="default: data/lane_cache_<backend>")
    parser.add_argument("--backend", choices=("polarrcnn", "ufldv2", "classical"),
                        default="polarrcnn")
    parser.add_argument("--model", type=Path,
                        help="Polar R-CNN .pth or UFLD-v2 .onnx checkpoint")
    parser.add_argument("--polar-root", type=Path, default=Path("third_party/PolarRCNN"))
    parser.add_argument("--device", default="cuda", help="Polar R-CNN device (cuda or cpu)")
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--reference-y-ratio", type=float, default=0.65)
    parser.add_argument("--max-extrapolation-px", type=float, default=80.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-default-calibration", action="store_true",
                        help="smoke tests only: use 5 deg/1.5 m if visual calibration fails")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.cache_root is None:
        args.cache_root = (Path("data/lane_cache_polarrcnn_ft_e15") if args.backend == "polarrcnn"
                           else Path(f"data/lane_cache_{args.backend}"))
    config = PipelineConfig()
    if args.backend in {"polarrcnn", "ufldv2"}:
        if args.model is None or not args.model.is_file():
            parser.error(f"--model is required and must exist for the {args.backend} backend")
    if args.backend == "polarrcnn":
        detector = PolarRCNNCULaneDetector(
            args.model, args.polar_root, args.device, args.confidence_threshold,
            args.reference_y_ratio, args.max_extrapolation_px,
        )
    elif args.backend == "ufldv2":
        detector = UFLDv2OnnxDetector(args.model)
    else:
        detector = ClassicalLaneDetector()

    all_segments = segments(args.data_root)
    if args.limit is not None:
        all_segments = all_segments[:args.limit]
    route_cache: dict[Path, list[PixelLanes]] = {}
    for number, segment in enumerate(all_segments, 1):
        relative = segment.relative_to(args.data_root)
        cache_path = (args.cache_root / relative).with_suffix(".npz")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists() and not args.overwrite:
            actual_backend = cached_backend(cache_path)
            if actual_backend != args.backend:
                raise RuntimeError(
                    f"{cache_path} was produced by {actual_backend!r}, not {args.backend!r}; "
                    "choose another --cache-root or pass --overwrite"
                )
            observations = unpack_observations(cache_path, stride=20)
            calibration_observations = observations
        else:
            expected = len(load_array(segment / "global_pose/frame_times"))
            capture = cv2.VideoCapture(str(segment / "video.hevc"))
            observations = []
            brightness = []
            while len(observations) < expected:
                ok, frame = capture.read()
                if not ok:
                    break
                observations.append(detector.detect(frame))
                brightness.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
            capture.release()
            if len(observations) != expected:
                raise RuntimeError(f"{segment}: decoded {len(observations)}/{expected} frames")
            pack_observations(observations, cache_path, args.backend, brightness)
            calibration_observations = observations[::20]
        route_cache.setdefault(segment.parent, []).extend(calibration_observations)
        print(f"[{number}/{len(all_segments)}] {relative}")

    calibrated: dict[Path, CameraCalibration] = {}
    failures: dict[Path, str] = {}
    for route, observations in route_cache.items():
        try:
            calibration = estimate_calibration(observations, config)
            if not (0.0 <= calibration.pitch_rad <= np.deg2rad(10.0)):
                raise ValueError(f"implausible pitch {np.rad2deg(calibration.pitch_rad):.2f} deg")
            calibrated[route] = calibration
        except ValueError as error:
            failures[route] = str(error)

    if failures and calibrated:
        fallback = CameraCalibration(
            float(np.median([item.pitch_rad for item in calibrated.values()])),
            float(np.median([item.height_m for item in calibrated.values()])),
        )
        for route in failures:
            calibrated[route] = fallback
    elif failures and args.allow_default_calibration:
        fallback = CameraCalibration(np.deg2rad(5.0), 1.5)
        calibrated.update({route: fallback for route in failures})
    elif failures:
        details = "; ".join(f"{route.name}: {reason}" for route, reason in failures.items())
        raise ValueError(f"no valid route calibration available; {details}")

    for route, calibration in calibrated.items():
        relative = route.relative_to(args.data_root)
        destination = args.cache_root / relative / "calibration.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(calibration), indent=2), encoding="utf-8")
        metadata = {
            "method": "route_vanishing_point_lane_width" if route not in failures else "dataset_median_fallback",
            "fallback_reason": failures.get(route),
        }
        destination.with_name("calibration_meta.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    print(f"calibration: {len(calibrated)-len(failures)} route-specific, "
          f"{len(failures)} dataset-median fallback")


if __name__ == "__main__":
    main()

