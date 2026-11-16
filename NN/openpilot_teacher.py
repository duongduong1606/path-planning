"""Offline Openpilot teacher inference for comma2k19.

This module intentionally keeps teacher predictions separate from immutable
human future-pose truth.  It implements the camera warp/YUV packing used by
Openpilot's current big driving model and feeds every recurrent state output
back into the next frame.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import pickle
from pathlib import Path
import time

import cv2
import numpy as np


MODEL_WIDTH = 512
MODEL_HEIGHT = 256
MODEL_CY = 47.6
PLAN_STEPS = 33
PLAN_WIDTH = 15
DESIRE_WIDTH = 8
TIME_INDICES = 10.0 * (np.arange(PLAN_STEPS, dtype=np.float32) / 32.0) ** 2

SOURCE_INTRINSICS = np.array(
    [[910.0, 0.0, 582.0], [0.0, 910.0, 437.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
MEDMODEL_INTRINSICS = np.array(
    [[910.0, 0.0, 256.0], [0.0, 910.0, MODEL_CY], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
SBIGMODEL_INTRINSICS = np.array(
    [[455.0, 0.0, 256.0], [0.0, 455.0, 0.5 * (256.0 + MODEL_CY)], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
VIEW_FROM_DEVICE = np.array(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    result = np.exp(np.clip(shifted, -100.0, 0.0))
    return result / np.sum(result, axis=axis, keepdims=True)


def rotation_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Openpilot Z-Y-X Euler convention."""
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float64)
    return rz @ ry @ rx


def warp_matrix(
    source_intrinsics: np.ndarray,
    model_intrinsics: np.ndarray,
    calibration_rpy: np.ndarray,
) -> np.ndarray:
    """Return destination-model pixels -> source-camera pixels."""
    model_from_calib = model_intrinsics @ VIEW_FROM_DEVICE
    camera_from_calib = (
        source_intrinsics @ VIEW_FROM_DEVICE @ rotation_from_euler(*calibration_rpy)
    )
    return (camera_from_calib @ np.linalg.inv(model_from_calib)).astype(np.float32)


def _split_i420(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = frame_bgr.shape[:2]
    if height % 2 or width % 2:
        raise ValueError(f"I420 requires even dimensions, got {width}x{height}")
    flat = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    y_size = height * width
    uv_size = y_size // 4
    y = flat[:y_size].reshape(height, width)
    u = flat[y_size:y_size + uv_size].reshape(height // 2, width // 2)
    v = flat[y_size + uv_size:y_size + 2 * uv_size].reshape(height // 2, width // 2)
    return y, u, v


def _warp_plane(
    plane: np.ndarray, matrix_destination_to_source: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    # Openpilot's warp clips coordinates to the source border and samples the
    # nearest pixel. WARP_INVERSE_MAP makes the supplied matrix dst -> src.
    return cv2.warpPerspective(
        plane,
        matrix_destination_to_source,
        size,
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def frame_to_model_tensor(frame_bgr: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Openpilot-compatible 512x256 I420 warp packed as [6,128,256]."""
    y, u, v = _split_i420(frame_bgr)
    y_warped = _warp_plane(y, matrix, (MODEL_WIDTH, MODEL_HEIGHT))
    # This is the exact UV_SCALE @ M @ UV_SCALE_INV simplification used in
    # tinygrad/examples/openpilot/compile_warp.py.
    uv_scale = np.array(
        [[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [2.0, 2.0, 1.0]], np.float32
    )
    matrix_uv = matrix * uv_scale
    u_warped = _warp_plane(u, matrix_uv, (MODEL_WIDTH // 2, MODEL_HEIGHT // 2))
    v_warped = _warp_plane(v, matrix_uv, (MODEL_WIDTH // 2, MODEL_HEIGHT // 2))
    packed = np.concatenate(
        (
            y_warped[0::2, 0::2],
            y_warped[1::2, 0::2],
            y_warped[0::2, 1::2],
            y_warped[1::2, 1::2],
            u_warped,
            v_warped,
        ),
        axis=0,
    )
    return np.ascontiguousarray(packed.reshape(6, MODEL_HEIGHT // 2, MODEL_WIDTH // 2))


def scaled_source_intrinsics(width: int, height: int) -> np.ndarray:
    scale = np.diag([width / 1164.0, height / 874.0, 1.0])
    return scale @ SOURCE_INTRINSICS


@dataclass(frozen=True)
class TeacherFrame:
    plan: np.ndarray
    plan_std: np.ndarray
    target: np.ndarray
    target_std: np.ndarray
    desire_probabilities: np.ndarray
    action: np.ndarray
    inference_ms: float


class OpenpilotTeacher:
    """Stateful wrapper around the official big driving model ONNX."""

    STATE_PAIRS = {
        "state_img_q": "next_state_img_q",
        "state_desire_q": "next_state_desire_q",
        "state_feat_q": "next_state_feat_q",
    }

    def __init__(
        self,
        model_path: Path,
        calibration_rpy: np.ndarray,
        frame_size: tuple[int, int] = (1164, 874),
        provider: str = "cuda",
        horizon_seconds: float = 2.5,
        lateral_sign: float = 1.0,
    ) -> None:
        try:
            # Loading torch first exposes its CUDA/cuDNN DLLs to ORT on Windows.
            import torch  # noqa: F401
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("Install torch and onnxruntime-gpu for the teacher") from error
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        options = ort.SessionOptions()
        options.log_severity_level = 3
        requested = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if provider == "cuda" else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(str(model_path), options, providers=requested)
        if provider == "cuda" and self.session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDA provider unavailable: {self.session.get_providers()}")
        self.model_path = Path(model_path)
        self.horizon_seconds = float(horizon_seconds)
        self.lateral_sign = float(lateral_sign)
        self.configure_camera(calibration_rpy, frame_size)
        self.inputs: dict[str, np.ndarray] = {}
        for item in self.session.get_inputs():
            dtype = np.uint8 if item.type == "tensor(uint8)" else np.float32
            self.inputs[item.name] = np.zeros(tuple(item.shape), dtype=dtype)
        self.inputs["traffic_convention"][0, 0] = 1.0  # right-hand traffic / left-hand drive
        self.inputs["action_t"][0] = (0.10, 0.35)
        self.output_names = [item.name for item in self.session.get_outputs()]
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.output_slices: dict[str, slice] = pickle.loads(
            base64.b64decode(metadata["output_slices"])
        )
        self.model_checkpoint = metadata.get("model_checkpoint", "unknown")

    def configure_camera(
        self, calibration_rpy: np.ndarray, frame_size: tuple[int, int]
    ) -> None:
        """Update a route calibration without reloading the 769 MB model."""
        width, height = frame_size
        source_intrinsics = scaled_source_intrinsics(width, height)
        self.matrices = np.stack(
            [
                warp_matrix(source_intrinsics, MEDMODEL_INTRINSICS, calibration_rpy),
                warp_matrix(source_intrinsics, SBIGMODEL_INTRINSICS, calibration_rpy),
            ]
        )

    def reset(self) -> None:
        for name in self.STATE_PAIRS:
            self.inputs[name].fill(0)
        self.inputs["desire"].fill(0)

    def infer(self, frame_bgr: np.ndarray, desire: int = 0) -> TeacherFrame:
        self.inputs["new_img"] = np.stack(
            [frame_to_model_tensor(frame_bgr, matrix) for matrix in self.matrices]
        )
        self.inputs["desire"].fill(0)
        if desire:
            if not 0 <= desire < DESIRE_WIDTH:
                raise ValueError(f"desire must be in [0,{DESIRE_WIDTH}), got {desire}")
            self.inputs["desire"][desire] = 1.0
        started = time.perf_counter()
        values = self.session.run(self.output_names, self.inputs)
        inference_ms = (time.perf_counter() - started) * 1000.0
        outputs = dict(zip(self.output_names, values))
        for state_name, next_name in self.STATE_PAIRS.items():
            self.inputs[state_name] = outputs[next_name]
        raw = outputs["outputs"][0]
        plan_raw = raw[self.output_slices["plan"]].reshape(2, PLAN_STEPS, PLAN_WIDTH)
        plan = plan_raw[0].astype(np.float32)
        plan_std = np.exp(np.clip(plan_raw[1], -20.0, 11.0)).astype(np.float32)
        desire_probabilities = softmax(raw[self.output_slices["desire_state"]]).astype(np.float32)
        action_raw = raw[self.output_slices["action"]]
        action = np.concatenate((action_raw[:2], np.exp(np.clip(action_raw[2:], -20.0, 11.0))))
        target = np.array(
            [
                np.interp(self.horizon_seconds, TIME_INDICES, plan[:, 1]),
                np.interp(self.horizon_seconds, TIME_INDICES, plan[:, 4]),
                np.interp(self.horizon_seconds, TIME_INDICES, plan[:, 7]),
            ],
            np.float32,
        )
        target_std = np.array(
            [
                np.interp(self.horizon_seconds, TIME_INDICES, plan_std[:, 1]),
                np.interp(self.horizon_seconds, TIME_INDICES, plan_std[:, 4]),
                np.interp(self.horizon_seconds, TIME_INDICES, plan_std[:, 7]),
            ],
            np.float32,
        )
        target *= self.lateral_sign
        plan[:, [1, 4, 7, 10, 13]] *= self.lateral_sign
        return TeacherFrame(plan, plan_std, target, target_std, desire_probabilities,
                            action.astype(np.float32), inference_ms)

    def metadata(self) -> dict:
        return {
            "model_path": self.model_path.name,
            "model_sha256": sha256_file(self.model_path),
            "model_checkpoint": self.model_checkpoint,
            "providers": self.session.get_providers(),
            "horizon_seconds": self.horizon_seconds,
            "lateral_sign": self.lateral_sign,
            "time_indices": TIME_INDICES.tolist(),
            "desire_policy": "zero/no-request; executed lane changes are not synthesized",
            "preprocessing": "Openpilot narrow+wide virtual views, I420 nearest-neighbor warp",
        }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_teacher_cache(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays, metadata=np.array(json.dumps(metadata)))
    temporary.replace(path)

