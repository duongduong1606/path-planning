"""Lane detector adapters, flat-road IPM and metric polynomial fitting."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Protocol

import cv2
import numpy as np

from pipeline_config import PipelineConfig


@dataclass
class PixelLanes:
    left: np.ndarray
    right: np.ndarray
    left_confidence: float
    right_confidence: float


class LaneDetector(Protocol):
    def detect(self, frame: np.ndarray) -> PixelLanes: ...


class UFLDv2OnnxDetector:
    """ONNX Runtime adapter for official UFLD-v2 four-head models.

    The model must expose loc_row, loc_col, exist_row and exist_col outputs.
    TuSimple's two ego boundaries are row lanes 1 and 2. Confidence is the
    mean probability of the selected existence class.
    """

    def __init__(
        self,
        model_path: Path,
        original_size: tuple[int, int] = (1164, 874),
        input_size: tuple[int, int] = (800, 320),
        crop_ratio: float = 0.8,
        num_row: int = 56,
        providers: list[str] | None = None,
    ) -> None:
        try:
            # Importing PyTorch first makes its bundled CUDA/cuDNN DLLs
            # available to ONNX Runtime on Windows.
            import torch  # noqa: F401
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("Install onnxruntime-gpu (or onnxruntime) for UFLD-v2") from error
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        self.session = ort.InferenceSession(
            str(model_path), providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.original_width, self.original_height = original_size
        self.input_width, self.input_height = input_size
        self.crop_ratio = crop_ratio
        # Official TuSimple config: 56 anchors from y=160..710 in a 720 px
        # source image (different from CULane's 0.42..1.0 anchors).
        self.row_anchor = np.linspace(160.0 / 720.0, 710.0 / 720.0, num_row)

    @staticmethod
    def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
        shifted = values - np.max(values, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def _decode_row_lane(
        self, locations: np.ndarray, existence: np.ndarray, lane_index: int
    ) -> tuple[np.ndarray, float]:
        # Expected shapes: [1, grid, anchors, lanes], [1, 2, anchors, lanes].
        logits = locations[0, :, :, lane_index]
        exist_prob = self._softmax(existence[0, :, :, lane_index], axis=0)[1]
        valid = exist_prob >= 0.5
        if valid.sum() <= len(valid) / 2:
            return np.empty((0, 2), np.float32), 0.0
        maxima = np.argmax(logits, axis=0)
        expected = np.zeros(logits.shape[1], np.float32)
        for anchor, maximum in enumerate(maxima):
            indices = np.arange(max(0, maximum-1), min(logits.shape[0]-1, maximum+1)+1)
            probabilities = self._softmax(logits[indices, anchor], axis=0)
            expected[anchor] = float(np.sum(probabilities*indices) + 0.5)
        u = expected / max(logits.shape[0] - 1, 1) * self.original_width
        anchors = self.row_anchor
        if len(anchors) != len(u):
            anchors = np.linspace(160.0 / 720.0, 710.0 / 720.0, len(u))
        v = anchors * self.original_height
        points = np.column_stack((u[valid], v[valid])).astype(np.float32)
        confidence = float(np.mean(exist_prob[valid])) if valid.any() else 0.0
        return points, confidence

    def detect(self, frame: np.ndarray) -> PixelLanes:
        resize_height = round(self.input_height / self.crop_ratio)
        image = cv2.resize(frame, (self.input_width, resize_height))
        image = image[-self.input_height:, :, :]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        tensor = np.transpose(image, (2, 0, 1))[None]
        raw = self.session.run(None, {self.input_name: tensor})
        names = [output.name for output in self.session.get_outputs()]
        outputs = dict(zip(names, raw))
        required = {"loc_row", "exist_row"}
        if not required.issubset(outputs):
            raise ValueError(f"UFLD-v2 ONNX outputs must include {sorted(required)}; got {names}")
        left, left_conf = self._decode_row_lane(outputs["loc_row"], outputs["exist_row"], 1)
        right, right_conf = self._decode_row_lane(outputs["loc_row"], outputs["exist_row"], 2)
        return PixelLanes(left, right, left_conf, right_conf)


class PolarRCNNCULaneDetector:
    """Native PyTorch adapter for Polar R-CNN ResNet18 trained on CULane.

    Polar R-CNN returns all visible lane markings.  This adapter maps its
    normalized coordinates back to the source frame and chooses the nearest
    marking on each side of the camera centre as the ego-lane boundaries.
    """

    def __init__(
        self,
        model_path: Path,
        repository: Path = Path("third_party/PolarRCNN"),
        device: str = "cuda",
        confidence_threshold: float = 0.30,
        reference_y_ratio: float = 0.65,
        max_extrapolation_px: float = 80.0,
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Install PyTorch to use Polar R-CNN") from error

        repository = repository.resolve()
        if not repository.is_dir():
            raise FileNotFoundError(f"Polar R-CNN repository not found: {repository}")
        if str(repository) not in sys.path:
            sys.path.insert(0, str(repository))
        cfg = importlib.import_module("Config.polarrcnn_culane_r18")
        # The complete released checkpoint includes the backbone.  Disabling
        # this prevents an unnecessary ResNet download during construction.
        cfg.pretrained = False
        from Models.build import build_model

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Polar R-CNN requested CUDA, but torch.cuda.is_available() is False")
        self.torch = torch
        self.device = torch.device(device)
        self.model = build_model(cfg)
        state = torch.load(str(model_path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.model.roi_head.conf_thres = float(confidence_threshold)
        self.input_width = int(cfg.img_w)
        self.input_height = int(cfg.img_h)
        self.crop_ratio = float(cfg.cut_height) / float(cfg.ori_img_h)
        self.reference_y_ratio = float(reference_y_ratio)
        self.max_extrapolation_px = float(max_extrapolation_px)

    @staticmethod
    def _position_at_reference(
        points: np.ndarray, reference_y: float, max_extrapolation_px: float = 30.0
    ) -> float | None:
        if len(points) < 2:
            return None
        order = np.argsort(points[:, 1])
        y = points[order, 1]
        x = points[order, 0]
        y, unique = np.unique(y, return_index=True)
        x = x[unique]
        if (len(y) < 2 or reference_y < y[0] - max_extrapolation_px or
                reference_y > y[-1] + max_extrapolation_px):
            return None
        if reference_y < y[0]:
            return float(x[0] + (reference_y-y[0])*(x[1]-x[0])/(y[1]-y[0]))
        if reference_y > y[-1]:
            return float(x[-1] + (reference_y-y[-1])*(x[-1]-x[-2])/(y[-1]-y[-2]))
        return float(np.interp(reference_y, y, x))

    def detect_candidates(self, frame: np.ndarray) -> list[tuple[np.ndarray, float]]:
        height, width = frame.shape[:2]
        cut = int(round(height * self.crop_ratio))
        image = cv2.resize(
            cv2.cvtColor(frame[cut:], cv2.COLOR_BGR2RGB),
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_CUBIC,
        )
        tensor = self.torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
        with self.torch.inference_mode():
            lanes = self.model(tensor.unsqueeze(0).to(self.device))["lane_list"][0]
        candidates = []
        for lane in lanes:
            normalized = np.asarray(lane["points"], np.float32).reshape(-1, 2)
            points = normalized * np.asarray([width, height], np.float32)
            if len(points) >= 2:
                candidates.append((points, float(lane.get("conf", 1.0))))
        return candidates

    @staticmethod
    def _expected_lane_width_px(reference_y: float, width: int, height: int) -> float:
        """Approximate 3.7 m projected width using the dataset median camera pose."""
        scale_x = width / 1164.0
        scale_y = height / 874.0
        fx, fy, cy = 910.0*scale_x, 910.0*scale_y, 437.0*scale_y
        pitch, camera_height = np.deg2rad(2.0), 1.25
        image_y = (reference_y-cy)/fy
        ray_x = np.cos(pitch) + image_y*np.sin(pitch)
        ray_z = -np.sin(pitch) - image_y*np.cos(pitch)
        if ray_z >= -1e-6:
            return 0.45*width
        depth = (-camera_height/ray_z)*ray_x
        return float(np.clip(3.7*fx/max(depth, 1e-3), 0.08*width, 1.25*width))

    @classmethod
    def select_pair(
        cls,
        candidates: list[tuple[np.ndarray, float]],
        width: int,
        height: int,
        confidence_threshold: float = 0.30,
        reference_y_ratio: float = 0.65,
        max_extrapolation_px: float = 80.0,
    ) -> PixelLanes:
        reference_y = reference_y_ratio*height
        positioned = []
        for points, confidence in candidates:
            if confidence < confidence_threshold:
                continue
            position = cls._position_at_reference(points, reference_y, max_extrapolation_px)
            if position is not None and -0.25*width <= position <= 1.25*width:
                positioned.append((position, points, confidence))
        positioned.sort(key=lambda item: item[0])

        expected_width = cls._expected_lane_width_px(reference_y, width, height)
        best: tuple[float, tuple, tuple] | None = None
        # Adjacent markings define a physical lane.  Searching pairs rather
        # than forcing one marking on each side supports lane-change frames.
        for left, right in zip(positioned, positioned[1:]):
            lane_width = right[0]-left[0]
            if not (0.45*expected_width <= lane_width <= 1.8*expected_width):
                continue
            midpoint_error = abs((left[0]+right[0])/2.0-width/2.0)/width
            width_error = abs(np.log(lane_width/expected_width))
            confidence_cost = 1.0-(left[2]+right[2])/2.0
            score = 1.4*midpoint_error + 0.55*width_error + 0.25*confidence_cost
            if best is None or score < best[0]:
                best = (score, left, right)

        empty = np.empty((0, 2), np.float32)
        if best is None:
            if not positioned:
                return PixelLanes(empty, empty.copy(), 0.0, 0.0)
            # Preserve one trustworthy marking so downstream preprocessing can
            # infer the missing side with confidence zero and nominal width.
            single = min(
                positioned,
                key=lambda item: abs(item[0]-width/2.0)/width + 0.25*(1.0-item[2]),
            )
            if single[0] < width/2.0:
                return PixelLanes(single[1], empty.copy(), single[2], 0.0)
            return PixelLanes(empty, single[1], 0.0, single[2])
        _, left, right = best
        return PixelLanes(left[1], right[1], left[2], right[2])

    def detect(self, frame: np.ndarray) -> PixelLanes:
        height, width = frame.shape[:2]
        candidates = self.detect_candidates(frame)
        return self.select_pair(
            candidates, width, height, self.model.roi_head.conf_thres,
            self.reference_y_ratio, self.max_extrapolation_px,
        )


class ClassicalLaneDetector:
    """Hough baseline; useful for smoke tests, never represented as UFLD."""

    def detect(self, frame: np.ndarray) -> PixelLanes:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 140)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [np.array([(0, h), (.38*w, .52*h), (.62*w, .52*h), (w, h)], np.int32)], 255)
        lines = cv2.HoughLinesP(cv2.bitwise_and(edges, mask), 1, np.pi/180, 25, None, 25, 40)
        sides: list[list[tuple[float, float]]] = [[], []]
        if lines is not None:
            for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
                if abs(x2 - x1) < 2:
                    continue
                slope = (y2-y1)/(x2-x1)
                mid = (x1+x2)/2
                if slope < -0.25 and mid < w/2:
                    sides[0].extend([(x1, y1), (x2, y2)])
                elif slope > 0.25 and mid > w/2:
                    sides[1].extend([(x1, y1), (x2, y2)])
        arrays = [np.asarray(points, np.float32).reshape(-1, 2) for points in sides]
        confidence = [min(1.0, len(points)/20.0) for points in arrays]
        return PixelLanes(arrays[0], arrays[1], confidence[0], confidence[1])


@dataclass(frozen=True)
class CameraCalibration:
    pitch_rad: float
    height_m: float
    yaw_rad: float = 0.0
    roll_rad: float = 0.0


class GroundProjector:
    def __init__(self, calibration: CameraCalibration, config: PipelineConfig) -> None:
        self.calibration = calibration
        self.config = config

    def image_to_ground(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2), np.float32)
        u, v = np.asarray(points, np.float64).T
        image_x = (u - self.config.cx) / self.config.fx
        image_y = (v - self.config.cy) / self.config.fy
        pitch = self.calibration.pitch_rad
        ray_x = np.cos(pitch) + image_y * np.sin(pitch)
        ray_y = -image_x
        ray_z = -np.sin(pitch) - image_y * np.cos(pitch)
        scale = np.divide(-self.calibration.height_m, ray_z, out=np.full_like(ray_z, np.nan), where=ray_z < -1e-6)
        x, y = scale * ray_x, scale * ray_y
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0)
        return np.column_stack((x[valid], y[valid])).astype(np.float32)

    def fit_cubic(self, points: np.ndarray) -> np.ndarray | None:
        points = self.image_to_ground(points)
        keep = (points[:, 0] >= self.config.lane_min_x_m) & (points[:, 0] <= self.config.lane_max_x_m)
        points = points[keep]
        if len(points) < 4 or np.ptp(points[:, 0]) < 8.0:
            return None
        coefficients = np.polynomial.polynomial.polyfit(points[:, 0], points[:, 1], 3)
        residual = points[:, 1] - np.polynomial.polynomial.polyval(points[:, 0], coefficients)
        if np.median(np.abs(residual)) > 0.35:
            return None
        return coefficients.astype(np.float32)


def estimate_calibration(
    observations: list[PixelLanes], config: PipelineConfig
) -> CameraCalibration:
    """Robust route calibration from vanishing points and 3.7 m lane width."""
    vanishing_v: list[float] = []
    width_pixels: list[tuple[float, float]] = []
    for observation in observations:
        fitted = []
        for points in (observation.left, observation.right):
            if len(points) < 4:
                fitted.append(None)
            else:
                fitted.append(np.polyfit(points[:, 1], points[:, 0], 1))
        if fitted[0] is None or fitted[1] is None:
            continue
        a1, b1 = fitted[0]
        a2, b2 = fitted[1]
        if abs(a1-a2) > 1e-6:
            vanishing_v.append(float((b2-b1)/(a1-a2)))
        for v in (600.0, 700.0, 800.0):
            width_pixels.append((v, abs(np.polyval(fitted[1], v)-np.polyval(fitted[0], v))))
    if len(vanishing_v) < 10:
        raise ValueError("insufficient two-lane observations for route calibration")
    horizon_v = float(np.median(vanishing_v))
    pitch = float(np.arctan((config.cy-horizon_v)/config.fy))
    height_estimates = []
    for v, pixel_width in width_pixels:
        image_y = (v-config.cy)/config.fy
        ray_z = -np.sin(pitch) - image_y*np.cos(pitch)
        if ray_z < -1e-5 and pixel_width > 20:
            height_estimates.append(config.lane_width_m*(-ray_z)*config.fx/pixel_width)
    plausible = np.asarray(height_estimates)
    plausible = plausible[(plausible >= 1.0) & (plausible <= 1.8)]
    if len(plausible) < 10:
        raise ValueError("lane-width calibration did not yield a plausible camera height")
    return CameraCalibration(pitch, float(np.median(plausible)))

