"""Go/no-go validation for the bundled comma2k19 one-minute sample."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent / "comma2k19_data"
EXPECTED_FRAMES = 1200
CAMERA_RATE_HZ = 20.0
FUTURE_SECONDS = 2.5


def load(relative: str) -> np.ndarray:
    return np.load(ROOT / relative)


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Hamilton quaternion [w, x, y, z] to rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (w * y + x * z)],
            [2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (w * x + y * z), w * w - x * x - y * y + z * z],
        ],
        dtype=np.float64,
    )


def frequency_hz(timestamps: np.ndarray) -> float:
    return float(1.0 / np.median(np.diff(timestamps)))


def interpolate_signal(
    frame_times: np.ndarray, signal_times: np.ndarray, values: np.ndarray
) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return np.interp(frame_times, signal_times, values)
    return np.column_stack(
        [np.interp(frame_times, signal_times, values[:, column]) for column in range(values.shape[1])]
    )


def extract_lane_polynomials(frame: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return left/right cubic x(y) polynomials in normalized image coordinates."""
    height, width = frame.shape[:2]
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    white = cv2.inRange(hls, np.array([0, 145, 0]), np.array([255, 255, 150]))
    yellow = cv2.inRange(hsv, np.array([12, 55, 80]), np.array([42, 255, 255]))
    color_mask = cv2.bitwise_or(white, yellow)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 35, 110)
    # Keep all geometric edges in the road ROI. The sample contains worn and
    # shadowed white markings that a strict color mask removes. The color mask
    # is still used to reinforce likely lane pixels.
    reinforced = cv2.bitwise_and(edges, cv2.dilate(color_mask, np.ones((7, 7), np.uint8)))
    edges = cv2.bitwise_or(edges, reinforced)

    roi = np.zeros_like(edges)
    polygon = np.array(
        [[
            (int(0.05 * width), int(0.82 * height)),
            (int(0.37 * width), int(0.50 * height)),
            (int(0.63 * width), int(0.50 * height)),
            (int(0.95 * width), int(0.82 * height)),
        ]],
        dtype=np.int32,
    )
    cv2.fillPoly(roi, polygon, 255)
    edges = cv2.bitwise_and(edges, roi)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=20,
        maxLineGap=45,
    )
    if lines is None:
        return None, None

    left_points: list[tuple[float, float]] = []
    right_points: list[tuple[float, float]] = []
    center = width / 2.0

    for line in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = map(float, line)
        if abs(x2 - x1) < 1.0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if not 0.25 <= abs(slope) <= 6.0:
            continue
        midpoint_x = 0.5 * (x1 + x2)
        points = [(x1 / width, y1 / height), (x2 / width, y2 / height)]
        if slope < 0 and midpoint_x < center:
            left_points.extend(points)
        elif slope > 0 and midpoint_x > center:
            right_points.extend(points)

    def fit(points: list[tuple[float, float]]) -> np.ndarray | None:
        if len(points) < 4:
            return None
        points_array = np.asarray(points)
        y = points_array[:, 1]
        x = points_array[:, 0]
        if np.ptp(y) < 0.025:
            return None
        unique_y = np.unique(np.round(y, decimals=5)).size
        degree = min(3, len(points) - 1, unique_y - 1)
        if len(points) < 8:
            degree = min(degree, 2)
        if degree < 1:
            return None
        coefficients = np.polyfit(y, x, deg=degree)
        residual = x - np.polyval(coefficients, y)
        if float(np.sqrt(np.mean(residual**2))) > 0.05:
            return None
        # Store c0..c3 for x = c0 + c1*y + c2*y^2 + c3*y^3.
        ascending = coefficients[::-1]
        return np.pad(ascending, (0, 4 - len(ascending)))

    return fit(left_points), fit(right_points)


def main() -> None:
    frame_times = load("global_pose/frame_times")
    positions = load("global_pose/frame_positions")
    orientations = load("global_pose/frame_orientations")

    signal_paths = {
        "speed": ("processed_log/CAN/speed/t", "processed_log/CAN/speed/value"),
        "steering": (
            "processed_log/CAN/steering_angle/t",
            "processed_log/CAN/steering_angle/value",
        ),
        "gyro": ("processed_log/IMU/gyro/t", "processed_log/IMU/gyro/value"),
        "accelerometer": (
            "processed_log/IMU/accelerometer/t",
            "processed_log/IMU/accelerometer/value",
        ),
    }

    frequencies: dict[str, float] = {}
    synchronized: dict[str, np.ndarray] = {}
    coverage: dict[str, bool] = {}
    for name, (time_path, value_path) in signal_paths.items():
        signal_times = load(time_path)
        values = load(value_path)
        frequencies[name] = frequency_hz(signal_times)
        coverage[name] = bool(signal_times[0] <= frame_times[1] and signal_times[-1] >= frame_times[-2])
        synchronized[name] = interpolate_signal(frame_times, signal_times, values)

    future_steps = round(FUTURE_SECONDS * CAMERA_RATE_HZ)
    labels = []
    for index in range(len(frame_times) - future_steps):
        delta_ecef = positions[index + future_steps] - positions[index]
        ecef_from_device = quaternion_to_rotation(orientations[index])
        delta_device = ecef_from_device.T @ delta_ecef
        # Device coordinates are forward, right, down. Lateral-left is negative right.
        labels.append((delta_device[0], -delta_device[1]))
    labels_array = np.asarray(labels)
    plausible_labels = (
        np.isfinite(labels_array).all(axis=1)
        & (labels_array[:, 0] > 0.0)
        & (labels_array[:, 0] < 80.0)
        & (np.abs(labels_array[:, 1]) < 20.0)
    )

    capture = cv2.VideoCapture(str(ROOT / "video.hevc"))
    decoded_frames = 0
    left_detected = 0
    right_detected = 0
    both_detected = 0
    finite_coefficients = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded_frames += 1
        left, right = extract_lane_polynomials(frame)
        left_detected += int(left is not None)
        right_detected += int(right is not None)
        both_detected += int(left is not None and right is not None)
        if left is not None:
            finite_coefficients += int(np.isfinite(left).all() and left.shape == (4,))
        if right is not None:
            finite_coefficients += int(np.isfinite(right).all() and right.shape == (4,))
    capture.release()

    frame_dt = np.diff(frame_times)
    all_synced_finite = all(np.isfinite(value).all() for value in synchronized.values())
    lane_any_rate = (left_detected + right_detected) / (2 * max(decoded_frames, 1))
    lane_both_rate = both_detected / max(decoded_frames, 1)

    result = {
        "video": {
            "decoded_frames": decoded_frames,
            "expected_frames": EXPECTED_FRAMES,
            "frame_size": [1164, 874],
            "timestamp_rate_hz": frequency_hz(frame_times),
            "timestamp_jitter_std_ms": float(np.std(frame_dt) * 1000.0),
        },
        "signals": {
            "frequency_hz": frequencies,
            "covers_video_interval": coverage,
            "synchronized_values_finite": all_synced_finite,
        },
        "trajectory_labels": {
            "count": int(len(labels_array)),
            "plausible_fraction": float(np.mean(plausible_labels)),
            "median_forward_m_at_2_5s": float(np.median(labels_array[:, 0])),
            "lateral_p05_p50_p95_m": np.percentile(labels_array[:, 1], [5, 50, 95]).tolist(),
        },
        "lane_extraction": {
            "left_detection_rate": left_detected / max(decoded_frames, 1),
            "right_detection_rate": right_detected / max(decoded_frames, 1),
            "both_detection_rate": lane_both_rate,
            "individual_boundary_detection_rate": lane_any_rate,
            "all_returned_coefficients_finite": finite_coefficients == left_detected + right_detected,
            "coefficient_order": ["c0", "c1", "c2", "c3"],
            "coordinate_note": "normalized image x(y); metric ego coefficients require calibrated IPM or a lane network",
        },
    }

    result["go_no_go"] = {
        "step_1_video_and_sample": decoded_frames == EXPECTED_FRAMES,
        "step_2_sync_and_labels": all(coverage.values())
        and all_synced_finite
        and float(np.mean(plausible_labels)) > 0.98,
        "step_3_lane_signal": lane_any_rate >= 0.40 and finite_coefficients > 0,
    }
    result["go_no_go"]["dataset_usable_for_pipeline"] = all(result["go_no_go"].values())

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
