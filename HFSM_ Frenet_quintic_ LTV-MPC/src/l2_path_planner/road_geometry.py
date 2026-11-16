"""Piecewise road geometry for scenario schema v2.

The signal-list frame is kept: ``x`` forward, ``y`` to the **right**, and a
positive Frenet offset ``d`` / curvature means "to the right".

A road is described by an ordered list of centreline segments::

    {"s_m": 0,   "type": "straight", "length_m": 80}
    {"s_m": 80,  "type": "arc",      "length_m": 160, "radius_m": 1000}
    {"s_m": 240, "type": "clothoid", "length_m": 60,
     "start_curvature_inv_m": 0.0, "end_curvature_inv_m": 0.001}

``s_m`` is the arc-length at which the segment starts; when omitted the
segments are concatenated.  ``radius_m`` is signed (positive = right turn).
The legacy ``cosine`` segment reproduces the old ``SimpleRoad`` bump and is
used by ``configs/baseline.json``.

The class exposes the same API as :class:`l2_path_planner.road.SimpleRoad`
(``geometry``, ``project``, ``frenet_to_global`` and ``lane_width_m``) so the
planner and controller do not need to know which model is in use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .road import wrap_angle

SEGMENT_TYPES = frozenset({"straight", "arc", "clothoid", "cosine"})


def _segment_curvature(segment: dict, local_s: float) -> float:
    kind = str(segment.get("type", "straight")).lower()
    if kind == "straight":
        return 0.0
    if kind == "arc":
        if "curvature_inv_m" in segment:
            return float(segment["curvature_inv_m"])
        radius = float(segment.get("radius_m", 1000.0))
        if abs(radius) < 1e-9:
            return 0.0
        return 1.0 / radius
    if kind == "clothoid":
        length = max(float(segment.get("length_m", 1.0)), 1e-9)
        k0 = float(segment.get("start_curvature_inv_m", 0.0))
        k1 = float(segment.get("end_curvature_inv_m", k0))
        return k0 + (k1 - k0) * min(max(local_s / length, 0.0), 1.0)
    if kind == "cosine":
        length = max(float(segment.get("length_m", 1.0)), 1e-9)
        amplitude = float(segment.get("amplitude_m", 0.0))
        u = min(max(local_s / length, 0.0), 1.0)
        w = 2.0 * math.pi / length
        phase = 2.0 * math.pi * u
        dy = 0.5 * amplitude * w * math.sin(phase)
        ddy = 0.5 * amplitude * w * w * math.cos(phase)
        return ddy / (1.0 + dy * dy) ** 1.5
    raise ValueError(f"unknown centreline segment type: {kind!r}")


@dataclass
class RoadGeometry:
    lane_width_m: float = 3.5
    centerline: list[dict] = field(default_factory=list)
    width_profile: list[dict] = field(default_factory=list)
    markings: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    lanes: dict | None = None
    slope_percent: float = 0.0
    ds: float = 0.5
    s_min: float = -40.0
    s_max: float = 1200.0

    def __post_init__(self) -> None:
        self.centerline = list(self.centerline or [{"type": "straight", "length_m": 2000.0}])
        self.width_profile = sorted(
            (dict(item) for item in (self.width_profile or [])),
            key=lambda item: float(item.get("s_m", 0.0)),
        )
        self.markings = sorted(
            (dict(item) for item in (self.markings or [])),
            key=lambda item: float(item.get("s_start_m", 0.0)),
        )
        self.lanes = self._normalise_lanes(self.lanes)
        self._build_centerline()

    # ------------------------------------------------------------------
    # Build / sample
    # ------------------------------------------------------------------
    def _segments(self) -> list[dict]:
        segments: list[dict] = []
        cursor = 0.0
        for index, raw in enumerate(self.centerline):
            segment = dict(raw)
            kind = str(segment.get("type", "straight")).lower()
            if kind not in SEGMENT_TYPES:
                raise ValueError(f"unknown centreline segment type: {kind!r}")
            start = float(segment.get("s_m", cursor))
            cursor = start
            if "length_m" in segment:
                length = float(segment["length_m"])
            elif kind == "straight" and index == len(self.centerline) - 1:
                length = max(self.s_max - start, 0.0)
            else:
                length = 0.0
            segment["_start"] = start
            segment["_length"] = max(length, 0.0)
            segments.append(segment)
            cursor = start + segment["_length"]
        return segments

    def _build_centerline(self) -> None:
        segments = self._segments()
        defined_end = max((seg["_start"] + seg["_length"] for seg in segments), default=0.0)
        self.s_max = max(float(self.s_max), defined_end + 200.0)
        n = int(math.ceil((self.s_max - self.s_min) / self.ds)) + 1
        s = self.s_min + self.ds * np.arange(n, dtype=float)
        x = np.zeros_like(s)
        y = np.zeros_like(s)
        yaw = np.zeros_like(s)
        kappa = np.zeros_like(s)

        # Backward extension is straight along the initial heading (zero).
        x[s < 0.0] = s[s < 0.0]

        def curvature_at(value: float) -> float:
            for seg in segments:
                start = seg["_start"]
                end = start + seg["_length"]
                if start <= value < end or (seg is segments[-1] and value >= start):
                    return _segment_curvature(seg, value - start)
            return 0.0

        # Midpoint integration from s = 0 to s_max.
        index = int(np.searchsorted(s, 0.0))
        cx, cy, cyaw = 0.0, 0.0, 0.0
        for i in range(index, n):
            value = float(s[i])
            k = curvature_at(value)
            kappa[i] = k
            x[i], y[i], yaw[i] = cx, cy, cyaw
            mid_yaw = cyaw + 0.5 * self.ds * k
            cx += self.ds * math.cos(mid_yaw)
            cy += self.ds * math.sin(mid_yaw)
            cyaw = wrap_angle(cyaw + self.ds * k)
        self._s, self._x, self._y, self._yaw, self._kappa = s, x, y, yaw, kappa

    # ------------------------------------------------------------------
    # Public API (mirrors SimpleRoad)
    # ------------------------------------------------------------------
    def centerline_at(self, s: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s_arr = np.asarray(s, dtype=float)
        x = np.interp(s_arr, self._s, self._x)
        y = np.interp(s_arr, self._s, self._y)
        yaw = np.interp(s_arr, self._s, self._yaw)
        kappa = np.interp(s_arr, self._s, self._kappa)
        return x, y, yaw, kappa

    def geometry(self, s: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s_arr = np.asarray(s, dtype=float)
        _, y, yaw, kappa = self.centerline_at(s_arr)
        return s_arr, y, yaw, kappa

    def project(self, x_m: float, y_m: float) -> tuple[float, float, float, float]:
        # Nearest centreline sample, then exact projection onto the two
        # adjacent polyline segments for a smooth (sub-sample) result.
        distance_sq = (self._x - x_m) ** 2 + (self._y - y_m) ** 2
        index = int(np.argmin(distance_sq))
        best: tuple[float, float, float, float] | None = None
        for i in (index - 1, index):
            if i < 0 or i + 1 >= len(self._s):
                continue
            x0, y0 = float(self._x[i]), float(self._y[i])
            x1, y1 = float(self._x[i + 1]), float(self._y[i + 1])
            dx, dy = x1 - x0, y1 - y0
            length_sq = dx * dx + dy * dy
            if length_sq <= 1e-12:
                continue
            u = ((x_m - x0) * dx + (y_m - y0) * dy) / length_sq
            u = min(max(u, 0.0), 1.0)
            px, py = x0 + u * dx, y0 + u * dy
            length = math.sqrt(length_sq)
            signed_d = (dx * (y_m - py) - dy * (x_m - px)) / length
            s_value = float(self._s[i]) + u * float(self._s[i + 1] - self._s[i])
            if best is None or abs(signed_d) < abs(best[1]):
                _, _, yaw, kappa = self.centerline_at(np.array([s_value]))
                best = (s_value, signed_d, float(yaw[0]), float(kappa[0]))
        if best is None:
            return 0.0, 0.0, 0.0, 0.0
        return best

    def frenet_to_global(
        self, s: np.ndarray, d: np.ndarray, d_dot: np.ndarray, speed_mps: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s_arr = np.asarray(s, dtype=float)
        d_arr = np.asarray(d, dtype=float)
        cx, cy, yaw_c, kappa_c = self.centerline_at(s_arr)
        x = cx - d_arr * np.sin(yaw_c)
        y = cy + d_arr * np.cos(yaw_c)
        yaw = wrap_angle(yaw_c + np.arctan2(np.asarray(d_dot, dtype=float), max(speed_mps, 0.1)))
        return x, y, np.asarray(yaw), kappa_c

    # ------------------------------------------------------------------
    # Scenario helpers
    # ------------------------------------------------------------------
    def width_at(self, s: float) -> float:
        if not self.width_profile:
            return self.lane_width_m
        points = self.width_profile
        if s <= float(points[0].get("s_m", 0.0)):
            return float(points[0].get("width_m", self.lane_width_m))
        for left, right in zip(points, points[1:]):
            s0, s1 = float(left.get("s_m", 0.0)), float(right.get("s_m", 0.0))
            if s0 <= s <= s1:
                w0, w1 = float(left.get("width_m", self.lane_width_m)), float(right.get("width_m", self.lane_width_m))
                alpha = 0.0 if s1 <= s0 else (s - s0) / (s1 - s0)
                return w0 + alpha * (w1 - w0)
        return float(points[-1].get("width_m", self.lane_width_m))

    def markings_at(self, s: float) -> dict:
        default = {
            "left": "DASHED", "right": "DASHED",
            "left_confidence": 1.0, "right_confidence": 1.0,
            "left_view_range_m": 100.0, "right_view_range_m": 100.0,
        }
        selected = default
        for entry in self.markings:
            start = float(entry.get("s_start_m", 0.0))
            end = float(entry.get("s_end_m", 1e9))
            if start <= s < end:
                selected = {**default, **entry}
        return selected

    def edge_at(self, side: str, s: float) -> dict | None:
        wanted = str(side).upper()
        found = None
        for entry in self.edges:
            if str(entry.get("side", "")).upper() != wanted:
                continue
            start = float(entry.get("s_start_m", 0.0))
            end = float(entry.get("s_end_m", 1e9))
            if start <= s < end:
                found = entry
        return found

    def lane_expanding(self, s: float) -> bool:
        return self.width_at(s) > self.lane_width_m + 0.4

    # ------------------------------------------------------------------
    # Lane layout (optional ``road.lanes`` block)
    # ------------------------------------------------------------------
    def _normalise_lanes(self, lanes: dict | None) -> dict | None:
        if not lanes:
            return None
        widths = [float(value) for value in lanes.get("widths_m", [])]
        if not widths:
            return None
        ego_index = int(lanes.get("ego_index", 0))
        roles = [str(role).upper() for role in (lanes.get("roles") or ["CAR"] * len(widths))]
        opposing = lanes.get("opposing")
        if opposing:
            opposing = {
                "count": int(opposing.get("count", 1)),
                "width_m": float(opposing.get("width_m", self.lane_width_m)),
                "separator": str(opposing.get("separator", "DOUBLE_YELLOW")).upper(),
            }
        return {
            "widths_m": widths,
            "ego_index": ego_index,
            "roles": roles,
            "opposing": opposing,
        }

    @property
    def lane_count(self) -> int:
        return len(self.lanes["widths_m"]) if self.lanes else 1

    @property
    def has_opposing(self) -> bool:
        return bool(self.lanes and self.lanes.get("opposing"))

    def lane_widths(self, s: float | None = None) -> list[float]:
        """Lane widths left to right; the ego lane follows ``width_at``."""
        if not self.lanes:
            return [self.width_at(s) if s is not None else self.lane_width_m]
        widths = list(self.lanes["widths_m"])
        if s is not None:
            widths[self.lanes["ego_index"]] = self.width_at(s)
        return widths

    def lane_centers(self, s: float | None = None) -> list[float]:
        """Lane centre offsets ``d`` (positive = right); the ego lane is at 0."""
        if not self.lanes:
            return [0.0]
        widths = self.lane_widths(s)
        ego = self.lanes["ego_index"]
        centers = [0.0] * len(widths)
        cursor = 0.0
        for index in range(ego - 1, -1, -1):
            cursor -= 0.5 * (widths[index] + widths[index + 1])
            centers[index] = cursor
        cursor = 0.0
        for index in range(ego + 1, len(widths)):
            cursor += 0.5 * (widths[index] + widths[index - 1])
            centers[index] = cursor
        return centers

    def lane_boundaries(self, s: float | None = None) -> list[dict]:
        """Boundary offsets left to right, ``index`` 0..lane_count."""
        if not self.lanes:
            width = self.width_at(s) if s is not None else self.lane_width_m
            return [{"d": -width / 2.0, "index": 0}, {"d": width / 2.0, "index": 1}]
        widths = self.lane_widths(s)
        centers = self.lane_centers(s)
        bounds = [{"d": centers[0] - widths[0] / 2.0, "index": 0}]
        for index, (width, center) in enumerate(zip(widths, centers)):
            bounds.append({"d": center + width / 2.0, "index": index + 1})
        return bounds

    def next_lane_offset(self, side: str, s: float | None = None) -> float:
        """Offset of the lane line beyond the ego lane on ``side``."""
        if not self.lanes:
            width = self.width_at(s) if s is not None else self.lane_width_m
            return -1.5 * width if str(side).upper() == "LEFT" else 1.5 * width
        widths = self.lane_widths(s)
        ego = self.lanes["ego_index"]
        if str(side).upper() == "LEFT":
            if ego == 0:
                return -widths[ego] / 2.0
            return -(widths[ego] / 2.0 + widths[ego - 1])
        if ego == len(widths) - 1:
            return widths[ego] / 2.0
        return widths[ego] / 2.0 + widths[ego + 1]

    def opposing_lane_centers(self) -> list[float]:
        """Centre offsets of the opposite-direction lanes, nearest first."""
        if not self.has_opposing:
            return []
        widths = self.lane_widths()
        centers = self.lane_centers()
        width = self.lanes["opposing"]["width_m"]
        edge = centers[0] - widths[0] / 2.0
        return [edge - width / 2.0 - index * width for index in range(self.lanes["opposing"]["count"])]

    def carriageway_extent(self, s: float | None = None) -> tuple[float, float]:
        """``(left, right)`` offsets of the outermost lane boundaries."""
        bounds = self.lane_boundaries(s)
        return bounds[0]["d"], bounds[-1]["d"]

    @classmethod
    def from_config(cls, road_cfg: dict) -> "RoadGeometry":
        cfg = dict(road_cfg or {})
        centerline = cfg.get("centerline")
        if not centerline:
            # Legacy compatibility: the old cosine bump.
            amplitude = float(cfg.get("curve_amplitude_m", 0.0))
            if abs(amplitude) > 1e-9:
                centerline = [
                    {"s_m": 0.0, "type": "straight",
                     "length_m": float(cfg.get("curve_start_m", 0.0))},
                    {"type": "cosine",
                     "length_m": float(cfg.get("curve_length_m", 160.0)),
                     "amplitude_m": amplitude},
                ]
            else:
                centerline = [{"type": "straight", "length_m": 2000.0}]
        return cls(
            lane_width_m=float(cfg.get("lane_width_m", 3.5)),
            centerline=centerline,
            width_profile=cfg.get("width_profile", []),
            markings=cfg.get("markings", []),
            edges=cfg.get("edges", []),
            lanes=cfg.get("lanes"),
            slope_percent=float(cfg.get("slope_percent", 0.0)),
            ds=float(cfg.get("sample_step_m", 0.5)),
            s_max=float(cfg.get("s_max_m", 1200.0)),
        )
