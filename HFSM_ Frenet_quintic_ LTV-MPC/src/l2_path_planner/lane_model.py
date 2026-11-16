"""Synthesise the published ``LaneModel`` from a scenario road.

The camera publishes line offsets in the ego frame (``c0``), heading (``c1``)
and curvature (``c2``) plus a class, an existence probability and a view
range.  Nothing here is a new signal: the class/confidence/range come from the
scenario road description, and the geometry is projected from the ego pose.
"""

from __future__ import annotations

from dataclasses import dataclass

from .messages import Header, LaneLine, LaneModel
from .road_geometry import RoadGeometry


@dataclass
class LaneModelSynthesizer:
    road: RoadGeometry

    def _line(
        self,
        c0: float,
        heading: float,
        c2_inv_km: float,
        line_class: str,
        confidence: float,
        view_range_m: float,
    ) -> LaneLine:
        return LaneLine(
            c0=c0,
            c1=heading,
            c2_inv_km=c2_inv_km,
            view_range_start_m=0.0,
            view_range_end_m=float(view_range_m),
            line_class=str(line_class).upper(),
            existence_probability=float(confidence),
        )

    def build(
        self,
        stamp_s: float,
        counter: int,
        ego_x_m: float,
        ego_y_m: float,
        ego_yaw_rad: float,
        lane_center_m: float,
        previous_width_m: float | None = None,
    ) -> LaneModel:
        s, d, road_yaw, kappa = self.road.project(ego_x_m, ego_y_m)
        width = self.road.width_at(s)
        half = width / 2.0
        # Lane centre expressed in the ego frame (positive = to the right).
        relative_center = lane_center_m - d
        heading = road_yaw - ego_yaw_rad
        c2_inv_km = 0.5 * kappa * 1000.0
        markings = self.road.markings_at(s)

        def side_line(side: str, offset_sign: float) -> LaneLine | None:
            key = side.lower()
            line_class = str(markings.get(key, "DASHED")).upper()
            confidence = float(markings.get(f"{key}_confidence", 1.0))
            view_range = float(markings.get(f"{key}_view_range_m", 100.0))
            edge = self.road.edge_at(side, s)
            if edge is not None:
                # A barrier / road edge is a hard boundary even when the paint
                # is not visible: ELK needs the physical edge, not the paint.
                line_class = str(edge.get("type", "ROAD_EDGE")).upper()
                confidence = max(confidence, 0.9)
            if line_class == "LOST":
                return None
            return self._line(
                relative_center + offset_sign * half,
                heading, c2_inv_km, line_class, confidence, view_range,
            )

        left = side_line("LEFT", -1.0)
        right = side_line("RIGHT", +1.0)
        next_left = self._line(
            relative_center + self.road.next_lane_offset("LEFT", s), heading, c2_inv_km,
            str(markings.get("left", "DASHED")).upper(),
            float(markings.get("left_confidence", 1.0)),
            float(markings.get("left_view_range_m", 100.0)),
        )
        next_right = self._line(
            relative_center + self.road.next_lane_offset("RIGHT", s), heading, c2_inv_km,
            str(markings.get("right", "DASHED")).upper(),
            float(markings.get("right_confidence", 1.0)),
            float(markings.get("right_view_range_m", 100.0)),
        )
        memory_width = float(previous_width_m) if previous_width_m else width
        return LaneModel(
            Header(int(round(stamp_s * 1e6)), counter % 256),
            left, right, lane_width=memory_width,
            next_left=next_left, next_right=next_right,
        )
