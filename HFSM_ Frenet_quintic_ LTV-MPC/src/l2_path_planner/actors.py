"""Actor trajectories and the perception views they produce.

Actors are benchmark stimuli, not planner inputs.  They are described in road
coordinates ``(t, s, d, v)`` and converted to the ego frame before being handed
to the camera/radar boundary layer.  This replaces the old static
``truck_x_m`` stimuli, so overtaking, approach and oncoming traffic actually
move.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np

from .messages import Header
from .road_geometry import RoadGeometry
from .signals import CameraObject, RadarObject, object_size_m


@dataclass(slots=True)
class ActorState:
    id: str
    object_class: str
    s_m: float
    d_m: float
    speed_mps: float
    x_m: float
    y_m: float
    yaw_rad: float
    ego_x_m: float = 0.0
    ego_y_m: float = 0.0
    ego_yaw_rad: float = 0.0
    length_m: float = 4.5
    width_m: float = 1.9


@dataclass
class ActorModel:
    actors: list[dict]
    road: RoadGeometry
    ego_width_m: float = 1.9
    ego_length_m: float = 4.5
    camera_range_m: float = 80.0
    camera_half_width_m: float = 5.25
    radar_range_m: float = 60.0
    radar_mount_x_m: float = -1.0
    radar_mount_y_m: float = 0.9

    def __post_init__(self) -> None:
        self.actors = [dict(actor) for actor in (self.actors or [])]
        self._ids: dict[str, int] = {}
        for index, actor in enumerate(self.actors):
            actor["trajectory"] = sorted(
                (dict(point) for point in actor.get("trajectory", [])),
                key=lambda point: float(point["t"]),
            )
            self._ids[str(actor.get("id", f"actor{index}"))] = index + 1

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def _sample(self, actor: dict, t: float) -> tuple[float, float, float]:
        points = actor.get("trajectory", [])
        times = [float(point["t"]) for point in points]
        if not times:
            return 0.0, 0.0, 0.0
        if t <= times[0]:
            first = points[0]
            return float(first.get("s", 0.0)), float(first.get("d", 0.0)), float(first.get("v", 0.0))
        if t >= times[-1]:
            last = points[-1]
            v = float(last.get("v", 0.0))
            return (
                float(last.get("s", 0.0)) + v * (t - times[-1]),
                float(last.get("d", 0.0)),
                v,
            )
        index = bisect.bisect_right(times, t) - 1
        left, right = points[index], points[index + 1]
        span = max(times[index + 1] - times[index], 1e-9)
        alpha = (t - times[index]) / span
        return (
            float(left.get("s", 0.0)) + alpha * (float(right.get("s", 0.0)) - float(left.get("s", 0.0))),
            float(left.get("d", 0.0)) + alpha * (float(right.get("d", 0.0)) - float(left.get("d", 0.0))),
            float(left.get("v", 0.0)) + alpha * (float(right.get("v", 0.0)) - float(left.get("v", 0.0))),
        )

    def _active(self, actor: dict, t: float) -> bool:
        start = float(actor.get("t_start", -1e9))
        end = float(actor.get("t_end", 1e9))
        return start <= t < end

    def states(self, t: float) -> list[ActorState]:
        result: list[ActorState] = []
        for actor in self.actors:
            if not self._active(actor, t):
                continue
            s, d, v = self._sample(actor, t)
            oncoming = str(actor.get("direction", "SAME")).upper() == "ONCOMING"
            x, y, yaw, _ = self.road.frenet_to_global(
                np.array([s]), np.array([d]), np.array([0.0]), max(abs(v), 0.1),
            )
            length, width = object_size_m(actor.get("class", "CAR"))
            result.append(ActorState(
                id=str(actor.get("id", "actor")),
                object_class=str(actor.get("class", "CAR")),
                s_m=float(s), d_m=float(d), speed_mps=float(v),
                x_m=float(x[0]), y_m=float(y[0]),
                yaw_rad=float(yaw[0]) + (math.pi if oncoming else 0.0),
                length_m=float(actor.get("length_m", length)),
                width_m=float(actor.get("width_m", width)),
            ))
        return result

    @staticmethod
    def _to_ego(state: ActorState, ego_x_m: float, ego_y_m: float, ego_yaw_rad: float) -> None:
        cos_yaw, sin_yaw = math.cos(-ego_yaw_rad), math.sin(-ego_yaw_rad)
        dx, dy = state.x_m - ego_x_m, state.y_m - ego_y_m
        state.ego_x_m = dx * cos_yaw - dy * sin_yaw
        state.ego_y_m = dx * sin_yaw + dy * cos_yaw
        state.ego_yaw_rad = state.yaw_rad - ego_yaw_rad

    def states_in_ego_frame(
        self, t: float, ego_x_m: float, ego_y_m: float, ego_yaw_rad: float
    ) -> list[ActorState]:
        states = self.states(t)
        for state in states:
            self._to_ego(state, ego_x_m, ego_y_m, ego_yaw_rad)
        return states

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------
    def camera_objects(
        self, t: float, ego_x_m: float, ego_y_m: float, ego_yaw_rad: float, header: Header,
    ) -> list[CameraObject]:
        camera: list[CameraObject] = []
        for state in self.states(t):
            self._to_ego(state, ego_x_m, ego_y_m, ego_yaw_rad)
            if -5.0 <= state.ego_x_m <= self.camera_range_m and abs(state.ego_y_m) <= self.camera_half_width_m:
                camera.append(CameraObject(
                    object_id=self._ids.get(state.id, 0),
                    object_class=state.object_class,
                    x_m=state.ego_x_m,
                    y_m=state.ego_y_m,
                    heading_rad=state.ego_yaw_rad,
                    existence_probability=1.0,
                ))
        return camera

    def radar_objects(
        self, t: float, ego_x_m: float, ego_y_m: float, ego_yaw_rad: float, ego_speed_mps: float,
        header: Header,
    ) -> tuple[list[RadarObject], list[RadarObject]]:
        rear_left: list[RadarObject] = []
        rear_right: list[RadarObject] = []
        for state in self.states(t):
            self._to_ego(state, ego_x_m, ego_y_m, ego_yaw_rad)
            if -self.radar_range_m <= state.ego_x_m < 0.0:
                left_side = state.ego_y_m <= 0.0
                mount_y = -self.radar_mount_y_m if left_side else self.radar_mount_y_m
                radar = RadarObject(
                    object_id=self._ids.get(state.id, 0),
                    x_m=state.ego_x_m - self.radar_mount_x_m,
                    y_m=state.ego_y_m - mount_y,
                    vx_rel_mps=state.speed_mps - ego_speed_mps,
                    vy_rel_mps=0.0,
                    rcs_dbsm=10.0 if state.object_class == "TRUCK" else 0.0,
                    motion_state="MOVING",
                    existence_probability=1.0,
                )
                (rear_left if left_side else rear_right).append(radar)
        return rear_left, rear_right

    # ------------------------------------------------------------------
    # Hazard / clearance summary used to fill BehaviorInputs
    # ------------------------------------------------------------------
    def occupancy(self, t: float, ego_x_m: float, ego_y_m: float, ego_speed_mps: float) -> dict:
        ego_s, ego_d, _, _ = self.road.project(ego_x_m, ego_y_m)
        width = self.road.width_at(ego_s)
        summary = {
            "lead_gap_m": 1e6,
            "lead_speed_mps": 1e6,
            "obstacle_distance_m": 1e6,
            "clearance_left_m": 10.0,
            "clearance_right_m": 10.0,
            "hazard_left_present": False,
            "hazard_right_present": False,
            "adjacent_hazard_side": 0,
            "adjacent_clearance_m": 10.0,
            "adjacent_lane_safe": True,
            "rear_ttc_s": 99.0,
            "rear_req_decel_mps2": 0.0,
        }
        left: list[tuple[float, bool]] = []
        right: list[tuple[float, bool]] = []
        for state in self.states(t):
            same_lane = abs(state.d_m - ego_d) <= 0.6 * width
            ahead = state.s_m - ego_s
            if same_lane and ahead > 0.0:
                gap = ahead - 0.5 * state.length_m - 0.5 * self.ego_length_m
                if gap < summary["lead_gap_m"]:
                    summary["lead_gap_m"] = gap
                    summary["lead_speed_mps"] = state.speed_mps
                summary["obstacle_distance_m"] = min(summary["obstacle_distance_m"], max(gap, 0.0))
            overlap = abs(ahead) <= 0.5 * (state.length_m + self.ego_length_m) + 4.0
            clearance = abs(state.d_m - ego_d) - 0.5 * (state.width_m + self.ego_width_m)
            if state.d_m < ego_d - 0.5 * width:
                left.append((clearance, overlap))
            elif state.d_m > ego_d + 0.5 * width:
                right.append((clearance, overlap))
            if ahead < 0.0 and abs(state.d_m - ego_d) <= 1.5 * width:
                distance = max(-ahead - 0.5 * (state.length_m + self.ego_length_m), 1e-3)
                closing = ego_speed_mps - state.speed_mps
                if closing > 0.1:
                    ttc = distance / closing
                    if ttc < summary["rear_ttc_s"]:
                        summary["rear_ttc_s"] = ttc
                        summary["rear_req_decel_mps2"] = closing ** 2 / (2.0 * distance)
        if left:
            summary["clearance_left_m"] = min(item[0] for item in left)
            summary["hazard_left_present"] = any(clearance < 1.2 and overlap for clearance, overlap in left)
        if right:
            summary["clearance_right_m"] = min(item[0] for item in right)
            summary["hazard_right_present"] = any(clearance < 1.2 and overlap for clearance, overlap in right)
        hazards: list[tuple[float, int]] = []
        if summary["hazard_left_present"]:
            hazards.append((summary["clearance_left_m"], 1))
        if summary["hazard_right_present"]:
            hazards.append((summary["clearance_right_m"], 2))
        if hazards:
            clearance, side = min(hazards)
            summary["adjacent_hazard_side"] = side
            summary["adjacent_clearance_m"] = clearance
            summary["adjacent_lane_safe"] = False
        return summary

    def trajectory_for_ui(
        self, times: list[float], ego_x_m: float, ego_y_m: float, ego_yaw_rad: float
    ) -> list[dict]:
        rows: list[dict] = []
        for t in times:
            for state in self.states(t):
                self._to_ego(state, ego_x_m, ego_y_m, ego_yaw_rad)
                rows.append({
                    "t": float(t), "id": state.id, "class": state.object_class,
                    "x_m": state.ego_x_m, "y_m": state.ego_y_m,
                    "gx_m": state.x_m, "gy_m": state.y_m,
                    "s_m": state.s_m, "d_m": state.d_m, "v_mps": state.speed_mps,
                })
        return rows
