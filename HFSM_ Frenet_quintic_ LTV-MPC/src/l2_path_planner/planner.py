from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from .estimation import LateralStateEstimator
from .messages import (
    Function,
    FunctionRequests,
    Header,
    InterventionPhase,
    LaneModel,
    LateralLimits,
    LateralManeuver,
    PathStatus,
    RequestState,
    Side,
    Trajectory,
    TrajectoryPoint,
    VehicleStateEstimate,
)
from .road import SimpleRoad, wrap_angle


MODE_LIMITS = {
    # The controller derates the HWA envelope to 80 % of the 2.5 m/s² nominal
    # capability.  Planning to the nominal value would hand the controller a
    # curve it is not allowed to hold, so the planner uses the derated budget.
    Function.HWA: (2.0, 0.5, 5.0),
    Function.LKA: (3.0, 1.5, 5.0),
    Function.ELK: (4.0, 3.0, 10.0),
}


@dataclass(slots=True)
class Quintic:
    coeff: np.ndarray

    @classmethod
    def from_boundary(
        cls, d0: float, v0: float, a0: float, d1: float, v1: float, a1: float, duration_s: float
    ) -> "Quintic":
        t = float(duration_s)
        coeff = np.zeros(6)
        coeff[:3] = (d0, v0, 0.5 * a0)
        mat = np.array(
            [
                [t**3, t**4, t**5],
                [3 * t**2, 4 * t**3, 5 * t**4],
                [6 * t, 12 * t**2, 20 * t**3],
            ],
            dtype=float,
        )
        rhs = np.array(
            [
                d1 - coeff[0] - coeff[1] * t - coeff[2] * t**2,
                v1 - coeff[1] - 2 * coeff[2] * t,
                a1 - 2 * coeff[2],
            ]
        )
        coeff[3:] = np.linalg.solve(mat, rhs)
        return cls(coeff)

    def evaluate(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        c = self.coeff
        d = sum(c[i] * t**i for i in range(6))
        v = sum(i * c[i] * t ** (i - 1) for i in range(1, 6))
        a = sum(i * (i - 1) * c[i] * t ** (i - 2) for i in range(2, 6))
        j = sum(i * (i - 1) * (i - 2) * c[i] * t ** (i - 3) for i in range(3, 6))
        return d, v, a, j


@dataclass(slots=True)
class Candidate:
    duration_s: float
    t: np.ndarray
    s: np.ndarray
    d: np.ndarray
    d_dot: np.ndarray
    d_ddot: np.ndarray
    jerk: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    curvature: np.ndarray
    front_angle: np.ndarray
    cost: float


class IntegrityMonitor:
    def __init__(self, max_age_s: float, future_tolerance_s: float):
        self.max_age_us = int(max_age_s * 1e6)
        self.future_tolerance_us = int(future_tolerance_s * 1e6)
        self.last_counter: dict[str, int] = {}
        self.last_stamp_us: dict[str, int] = {}

    def valid(self, name: str, header: Header, now_us: int) -> bool:
        if not header.valid:
            return False
        age = now_us - header.stamp_us
        if age > self.max_age_us or age < -self.future_tolerance_us:
            return False
        previous_counter = self.last_counter.get(name)
        previous_stamp = self.last_stamp_us.get(name)
        if previous_counter is not None and header.counter == previous_counter:
            if previous_stamp is not None and header.stamp_us == previous_stamp:
                # A consumer faster than the sensor (planner 40 Hz, camera 10 Hz)
                # legitimately re-reads the same frame; the age check above is
                # what bounds staleness.
                return True
            # F15: a new stamp with a repeated counter is a counter fault.
            return False
        if previous_stamp is not None and header.stamp_us < previous_stamp:
            return False
        self.last_counter[name] = header.counter
        self.last_stamp_us[name] = header.stamp_us
        return True


class HFSM:
    def __init__(self) -> None:
        self.active = Function.NONE
        self.phase = InterventionPhase.NONE
        self.completed_latch: set[Function] = set()
        self.intervention_start_s: float | None = None
        self.stable_since_s: float | None = None

    def select(self, requests: FunctionRequests, state_valid: bool, now_s: float) -> Function:
        request_map = {r.function: r for r in requests.requests}
        for f in (Function.LKA, Function.ELK):
            req = request_map.get(f)
            if req is None or req.state != RequestState.REQUEST:
                self.completed_latch.discard(f)
        requested = []
        for f in (Function.ELK, Function.LKA, Function.HWA):
            req = request_map.get(f)
            if req and req.state == RequestState.REQUEST and f not in self.completed_latch:
                if f in (Function.LKA, Function.ELK) and not state_valid:
                    continue
                requested.append(f)
        selected = requested[0] if requested else Function.NONE
        if selected != self.active:
            self.active = selected
            self.phase = InterventionPhase.ACTIVE if selected in (Function.LKA, Function.ELK) else InterventionPhase.NONE
            self.intervention_start_s = now_s if self.phase == InterventionPhase.ACTIVE else None
            self.stable_since_s = None
        return self.active

    def update_completion(
        self, now_s: float, d_m: float, v_lat_mps: float, heading_error_rad: float,
        ay_added_mps2: float, ay_added_limit_mps2: float = 0.2,
    ) -> bool:
        if self.active not in (Function.LKA, Function.ELK):
            self.stable_since_s = None
            return False
        stable = (
            abs(d_m) <= 0.1
            and abs(v_lat_mps) <= 0.1
            and abs(heading_error_rad) <= math.radians(0.5)
            and abs(ay_added_mps2) <= ay_added_limit_mps2
        )
        if stable:
            self.stable_since_s = now_s if self.stable_since_s is None else self.stable_since_s
            if now_s - self.stable_since_s >= 0.5:
                completed = self.active
                self.completed_latch.add(completed)
                self.phase = InterventionPhase.COMPLETED
                return True
        else:
            self.stable_since_s = None
        return False

    def after_completion(self, requests: FunctionRequests, now_s: float) -> None:
        hwa = requests.request_for(Function.HWA)
        self.active = Function.HWA if hwa and hwa.state == RequestState.REQUEST else Function.NONE
        self.intervention_start_s = None
        self.stable_since_s = None


class FrenetPlanner:
    def __init__(self, config: dict, road: SimpleRoad, vehicle_config: dict):
        self.cfg = config
        self.road = road
        self.vehicle = vehicle_config
        self.path_dt = float(config.get("path_point_dt_s", 0.1))
        self.horizon = float(config.get("path_horizon_s", 3.0))
        self.t = np.arange(0.0, self.horizon + 0.5 * self.path_dt, self.path_dt)
        self.counter = 0
        self.hfsm = HFSM()
        # v_y is not published by the sheet; the planner reconstructs it.
        self.estimator = LateralStateEstimator(
            dt_s=float(config.get("state_estimator_dt_s", 0.01)),
        )
        pcfg = config["planner"]
        self.monitor = IntegrityMonitor(pcfg["max_input_age_s"], pcfg["future_stamp_tolerance_s"])
        self.last_trajectory: Trajectory | None = None
        self.last_lane_width = road.lane_width_m
        self.last_valid_lane_s: float | None = None
        self.compute_times_ms: list[float] = []
        self.lane_change_started_s: float | None = None
        self.lane_change_target_m: float | None = None
        self.lane_change_initial: tuple[float, float, float] | None = None

    def update_estimation(
        self,
        lane: LaneModel | None,
        speed_mps: float,
        yaw_rate_rps: float,
        lateral_acceleration_mps2: float,
    ) -> None:
        """Advance the lateral state estimate at the IMU rate."""
        self.estimator.update(lane, speed_mps, yaw_rate_rps, lateral_acceleration_mps2)

    @property
    def estimated_lateral_velocity_mps(self) -> float:
        return self.estimator.lateral_velocity_mps

    @property
    def estimated_offset_m(self) -> float:
        return self.estimator.offset_m

    def _lane_valid(self, lane: LaneModel, now_us: int) -> bool:
        if not self.monitor.valid("lane", lane.header, now_us):
            return False
        if lane.left is None and lane.right is None:
            return False
        if lane.left is not None and lane.right is not None:
            width = lane.derived_lane_width()
            if width < 2.2 or width > 5.5:
                return False
        return True

    def _initial_state(
        self, road_s: float, road_d: float, heading_error: float, state: VehicleStateEstimate, active: Function
    ) -> tuple[float, float, float]:
        measured = (road_d, self.estimator.lateral_velocity_mps, 0.0)
        if self.last_trajectory is None or active in (Function.LKA, Function.ELK):
            return measured
        points = self.last_trajectory.points
        if len(points) < 2:
            return measured
        # Planner ticks at 25 ms while published points are 100 ms apart.
        # Advance the old path by the actual elapsed planner period rather than
        # jumping to point 1 on every replan.
        alpha = min(1.0, self.cfg["planner_dt_s"] / max(self.cfg["path_point_dt_s"], 1e-6))
        stitched_d = (1.0 - alpha) * points[0].road_d_m + alpha * points[1].road_d_m
        stitched_d_dot = (
            (1.0 - alpha) * points[0].lateral_velocity_mps
            + alpha * points[1].lateral_velocity_mps
        )
        if abs(stitched_d - road_d) <= self.cfg["planner"]["stitch_threshold_m"]:
            return stitched_d, stitched_d_dot, 0.0
        return measured

    def _durations(
        self, active: Function, maneuver: LateralManeuver = LateralManeuver.NONE,
    ) -> list[float]:
        if maneuver in (LateralManeuver.LANE_CHANGE_LEFT, LateralManeuver.LANE_CHANGE_RIGHT):
            return list(self.cfg["planner"].get("lane_change_durations_s", [8.0, 9.0, 10.0]))
        key = {Function.HWA: "hwa_durations_s", Function.LKA: "lka_durations_s", Function.ELK: "elk_durations_s"}.get(active)
        return list(self.cfg["planner"].get(key, [3.0])) if key else [3.0]

    def _build_candidate(
        self, active: Function, road_s: float, speed: float, initial: tuple[float, float, float], duration: float,
        target_d_m: float = 0.0,
    ) -> Candidate:
        poly = Quintic.from_boundary(*initial, target_d_m, 0.0, 0.0, duration)
        eval_t = np.minimum(self.t, duration)
        d, d_dot, d_ddot, jerk = poly.evaluate(eval_t)
        after = self.t >= duration
        d[after], d_dot[after], d_ddot[after], jerk[after] = target_d_m, 0.0, 0.0, 0.0
        s = road_s + speed * self.t
        x, y, yaw, kappa_center = self.road.frenet_to_global(s, d, d_dot, speed)
        curvature = kappa_center + d_ddot / max(speed * speed, 1.0)
        front_angle = np.arctan(self.vehicle["wheelbase_m"] * curvature)
        target_error = abs(d[-1] - target_d_m)
        cost = 8.0 * float(np.mean(jerk * jerk)) + 50.0 * target_error**2 + 0.03 * duration
        return Candidate(duration, self.t.copy(), s, d, d_dot, d_ddot, jerk, x, y, yaw, curvature, front_angle, cost)

    def _build_latched_candidate(
        self, active: Function, road_s: float, speed: float,
        initial: tuple[float, float, float], duration: float,
        target_d_m: float, elapsed_s: float,
    ) -> Candidate:
        poly = Quintic.from_boundary(*initial, target_d_m, 0.0, 0.0, duration)
        profile_t = np.minimum(elapsed_s + self.t, duration)
        d, d_dot, d_ddot, jerk = poly.evaluate(profile_t)
        after = profile_t >= duration
        d[after], d_dot[after], d_ddot[after], jerk[after] = target_d_m, 0.0, 0.0, 0.0
        s = road_s + speed * self.t
        x, y, yaw, kappa_center = self.road.frenet_to_global(s, d, d_dot, speed)
        curvature = kappa_center + d_ddot / max(speed * speed, 1.0)
        front_angle = np.arctan(self.vehicle["wheelbase_m"] * curvature)
        target_error = abs(d[-1] - target_d_m)
        cost = 8.0 * float(np.mean(jerk * jerk)) + 50.0 * target_error**2 + 0.03 * duration
        return Candidate(
            duration, self.t.copy(), s, d, d_dot, d_ddot, jerk,
            x, y, yaw, curvature, front_angle, cost,
        )

    def _feasible(
        self, c: Candidate, active: Function, speed: float,
        target_d_m: float = 0.0, maneuver: LateralManeuver = LateralManeuver.NONE,
    ) -> bool:
        if active == Function.NONE:
            return True
        ay_max, added_max, jerk_max = MODE_LIMITS[active]
        _, _, _, center_kappa = self.road.geometry(c.s)
        ay = speed * speed * c.curvature
        ay_center = speed * speed * center_kappa
        margin = self.road.lane_width_m / 2 - self.vehicle["width_m"] / 2
        steering_rate = np.gradient(c.front_angle, self.path_dt) * self.vehicle["steering_ratio"]
        lane_change = maneuver in (LateralManeuver.LANE_CHANGE_LEFT, LateralManeuver.LANE_CHANGE_RIGHT)
        corridor_ok = (
            np.max(np.abs(c.d)) <= 1.5 * self.road.lane_width_m + 1e-6 if lane_change
            else np.max(np.abs(c.d - target_d_m)) <= margin + 1e-6
        )
        return bool(
            corridor_ok
            and np.max(np.abs(ay)) <= ay_max + 1e-6
            and np.max(np.abs(ay - ay_center)) <= added_max + 1e-6
            and np.max(np.abs(c.jerk)) <= jerk_max + 1e-6
            and np.max(np.abs(c.front_angle)) <= self.vehicle["max_wheel_angle_rad"] + 1e-6
            and np.max(np.abs(steering_rate)) <= self.vehicle["max_steering_wheel_rate_radps"] + 1e-6
        )

    def _continuity_blend(self, candidate: Candidate) -> Candidate:
        if self.last_trajectory is None:
            return candidate
        old = self.last_trajectory.points
        old_d = np.array([p.road_d_m for p in old])
        old_d_dot = np.array([p.lateral_velocity_mps for p in old])
        old_curvature = np.array([p.curvature_inv_m for p in old])
        limit = self.cfg["planner"]["continuity_position_m"]
        delta = np.clip(candidate.d - old_d, -limit, limit)
        d = old_d + delta
        d_dot = np.gradient(d, self.path_dt)
        heading_limit = self.cfg["planner"]["continuity_heading_rad"]
        old_lateral_heading = np.arctan2(old_d_dot, self.vehicle["speed_mps"])
        lateral_heading = np.arctan2(d_dot, self.vehicle["speed_mps"])
        lateral_heading = old_lateral_heading + np.clip(lateral_heading - old_lateral_heading, -heading_limit, heading_limit)
        d_dot = self.vehicle["speed_mps"] * np.tan(lateral_heading)
        x, y, yaw, kappa_center = self.road.frenet_to_global(candidate.s, d, d_dot, self.vehicle["speed_mps"])
        raw_curvature = kappa_center + np.gradient(d_dot, self.path_dt) / max(self.vehicle["speed_mps"] ** 2, 1.0)
        curvature_limit = self.cfg["planner"]["continuity_curvature_inv_m"]
        curvature = old_curvature + np.clip(raw_curvature - old_curvature, -curvature_limit, curvature_limit)
        d_ddot = (curvature - kappa_center) * self.vehicle["speed_mps"] ** 2
        jerk = np.gradient(d_ddot, self.path_dt)
        front_angle = np.arctan(self.vehicle["wheelbase_m"] * curvature)
        return Candidate(
            candidate.duration_s, candidate.t, candidate.s, d, d_dot, d_ddot, jerk,
            x, y, yaw, curvature, front_angle, candidate.cost
        )

    def plan(
        self,
        now_s: float,
        lane: LaneModel,
        requests: FunctionRequests,
        state: VehicleStateEstimate,
        ego_x_m: float,
        ego_y_m: float,
        ego_yaw_rad: float,
        speed_mps: float,
        lateral_acceleration_mps2: float = 0.0,
    ) -> tuple[Trajectory, PathStatus, LateralLimits, dict]:
        started = time.perf_counter()
        now_us = int(round(now_s * 1e6))
        lane_valid = self._lane_valid(lane, now_us)
        state_valid = state.header.valid and now_us - state.header.stamp_us <= 60_000
        requests_valid = requests.header.valid and now_us - requests.header.stamp_us <= 75_000
        road_s, road_d, road_yaw, road_kappa = self.road.project(ego_x_m, ego_y_m)
        heading_error = float(wrap_angle(np.array([ego_yaw_rad - road_yaw]))[0])
        if not requests_valid:
            requests = FunctionRequests(Header(now_us, 0, False), [], False)
        active = self.hfsm.select(requests, state_valid, now_s)
        active_request = requests.request_for(active)
        target_d_m = active_request.target_offset_m if active_request is not None else 0.0
        # F20: If ego-motion/state or requests are invalid, path is immediately invalid (<= 1 cycle)
        # F05 & F12: If lane is lost, coast up to coast_time_s (1.0s) provided state and requests remain valid
        if not state_valid or not requests_valid:
            path_valid = False
        elif lane_valid:
            self.last_valid_lane_s = now_s
            self.last_lane_width = lane.derived_lane_width()
            path_valid = True
        elif self.last_valid_lane_s is not None and now_s - self.last_valid_lane_s <= self.cfg["planner"]["coast_time_s"]:
            path_valid = self.last_trajectory is not None
        else:
            path_valid = False
        initial = self._initial_state(road_s, road_d, heading_error, state, active)
        maneuver = active_request.maneuver if active_request is not None else LateralManeuver.NONE
        lane_change = maneuver in (LateralManeuver.LANE_CHANGE_LEFT, LateralManeuver.LANE_CHANGE_RIGHT)
        if lane_change:
            if self.lane_change_target_m is None or abs(self.lane_change_target_m - target_d_m) > 1e-6:
                self.lane_change_started_s = now_s
                self.lane_change_target_m = target_d_m
                self.lane_change_initial = initial
            maneuver_elapsed = now_s - (self.lane_change_started_s or now_s)
        else:
            self.lane_change_started_s = None
            self.lane_change_target_m = None
            self.lane_change_initial = None
            maneuver_elapsed = 0.0
        if lane_change:
            candidates = [
                self._build_latched_candidate(
                    active, road_s, speed_mps, self.lane_change_initial or initial,
                    duration, target_d_m, maneuver_elapsed,
                )
                for duration in self._durations(active, maneuver)
            ]
        else:
            candidates = [
                self._build_candidate(active, road_s, speed_mps, initial, duration, target_d_m)
                for duration in self._durations(active, maneuver)
            ]
        feasible = [c for c in candidates if self._feasible(c, active, speed_mps, target_d_m, maneuver)]
        selected = min(feasible or candidates, key=lambda c: c.cost)
        selected = self._continuity_blend(selected)
        points = [
            TrajectoryPoint(
                time_from_start_s=float(selected.t[i]),
                x_m=float(selected.x[i] - ego_x_m),
                y_m=float(selected.y[i] - ego_y_m),
                yaw_rad=float(wrap_angle(np.array([selected.yaw[i] - ego_yaw_rad]))[0]),
                longitudinal_velocity_mps=speed_mps,
                lateral_velocity_mps=float(selected.d_dot[i]),
                acceleration_mps2=0.0,
                heading_rate_rps=float(speed_mps * selected.curvature[i]),
                curvature_inv_m=float(selected.curvature[i]),
                front_wheel_angle_rad=float(selected.front_angle[i]),
                road_s_m=float(selected.s[i]),
                road_d_m=float(selected.d[i]),
            )
            for i in range(len(self.t))
        ]
        header = Header(now_us, self.counter, path_valid, "base_link")
        self.counter = (self.counter + 1) % 256
        trajectory = Trajectory(header, points)
        # Completion uses the measured/estimated CG lateral acceleration, not
        # the commanded steering proxy.  Subtract the acceleration required by
        # the road centreline so only the recovery/intervention contribution
        # remains.  Road bank, crosswind and transient tyre response therefore
        # keep the intervention active until the vehicle has genuinely settled.
        ay_road = speed_mps * speed_mps * road_kappa
        ay_added = float(lateral_acceleration_mps2 - ay_road)
        completed = self.hfsm.update_completion(
            now_s, road_d - target_d_m, state.lateral_velocity_mps,
            heading_error, ay_added,
            float(self.cfg["planner"].get("epsilon_mps2", 0.2)),
        )
        completed_function = active if completed else Function.NONE
        if completed:
            self.hfsm.after_completion(requests, now_s)
            active = self.hfsm.active
        _, _, _, center_kappa = self.road.geometry(selected.s)
        ay_required = speed_mps * speed_mps * np.abs(center_kappa)
        ay_budget = MODE_LIMITS.get(active, MODE_LIMITS[Function.HWA])[0]
        peak_idx = int(np.argmax(ay_required))
        kappa_peak = abs(float(center_kappa[peak_idx]))
        v_max = math.sqrt(ay_budget / kappa_peak) if kappa_peak > 1e-8 else 1000.0
        limit_reached = bool(np.max(ay_required) > ay_budget + self.cfg["planner"]["delta_hysteresis_mps2"])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.compute_times_ms.append(elapsed_ms)
        status = PathStatus(
            Header(now_us, header.counter, True, ""), path_valid, active,
            InterventionPhase.COMPLETED if completed else self.hfsm.phase, limit_reached, elapsed_ms
        )
        limits = LateralLimits(Header(now_us, header.counter, True, ""), v_max, float(selected.s[peak_idx] - road_s), limit_reached)
        self.last_trajectory = trajectory
        diagnostics = {
            "candidate_count": len(candidates),
            "feasible_count": len(feasible),
            "selected_duration_s": selected.duration_s,
            "completed_function": completed_function.name,
            "road_d_m": road_d,
            "target_d_m": target_d_m,
            "heading_error_rad": heading_error,
            "completion_ay_added_mps2": ay_added,
        }
        return trajectory, status, limits, diagnostics

