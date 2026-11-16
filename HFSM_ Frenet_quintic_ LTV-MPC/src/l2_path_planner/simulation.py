from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np

from .actors import ActorModel
from .adas_stub import AdasTimeline
from .behavior import BehaviorInputs, HierarchicalSFM
from .controller import LTVMPCController
from .disturbance import DisturbanceModel
from .evaluation import write_outputs
from .lane_model import LaneModelSynthesizer
from .messages import (
    Function, Header, LaneModel, Side, VehicleStateEstimate,
)
from .planner import FrenetPlanner
from .road_geometry import RoadGeometry
from .road import wrap_angle
from .scenario import Scenario, coerce
from .signals import (
    ChassisSignalModel, RATES_HZ, mps_to_kmh, rad_to_deg, radps_to_degps,
)
from .vehicle import DynamicBicycle


class ClosedLoopSimulation:
    def __init__(self, config: dict | Scenario, config_path: Path | None = None, controller_backend: str = "auto"):
        self.scenario = coerce(config)
        self.cfg = self.scenario.runtime
        self.config_path = config_path or self.scenario.source
        np.random.seed(int(self.cfg.get("seed", 7)))
        self.dt = float(self.cfg["simulation_dt_s"])
        self.vehicle_cfg = dict(self.cfg["vehicle"])
        self.speed = float(self.cfg["ego"]["speed_mps"])
        self.vehicle_cfg["speed_mps"] = self.speed

        self.road = RoadGeometry.from_config(self.cfg["road"])
        self.planner = FrenetPlanner(self.cfg, self.road, self.vehicle_cfg)
        self.controller = LTVMPCController(self.cfg["controller"], self.vehicle_cfg, self.dt, controller_backend)
        self.plant = DynamicBicycle(self.vehicle_cfg, self.dt)
        initial = self.cfg["ego"].get("initial", {})
        self.plant.state.x_m = float(initial.get("x_m", 0.0))
        self.plant.state.y_m = float(initial.get("y_m", 0.0))
        self.plant.state.yaw_rad = float(initial.get("yaw_rad", 0.0))
        self.plant.state.lateral_velocity_mps = float(initial.get("v_lat_mps", 0.0))

        self.behavior = HierarchicalSFM(self.cfg.get("behavior", {}), self.vehicle_cfg)
        self.disturbances = DisturbanceModel(self.cfg.get("disturbances", []), self.vehicle_cfg)
        self.disturbance_events_emitted: set[tuple[float, str]] = set()

        sensors = self.cfg.get("sensors", {})
        self.actor_model = ActorModel(
            self.cfg.get("actors", []), self.road,
            ego_width_m=float(self.vehicle_cfg["width_m"]),
            ego_length_m=float(self.vehicle_cfg.get("length_m", 4.5)),
            camera_range_m=float(sensors.get("camera_range_m", 80.0)),
            camera_half_width_m=float(sensors.get("camera_half_width_m", 5.25)),
            radar_range_m=float(sensors.get("radar_range_m", 60.0)),
        )
        self.adas = AdasTimeline(
            self.cfg.get("adas", []),
            {"desired_speed_mps": float(self.cfg["ego"].get("desired_speed_mps", self.speed))},
        )
        self.longitudinal_cfg = dict(self.cfg.get("longitudinal", {}))
        self.reads_lateral_limits = bool(self.longitudinal_cfg.get("reads_lateral_limits", False))
        self.last_limits = None
        self.last_behavior_stimuli = self.adas.at(0.0)

        self.trajectory = self.status = None
        self.lane_counter = self.request_counter = self.state_counter = 0
        self.object_counter = 0
        self.lane_latency_s = float(self.cfg.get("lane_capture_to_send_s", 0.03))
        self.lane_queue: deque[tuple[float, LaneModel]] = deque()
        self.lane_synth = LaneModelSynthesizer(self.road)

        self.chassis_model = ChassisSignalModel(
            self.vehicle_cfg, self.dt,
            road_slope_percent=float(self.cfg.get("road", {}).get("slope_percent", 0.0)),
        )
        self.chassis = self.chassis_model.update(self.plant.state, self.speed)

        state = self.plant.state
        self.latest_camera = self.actor_model.camera_objects(
            0.0, state.x_m, state.y_m, state.yaw_rad, Header(0, self.object_counter),
        )
        self.latest_radar_left, self.latest_radar_right = self.actor_model.radar_objects(
            0.0, state.x_m, state.y_m, state.yaw_rad, self.speed, Header(0, self.object_counter),
        )
        self.latest_lane = self._lane_model(0.0)
        self.latest_state = self._state_estimate(0.0)
        self.latest_behavior = self.behavior.step(self._behavior_inputs(0.0))
        self.latest_requests = self.latest_behavior.requests
        self.longitudinal_plan = self.latest_behavior.longitudinal

        self.timeseries: list[dict] = []
        self.trajectories: list[dict] = []
        self.events: list[dict] = []
        self.last_mode = Function.NONE
        self.previous_ay = 0.0
        self.previous_trajectory = None
        self.planner_tick = 0
        self.last_lateral_state = self.latest_behavior.lateral_state
        self.last_longitudinal_mode = self.longitudinal_plan.mode
        self.ego_trace: list[dict] = []

    @classmethod
    def from_file(cls, path: str | Path, controller_backend: str = "auto") -> "ClosedLoopSimulation":
        from .scenario import load_all

        scenarios = load_all(path)
        return cls(scenarios[0], Path(path).resolve(), controller_backend)

    # ------------------------------------------------------------------
    # Perception / behaviour wiring
    # ------------------------------------------------------------------
    def _publish_due_lane(self, t: float) -> None:
        while self.lane_queue and self.lane_queue[0][0] <= t + 1e-9:
            self.latest_lane = self.lane_queue.popleft()[1]

    def _lane_model(self, t: float) -> LaneModel:
        state = self.plant.state
        previous = getattr(self, "latest_lane", None)
        previous_width = previous.derived_lane_width() if previous is not None else None
        lane_center = self.behavior.current_lane_center_m if hasattr(self, "behavior") else 0.0
        msg = self.lane_synth.build(
            t, self.lane_counter, state.x_m, state.y_m, state.yaw_rad,
            lane_center, previous_width,
        )
        self.lane_counter = (self.lane_counter + 1) % 256
        return msg

    def _driver_torque(self, t: float) -> float:
        return self.adas.at(t).driver_torque_nm

    def _behavior_inputs(self, t: float) -> BehaviorInputs:
        state = self.plant.state
        road_s, road_d_truth, road_yaw, road_kappa = self.road.project(state.x_m, state.y_m)
        heading_error = float(wrap_angle(state.yaw_rad - road_yaw))
        lane = self.latest_lane
        if lane.left is not None and lane.right is not None:
            relative_d = -0.5 * (lane.left.c0 + lane.right.c0)
            road_d = relative_d + self.behavior.current_lane_center_m
        else:
            road_d = road_d_truth

        adas = self.adas.at(t)
        occupancy = self.actor_model.occupancy(t, state.x_m, state.y_m, self.speed)
        disturbance = self.disturbances.sample(t, state.lateral_velocity_mps)
        lane_change_side = adas.lane_change_request
        preview_m = float(self.cfg.get("behavior", {}).get("curve_preview_m", 100.0))
        probe = road_s + np.linspace(0.0, max(preview_m, 1.0), 13)
        _, _, _, kappa_probe = self.road.centerline_at(probe)
        preview_kappa = float(np.max(np.abs(kappa_probe)))
        return BehaviorInputs(
            now_s=t, speed_mps=self.speed, road_d_m=road_d,
            road_heading_error_rad=heading_error,
            lateral_velocity_mps=self.planner.estimated_lateral_velocity_mps,
            yaw_rate_rps=self.chassis.yaw_rate_filtered_radps,
            lane=lane, path_status=self.status,
            hwa_button=adas.hwa, acc_active=adas.acc_active,
            lka_enabled=adas.lka, elk_enabled=adas.elk,
            esc_enabled=adas.esc_enabled, esc_intervening=adas.esc_intervening,
            hands_on=adas.hands_on,
            turn_signal=adas.turn_signal,
            lane_change_request=lane_change_side,
            driver_torque_nm=adas.driver_torque_nm,
            steering_wheel_rate_dps=adas.driver_rate_dps,
            adjacent_hazard_side=Side(occupancy["adjacent_hazard_side"]),
            adjacent_clearance_m=float(occupancy["adjacent_clearance_m"]),
            adjacent_lane_safe=bool(occupancy["adjacent_lane_safe"]),
            rear_approaching_vehicle_ttc_s=float(occupancy["rear_ttc_s"]),
            rear_approaching_vehicle_req_decel_mps2=float(occupancy["rear_req_decel_mps2"]),
            lead_gap_m=float(occupancy["lead_gap_m"]),
            lead_speed_mps=float(occupancy["lead_speed_mps"]),
            obstacle_distance_m=float(occupancy["obstacle_distance_m"]),
            desired_speed_mps=adas.desired_speed_mps,
            clearance_left_m=float(occupancy["clearance_left_m"]),
            clearance_right_m=float(occupancy["clearance_right_m"]),
            hazard_left_present=bool(occupancy["hazard_left_present"]),
            hazard_right_present=bool(occupancy["hazard_right_present"]),
            road_curvature_inv_m=float(road_kappa),
            preview_curvature_inv_m=preview_kappa,
            road_friction_mu=float(disturbance.friction_coefficient),
            lane_expansion_detected=self.road.lane_expanding(road_s),
            lane_change_completed_by_driver=adas.driver_lane_change_completed,
            lateral_v_max_mps=float(self.last_limits.v_max_mps) if self.last_limits is not None else 1000.0,
            reads_lateral_limits=self.reads_lateral_limits,
        )

    def _state_estimate(self, t: float) -> VehicleStateEstimate:
        lateral_velocity = self.planner.estimated_lateral_velocity_mps
        msg = VehicleStateEstimate(
            Header(int(round(t * 1e6)), self.state_counter),
            lateral_velocity, self.planner.estimator.variance,
            math.atan2(lateral_velocity, self.speed) if self.speed > 0.1 else 0.0,
        )
        self.state_counter = (self.state_counter + 1) % 256
        return msg

    @staticmethod
    def _due(step: int, period_s: float, dt_s: float) -> bool:
        return step % max(1, int(round(period_s / dt_s))) == 0

    def _record_trajectory(self, t: float) -> tuple[float, float, float]:
        continuity = (0.0, 0.0, 0.0)
        if self.previous_trajectory is not None:
            a, b = self.previous_trajectory.points, self.trajectory.points
            continuity = (
                abs(a[0].road_d_m - b[0].road_d_m),
                abs(math.atan2(a[0].lateral_velocity_mps, self.speed) - math.atan2(b[0].lateral_velocity_mps, self.speed)),
                abs(a[0].curvature_inv_m - b[0].curvature_inv_m),
            )
        for point in self.trajectory.points:
            gx, gy, _, _ = self.road.frenet_to_global(
                np.array([point.road_s_m]), np.array([point.road_d_m]),
                np.array([point.lateral_velocity_mps]), max(self.speed, 0.1),
            )
            self.trajectories.append({
                "planner_tick": self.planner_tick, "stamp_s": t,
                "time_from_start_s": point.time_from_start_s,
                "global_x_m": float(gx[0]), "global_y_m": float(gy[0]),
                "road_s_m": point.road_s_m, "road_d_m": point.road_d_m,
                "curvature_inv_m": point.curvature_inv_m,
            })
        self.previous_trajectory = self.trajectory
        self.planner_tick += 1
        return continuity

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, output_dir: str | Path | None = None, write_artifacts: bool = True) -> dict:
        n_steps = int(round(float(self.cfg["duration_s"]) / self.dt)) + 1
        continuity = (0.0, 0.0, 0.0)
        planner_time = controller_time = command = 0.0
        for step in range(n_steps):
            t = step * self.dt
            for start, kind in self.disturbances.start_events():
                key = (start, kind)
                if key not in self.disturbance_events_emitted and abs(t - start) < self.dt / 2:
                    self.events.append({"time_s": t, "event": "DISTURBANCE", "detail": kind})
                    self.disturbance_events_emitted.add(key)
            if self._due(step, self.cfg["lane_model_dt_s"], self.dt):
                self.lane_queue.append((t + self.lane_latency_s, self._lane_model(t)))
            self._publish_due_lane(t)
            state = self.plant.state
            if self._due(step, 1.0 / RATES_HZ["camera"], self.dt):
                self.latest_camera = self.actor_model.camera_objects(
                    t, state.x_m, state.y_m, state.yaw_rad,
                    Header(int(round(t * 1e6)), self.object_counter),
                )
                self.object_counter = (self.object_counter + 1) % 256
            if self._due(step, 1.0 / RATES_HZ["radar"], self.dt):
                self.latest_radar_left, self.latest_radar_right = self.actor_model.radar_objects(
                    t, state.x_m, state.y_m, state.yaw_rad, self.speed,
                    Header(int(round(t * 1e6)), self.object_counter),
                )
            if self._due(step, 1.0 / RATES_HZ["yaw_rate"], self.dt):
                self.chassis = self.chassis_model.update(
                    self.plant.state, self.speed, self._driver_torque(t),
                )
            if self._due(step, self.cfg["planner_dt_s"], self.dt):
                self.latest_behavior = self.behavior.step(self._behavior_inputs(t))
                self.latest_requests = self.latest_behavior.requests
                self.longitudinal_plan = self.latest_behavior.longitudinal
                if self.latest_behavior.lateral_state != self.last_lateral_state:
                    self.events.append({"time_s": t, "event": "SFM_LATERAL", "detail": self.latest_behavior.lateral_state.name})
                    self.last_lateral_state = self.latest_behavior.lateral_state
                if self.longitudinal_plan.mode != self.last_longitudinal_mode:
                    self.events.append({"time_s": t, "event": "SFM_LONGITUDINAL", "detail": self.longitudinal_plan.mode.name})
                    self.last_longitudinal_mode = self.longitudinal_plan.mode
            if self._due(step, self.cfg["state_estimator_dt_s"], self.dt):
                self.planner.update_estimation(
                    self.latest_lane, self.speed,
                    self.chassis.yaw_rate_filtered_radps, self.chassis.ay_filtered_mps2,
                )
                self.latest_state = self._state_estimate(t)
            state = self.plant.state
            if self._due(step, self.cfg["planner_dt_s"], self.dt):
                self.trajectory, self.status, limits, diag = self.planner.plan(
                    t, self.latest_lane, self.latest_requests, self.latest_state,
                    state.x_m, state.y_m, state.yaw_rad, self.speed,
                    self.chassis.ay_filtered_mps2,
                )
                self.last_limits = limits
                planner_time = self.status.compute_time_ms
                continuity = self._record_trajectory(t)
                if self.status.active_function != self.last_mode:
                    self.events.append({"time_s": t, "event": "MODE", "detail": self.status.active_function.name})
                    self.last_mode = self.status.active_function
                if diag["completed_function"] != "NONE":
                    self.events.append({"time_s": t, "event": "COMPLETED", "detail": diag["completed_function"]})
            road_s, road_d, road_yaw, road_curvature = self.road.project(state.x_m, state.y_m)
            heading_error = float(wrap_angle(state.yaw_rad - road_yaw))
            command, cdiag = self.controller.command(
                self.trajectory, road_d, heading_error,
                self.planner.estimated_lateral_velocity_mps,
                state.yaw_rate_rps, self.speed,
                self.status.active_function if self.status else Function.NONE,
                road_curvature,
            )
            controller_time = cdiag["solve_time_ms"]
            disturbance = self.disturbances.sample(t, state.lateral_velocity_mps)
            self.plant.step(command, self.speed, disturbance)
            self.speed = max(0.0, self.speed + self.longitudinal_plan.acceleration_mps2 * self.dt)
            ay = self.speed * self.plant.state.yaw_rate_rps
            ay_curve = self.speed**2 * road_curvature
            ay_added_command = self.speed**2 * math.tan(command) / self.vehicle_cfg["wheelbase_m"] - ay_curve
            jerk = (ay - self.previous_ay) / self.dt
            self.previous_ay = ay
            lane_width = self.latest_lane.derived_lane_width()
            margin = lane_width / 2 - self.vehicle_cfg["width_m"] / 2
            occupancy = self.actor_model.occupancy(t, state.x_m, state.y_m, self.speed)
            adas = self.adas.at(t)
            self.timeseries.append({
                "time_s": t, "x_m": self.plant.state.x_m, "y_m": self.plant.state.y_m,
                "yaw_rad": self.plant.state.yaw_rad,
                "speed_mps": self.speed,
                "road_s_m": road_s, "road_d_m": road_d, "heading_error_rad": heading_error,
                "lateral_velocity_mps": self.plant.state.lateral_velocity_mps,
                "estimated_lateral_velocity_mps": self.planner.estimated_lateral_velocity_mps,
                "estimated_offset_m": self.planner.estimated_offset_m,
                "yaw_rate_rps": self.plant.state.yaw_rate_rps,
                "lateral_acceleration_mps2": ay, "lateral_jerk_mps3": jerk,
                "road_lateral_acceleration_mps2": ay_curve,
                "commanded_added_acceleration_mps2": ay_added_command,
                "steering_command_rad": command, "steering_angle_rad": self.plant.state.steering_angle_rad,
                "steering_wheel_rate_limit_radps": 0.8 * self.vehicle_cfg["max_steering_wheel_rate_radps"],
                "speed_kmh": mps_to_kmh(self.speed),
                "yaw_rate_raw_degps": self.chassis.yaw_rate_raw_degps,
                "yaw_rate_filtered_degps": self.chassis.yaw_rate_filtered_degps,
                "ay_raw_mps2": self.chassis.ay_raw_mps2,
                "ay_filtered_mps2": self.chassis.ay_filtered_mps2,
                "ax_raw_mps2": self.chassis.ax_raw_mps2,
                "ax_filtered_mps2": self.chassis.ax_filtered_mps2,
                "swa_deg": rad_to_deg(self.chassis.steering_wheel_angle_rad),
                "sws_degps": radps_to_degps(self.chassis.steering_wheel_rate_radps),
                "driver_torque_nm": self.chassis.driver_steering_torque_nm,
                "standstill": self.chassis.standstill,
                "road_slope_percent": self.chassis.road_slope_percent,
                "odometer_m": self.chassis.odometer_m,
                "wheel_speed_fl_mps": self.chassis.wheel_speed_mps[0],
                "wheel_angular_speed_fl_radps": self.chassis.wheel_angular_speed_radps[0],
                "lane_width_derived_m": lane_width,
                "lane_left_c0_m": self.latest_lane.left.c0 if self.latest_lane.left else float("nan"),
                "lane_right_c0_m": self.latest_lane.right.c0 if self.latest_lane.right else float("nan"),
                "lane_next_left_c0_m": self.latest_lane.next_left.c0 if self.latest_lane.next_left else float("nan"),
                "lane_next_right_c0_m": self.latest_lane.next_right.c0 if self.latest_lane.next_right else float("nan"),
                "camera_object_count": len(self.latest_camera),
                "radar_rear_left_count": len(self.latest_radar_left),
                "radar_rear_right_count": len(self.latest_radar_right),
                "supervisor_limited": cdiag.get("supervisor_limited", False),
                "controller_constraint_active": cdiag.get("controller_constraint_active", False),
                "ay_added_limit_mps2": cdiag.get("ay_added_limit_mps2", 0.0),
                "ay_total_limit_mps2": cdiag.get("ay_total_limit_mps2", 0.0),
                "active_function": int(self.status.active_function) if self.status else 0,
                "sfm_system_state": int(self.latest_behavior.system_state),
                "sfm_lateral_state": int(self.latest_behavior.lateral_state),
                "sfm_longitudinal_mode": int(self.longitudinal_plan.mode),
                "target_speed_mps": self.longitudinal_plan.target_speed_mps,
                "longitudinal_acceleration_command_mps2": self.longitudinal_plan.acceleration_mps2,
                "dlc_left_m": self.latest_behavior.dlc_left_m,
                "dlc_right_m": self.latest_behavior.dlc_right_m,
                "tlc_left_s": self.latest_behavior.tlc_left_s,
                "tlc_right_s": self.latest_behavior.tlc_right_s,
                "path_valid": bool(self.status.path_valid) if self.status else False,
                "planner_time_ms": planner_time if self._due(step, self.cfg["planner_dt_s"], self.dt) else 0.0,
                "controller_time_ms": controller_time, "lane_margin_m": margin,
                "continuity_position_m": continuity[0], "continuity_heading_rad": continuity[1],
                "continuity_curvature_inv_m": continuity[2],
                "crosswind_mps": disturbance.crosswind_mps,
                "aerodynamic_force_n": disturbance.aerodynamic_force_n,
                "aerodynamic_yaw_moment_nm": disturbance.aerodynamic_yaw_moment_nm,
                "road_bank_angle_deg": math.degrees(disturbance.bank_angle_rad),
                "friction_coefficient": disturbance.friction_coefficient,
                # --- scenario v2 channels ---
                "lane_center_m": self.behavior.current_lane_center_m,
                "road_curvature_inv_m": road_curvature,
                "lane_expanding": self.road.lane_expanding(road_s),
                "clearance_left_m": occupancy["clearance_left_m"],
                "clearance_right_m": occupancy["clearance_right_m"],
                "hazard_left_present": occupancy["hazard_left_present"],
                "hazard_right_present": occupancy["hazard_right_present"],
                "obstacle_distance_m": occupancy["obstacle_distance_m"],
                "actor_count": len(self.actor_model.states(t)),
                "adas_hwa": adas.hwa, "adas_lka": adas.lka, "adas_elk": adas.elk,
                "turn_signal": int(adas.turn_signal),
                "lane_change_request": int(adas.lane_change_request),
                "driver_lane_change_completed": adas.driver_lane_change_completed,
                "lateral_v_max_mps": self.last_limits.v_max_mps if self.last_limits else 1000.0,
            })
            self.ego_trace.append({
                "t": t, "x_m": self.plant.state.x_m, "y_m": self.plant.state.y_m,
                "yaw_rad": self.plant.state.yaw_rad, "speed_mps": self.speed,
            })
        summary = self._summary()
        if write_artifacts:
            if output_dir is None:
                output_dir = Path(self.cfg.get("output_dir", f"outputs/{self.scenario.id}"))
            summary = write_outputs(
                Path(output_dir), self.timeseries, self.trajectories, self.events, extra=summary,
            )
            from .ui_export import export_ui_data

            export_ui_data(self, Path(output_dir) / "ui_data.json")
        return summary

    # ------------------------------------------------------------------
    # Summary / expectations
    # ------------------------------------------------------------------
    def evaluate_expectations(self) -> dict:
        ts = self.timeseries
        if not ts:
            return {"passed": False, "checks": {}}
        expectations = self.cfg.get("expectations", {})
        active = {row["active_function"] for row in ts}
        names = {int(Function.HWA): "HWA", int(Function.LKA): "LKA", int(Function.ELK): "ELK"}
        active_names = {names[value] for value in active if value in names}
        margins = [row["lane_margin_m"] - abs(row["road_d_m"]) for row in ts]
        max_ay = max(abs(row["lateral_acceleration_mps2"]) for row in ts)
        max_jerk = max(abs(row["lateral_jerk_mps3"]) for row in ts)
        max_d = max(abs(row["road_d_m"]) for row in ts)
        checks: dict[str, dict] = {}

        must = set(expectations.get("must_activate", []))
        if must:
            checks["must_activate"] = {"passed": must.issubset(active_names), "expected": sorted(must), "observed": sorted(active_names)}
        forbidden = set(expectations.get("must_not_activate", []))
        if forbidden:
            observed = forbidden & active_names
            checks["must_not_activate"] = {"passed": not observed, "forbidden": sorted(forbidden), "observed": sorted(observed)}
        if "dtle_min_m" in expectations:
            minimum = min(margins)
            checks["dtle_min_m"] = {"passed": minimum >= float(expectations["dtle_min_m"]) - 1e-9, "observed": minimum, "limit": float(expectations["dtle_min_m"])}
        if "max_ay_mps2" in expectations:
            checks["max_ay_mps2"] = {"passed": max_ay <= float(expectations["max_ay_mps2"]) + 1e-6, "observed": max_ay, "limit": float(expectations["max_ay_mps2"])}
        if "max_jerk_mps3" in expectations:
            checks["max_jerk_mps3"] = {"passed": max_jerk <= float(expectations["max_jerk_mps3"]) + 1e-6, "observed": max_jerk, "limit": float(expectations["max_jerk_mps3"])}
        if "max_dlc_overshoot_m" in expectations:
            checks["max_dlc_overshoot_m"] = {"passed": max_d <= float(expectations["max_dlc_overshoot_m"]) + 1e-6, "observed": max_d, "limit": float(expectations["max_dlc_overshoot_m"])}
        if "max_abs_lateral_error_m" in expectations:
            checks["max_abs_lateral_error_m"] = {"passed": max_d <= float(expectations["max_abs_lateral_error_m"]) + 1e-6, "observed": max_d, "limit": float(expectations["max_abs_lateral_error_m"])}
        if "max_target_speed_mps" in expectations:
            observed = max(float(row["target_speed_mps"]) for row in ts)
            checks["max_target_speed_mps"] = {"passed": observed <= float(expectations["max_target_speed_mps"]) + 1e-6, "observed": observed, "limit": float(expectations["max_target_speed_mps"])}
        if "min_target_speed_mps" in expectations:
            # Speed adaptation is demonstrated by the target dropping, not by
            # the cruise set speed before the curve is reached.
            observed = min(float(row["target_speed_mps"]) for row in ts)
            checks["min_target_speed_mps"] = {"passed": observed <= float(expectations["min_target_speed_mps"]) + 1e-6, "observed": observed, "limit": float(expectations["min_target_speed_mps"])}
        if "path_valid_at_end" in expectations:
            observed = bool(ts[-1]["path_valid"])
            checks["path_valid_at_end"] = {"passed": observed == bool(expectations["path_valid_at_end"]), "observed": observed}
        window = expectations.get("activation_window_s")
        if window and must:
            start, end = float(window[0]), float(window[1])
            first = next((row["time_s"] for row in ts if names.get(row["active_function"]) in must), None)
            checks["activation_window_s"] = {
                "passed": first is not None and start <= first <= end,
                "observed": first, "window": [start, end],
            }
        return {
            "passed": all(item["passed"] for item in checks.values()),
            "checks": checks,
            "metrics": {
                "min_lane_margin_m": min(margins),
                "max_abs_lateral_error_m": max_d,
                "max_abs_lateral_acceleration_mps2": max_ay,
                "max_abs_lateral_jerk_mps3": max_jerk,
                "active_functions": sorted(active_names),
            },
        }

    def _summary(self) -> dict:
        d = np.array([r["road_d_m"] for r in self.timeseries])
        supervisor = np.array([r["supervisor_limited"] for r in self.timeseries], dtype=float)
        constrained = np.array([r["controller_constraint_active"] for r in self.timeseries], dtype=float)
        return {
            "scenario_id": self.scenario.id,
            "family": self.scenario.family,
            "title": self.scenario.title,
            "seed": self.scenario.seed,
            "samples": len(d),
            "max_abs_lateral_error_m": float(np.max(np.abs(d))) if len(d) else 0.0,
            "supervisor_intervention_fraction": float(np.mean(supervisor)) if len(supervisor) else 0.0,
            "controller_constraint_active_fraction": float(np.mean(constrained)) if len(constrained) else 0.0,
            "expectations": self.evaluate_expectations(),
            "events": self.events,
        }
