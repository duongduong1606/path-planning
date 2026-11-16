from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

from .messages import (
    Function, FunctionRequest, FunctionRequests, Header, InterventionPhase,
    LaneModel, LateralManeuver, LongitudinalMode, LongitudinalPlan, PathStatus,
    RequestState, Side,
)


class SystemState(IntEnum):
    OFF = 0
    STANDBY = 1
    ACTIVE = 2
    DEGRADED = 3
    FAULT = 4


class LateralState(IntEnum):
    OFF = 0
    STANDBY = 1
    HWA_CENTER = 2
    HWA_OFFSET_LEFT = 3
    HWA_OFFSET_RIGHT = 4
    LANE_CHANGE_LEFT = 5
    LANE_CHANGE_RIGHT = 6
    LKA_LEFT = 7
    LKA_RIGHT = 8
    ELK_LEFT = 9
    ELK_RIGHT = 10
    DRIVER_OVERRIDE = 11


@dataclass(slots=True)
class BehaviorInputs:
    now_s: float
    speed_mps: float
    road_d_m: float
    road_heading_error_rad: float
    lateral_velocity_mps: float
    yaw_rate_rps: float
    lane: LaneModel
    path_status: PathStatus | None = None
    hwa_button: bool = False
    acc_active: bool = False
    lka_enabled: bool = True
    elk_enabled: bool = True
    esc_enabled: bool = True
    esc_intervening: bool = False
    hands_on: bool = True
    turn_signal: Side = Side.NONE
    driver_torque_nm: float = 0.0
    steering_wheel_rate_dps: float = 0.0
    lane_change_request: Side = Side.NONE
    adjacent_hazard_side: Side = Side.NONE
    adjacent_clearance_m: float = 10.0
    adjacent_lane_safe: bool = True
    lead_gap_m: float = 1e6
    lead_speed_mps: float = 1e6
    obstacle_distance_m: float = 1e6
    desired_speed_mps: float = 20.0
    inputs_valid: bool = True
    # Benchmark S1-S17 extensions
    clearance_left_m: float = 10.0
    clearance_right_m: float = 10.0
    hazard_left_present: bool = False
    hazard_right_present: bool = False
    rear_approaching_vehicle_ttc_s: float = 99.0
    rear_approaching_vehicle_req_decel_mps2: float = 0.0
    road_curvature_inv_m: float = 0.0
    #: Peak curvature within the look-ahead window, used for early speed
    #: adaptation so the longitudinal planner brakes before a sharp curve.
    preview_curvature_inv_m: float = 0.0
    road_friction_mu: float = 0.9
    lane_expansion_detected: bool = False
    lane_change_completed_by_driver: bool = False
    #: Lateral speed limit published by the planner (S14 ACC exchange).
    lateral_v_max_mps: float = 1000.0
    reads_lateral_limits: bool = False


@dataclass(slots=True)
class BehaviorOutput:
    requests: FunctionRequests
    longitudinal: LongitudinalPlan
    system_state: SystemState
    lateral_state: LateralState
    dlc_left_m: float
    dlc_right_m: float
    tlc_left_s: float
    tlc_right_s: float
    driver_override: bool
    hmi_warning: str = ""


class HierarchicalSFM:
    """Hierarchical State-Flow Model covering all S1-S17 benchmark scenarios."""

    def __init__(self, cfg: dict, vehicle_cfg: dict):
        self.cfg = cfg
        self.vehicle = vehicle_cfg
        self.counter = 0
        self.system_state = SystemState.STANDBY
        self.lateral_state = LateralState.STANDBY
        self.longitudinal_mode = LongitudinalMode.CRUISE
        self.active_intervention = Function.NONE
        self.active_side = Side.NONE
        self.cooldown_until = {Side.LEFT: 0.0, Side.RIGHT: 0.0}
        self.lka_confirm_frames = {Side.LEFT: 0, Side.RIGHT: 0}
        self.override_since: float | None = None
        self.current_lane_center_m = 0.0
        self.lane_change_target_m: float | None = None
        self.lane_change_side = Side.NONE
        self._driver_lane_change_applied = False

    def step(self, inputs: BehaviorInputs) -> BehaviorOutput:
        now = inputs.now_s
        dlc_l, dlc_r, tlc_l, tlc_r, v_front = self._lane_metrics(inputs)
        self._consume_path_status(inputs.path_status, now, dlc_l, dlc_r)
        override = self._driver_override(inputs, now)
        lane_available = self._lane_available(inputs.lane)

        if not inputs.inputs_valid:
            self.system_state = SystemState.FAULT
        elif not lane_available:
            self.system_state = SystemState.DEGRADED
        elif inputs.hwa_button or inputs.lka_enabled or inputs.elk_enabled:
            self.system_state = SystemState.ACTIVE
        else:
            self.system_state = SystemState.OFF

        requests = self._lateral_region(inputs, dlc_l, dlc_r, tlc_l, tlc_r, v_front, override, lane_available)
        longitudinal = self._longitudinal_region(inputs)

        header = Header(int(round(now * 1e6)), self.counter, inputs.inputs_valid, "")
        self.counter = (self.counter + 1) % 256
        requests.header = header
        requests.driver_override = override
        longitudinal.header = Header(header.stamp_us, header.counter, header.valid, "")

        warning = "DRIVER_OVERRIDE" if override else ("LANE_INPUT_UNAVAILABLE" if not lane_available else "")
        return BehaviorOutput(
            requests, longitudinal, self.system_state, self.lateral_state,
            dlc_l, dlc_r, tlc_l, tlc_r, override, warning,
        )

    def _lane_metrics(self, inputs: BehaviorInputs) -> tuple[float, float, float, float, float]:
        # Lane width is not a published signal: derive it from the two line offsets,
        # falling back to the remembered width when only one line is visible.
        nominal_width = inputs.lane.derived_lane_width()
        # S6b: When lane expansion is detected, clamp width to nominal highway lane width (3.5 m)
        if inputs.lane_expansion_detected or nominal_width > 4.2:
            effective_width = float(self.cfg.get("nominal_lane_width_m", 3.5))
        else:
            effective_width = nominal_width

        lane_margin = effective_width / 2.0 - self.vehicle["width_m"] / 2.0
        relative_d = inputs.road_d_m - self.current_lane_center_m
        # Sheet frame: relative_d > 0 means the ego sits right of the lane centre,
        # so it is closer to the right line and further from the left one.
        dlc_left = lane_margin + relative_d
        dlc_right = lane_margin - relative_d
        v_front = inputs.lateral_velocity_mps + inputs.speed_mps * inputs.road_heading_error_rad

        # S10: Strict drift rate check - if below 0.05 m/s, vehicle is holding line, not drifting
        # v_front > 0 drifts right in the sheet frame.
        left_rate, right_rate = max(-v_front, 0.0), max(v_front, 0.0)
        tlc_left = dlc_left / left_rate if left_rate >= 0.05 else 99.0
        tlc_right = dlc_right / right_rate if right_rate >= 0.05 else 99.0
        return dlc_left, dlc_right, tlc_left, tlc_right, v_front

    @staticmethod
    def _lane_available(lane: LaneModel) -> bool:
        lines = [line for line in (lane.left, lane.right) if line is not None]
        return bool(lines) and any(line.confidence >= 0.6 and line.view_range_end >= 20.0 for line in lines)

    def _driver_override(self, inputs: BehaviorInputs, now: float) -> bool:
        # Deliberate steering wheel torque override
        deliberate_torque = abs(inputs.driver_torque_nm) >= 3.0
        self.override_since = now if deliberate_torque and self.override_since is None else self.override_since
        if not deliberate_torque:
            self.override_since = None
        torque_override = self.override_since is not None and (now - self.override_since >= 0.1)

        # S8c: Rapid evasive steering maneuver without turn signal (dodging pothole/obstacle)
        # Immediate release / override so LKA/ELK does not fight evasive driver
        evasive_vlat = float(self.cfg.get("evasive_vlat_mps", 1.0))
        evasive_action = (
            abs(inputs.lateral_velocity_mps) >= evasive_vlat
            or abs(inputs.steering_wheel_rate_dps) >= 150.0
        ) and (inputs.turn_signal == Side.NONE)

        return torque_override or evasive_action

    def _consume_path_status(self, status: PathStatus | None, now: float, dlc_l: float, dlc_r: float) -> None:
        if status is None or status.intervention_phase != InterventionPhase.COMPLETED:
            return
        if self.active_side != Side.NONE:
            self.cooldown_until[self.active_side] = now + float(self.cfg.get("reactivation_delay_s", 2.0))
        self.active_intervention = Function.NONE
        self.active_side = Side.NONE

    def _function_state(self, enabled: bool, available: bool, request: bool) -> RequestState:
        if not enabled:
            return RequestState.OFF
        if not available:
            return RequestState.UNAVAILABLE
        return RequestState.REQUEST if request else RequestState.STANDBY

    def _lateral_region(
        self, inputs: BehaviorInputs, dlc_l: float, dlc_r: float, tlc_l: float, tlc_r: float,
        v_front: float, override: bool, lane_available: bool,
    ) -> FunctionRequests:
        now = inputs.now_s
        speed_kph = inputs.speed_mps * 3.6
        relative_d = inputs.road_d_m - self.current_lane_center_m

        # S2 & S11: HWA entry and availability
        # HWA operates down to 0 km/h (S11 Stop & Go support on highway)
        hwa_entry_ok = (
            min(dlc_l, dlc_r) >= float(self.cfg.get("hwa_min_dlc_m", 0.2))
            and abs(relative_d) <= float(self.cfg.get("hwa_max_center_offset_m", 0.5))
            and abs(inputs.road_heading_error_rad) <= math.radians(float(self.cfg.get("hwa_max_heading_error_deg", 2.0)))
            and inputs.hands_on
        )
        hwa_available = lane_available and 0.0 <= speed_kph <= 130.0 and inputs.acc_active and (
            hwa_entry_ok or self.lateral_state not in (LateralState.OFF, LateralState.STANDBY)
        )

        # LKA and ELK operate in the Euro NCAP 65-130 km/h band
        intervention_available = lane_available and 65.0 <= speed_kph <= 130.0 and inputs.esc_enabled and not inputs.esc_intervening
        active_intervention_available = lane_available and inputs.esc_enabled and not inputs.esc_intervening

        # S17: Driver lane change during HWA
        if override:
            if inputs.lane_change_completed_by_driver and not self._driver_lane_change_applied:
                # Moving to the left lane lowers the lateral coordinate in the
                # sheet frame (positive y is to the right).  Latch the shift so
                # a held flag cannot keep moving the lane centre.
                if inputs.turn_signal == Side.LEFT:
                    self.current_lane_center_m -= inputs.lane.derived_lane_width()
                elif inputs.turn_signal == Side.RIGHT:
                    self.current_lane_center_m += inputs.lane.derived_lane_width()
                self._driver_lane_change_applied = True
            elif not inputs.lane_change_completed_by_driver:
                self._driver_lane_change_applied = False
            self.lateral_state = LateralState.DRIVER_OVERRIDE
            self.active_intervention = Function.NONE
            requests = [
                FunctionRequest(Function.HWA, self._function_state(inputs.hwa_button, hwa_available, False), cancel_reason="DRIVER_OVERRIDE"),
                FunctionRequest(Function.LKA, self._function_state(inputs.lka_enabled, intervention_available, False), cancel_reason="DRIVER_OVERRIDE"),
                FunctionRequest(Function.ELK, self._function_state(inputs.elk_enabled, intervention_available, False), cancel_reason="DRIVER_OVERRIDE"),
            ]
            return FunctionRequests(Header(0, 0), requests, override)

        side = Side.LEFT if tlc_l < tlc_r else Side.RIGHT
        tlc = min(tlc_l, tlc_r)
        dlc = dlc_l if side == Side.LEFT else dlc_r
        # Positive v_front is a drift to the right in the sheet frame.
        toward_rate = max(-v_front, 0.0) if side == Side.LEFT else max(v_front, 0.0)

        line = inputs.lane.left if side == Side.LEFT else inputs.lane.right
        line_type = line.line_type if line else "LOST"
        hard_boundary = bool(line and (not line.crossable or line_type in ("SOLID", "DOUBLE", "ROAD_EDGE", "BARRIER")))

        # Check hazards for both left and right (S3, S3b, S7, S8, S15)
        hazard_on_side = (
            (inputs.adjacent_hazard_side == side and not inputs.adjacent_lane_safe)
            or (side == Side.LEFT and inputs.hazard_left_present)
            or (side == Side.RIGHT and inputs.hazard_right_present)
        )

        # S8b: Lane change risk assessment with rear approaching vehicle
        # a_req = (delta_v)^2 / (2 * distance)
        rear_vehicle_hazardous = (
            inputs.rear_approaching_vehicle_ttc_s <= 1.5
            or inputs.rear_approaching_vehicle_req_decel_mps2 > 3.0
        )
        if inputs.lane_change_request == side and rear_vehicle_hazardous:
            hazard_on_side = True

        elk_threshold = float(self.cfg.get("elk_object_tlc_s", 0.8) if hazard_on_side else self.cfg.get("elk_boundary_tlc_s", 0.6))

        # S8 & S8b: Turn signal suppresses boundary ELK/LKA, but does NOT suppress collision ELK!
        turn_signal_suppresses_elk = (inputs.turn_signal == side) and not hazard_on_side

        elk_condition = (
            inputs.elk_enabled and intervention_available and now >= self.cooldown_until[side]
            and 0.05 <= toward_rate <= 1.0 and tlc <= elk_threshold
            and (hard_boundary or hazard_on_side)
            and not turn_signal_suppresses_elk
        )

        intentional_lane_change = (
            (inputs.lane_change_request != Side.NONE or self.lane_change_target_m is not None)
            and not rear_vehicle_hazardous
        )

        lka_raw = (
            inputs.lka_enabled and intervention_available and now >= self.cooldown_until[side]
            and not intentional_lane_change
            and 0.05 <= toward_rate <= 1.0
            and (tlc <= float(self.cfg.get("lka_tlc_s", 1.0)) or dlc <= 0.1)
            and inputs.turn_signal != side
        )
        self.lka_confirm_frames[side] = self.lka_confirm_frames[side] + 1 if lka_raw else 0
        lka_condition = self.lka_confirm_frames[side] >= int(self.cfg.get("lka_confirm_frames", 2))

        if self.active_intervention == Function.ELK:
            elk_condition, side = True, self.active_side
        elif self.active_intervention == Function.LKA:
            lka_condition, side = True, self.active_side

        hwa_target = self.current_lane_center_m
        maneuver = LateralManeuver.LANE_CENTER
        hwa_reason = "LANE_CENTER"

        if elk_condition:
            self.active_intervention, self.active_side = Function.ELK, side
            self.lateral_state = LateralState.ELK_LEFT if side == Side.LEFT else LateralState.ELK_RIGHT
        elif lka_condition:
            self.active_intervention, self.active_side = Function.LKA, side
            self.lateral_state = LateralState.LKA_LEFT if side == Side.LEFT else LateralState.LKA_RIGHT
        elif intentional_lane_change and inputs.adjacent_lane_safe and hwa_available:
            if self.lane_change_target_m is None:
                self.lane_change_side = inputs.lane_change_request
                # Left lane = negative lateral coordinate in the sheet frame.
                direction = -1.0 if self.lane_change_side == Side.LEFT else 1.0
                self.lane_change_target_m = self.current_lane_center_m + direction * inputs.lane.derived_lane_width()
            hwa_target = self.lane_change_target_m
            direction = -1.0 if self.lane_change_side == Side.LEFT else 1.0
            maneuver = LateralManeuver.LANE_CHANGE_LEFT if direction < 0 else LateralManeuver.LANE_CHANGE_RIGHT
            self.lateral_state = LateralState.LANE_CHANGE_LEFT if direction < 0 else LateralState.LANE_CHANGE_RIGHT
            hwa_reason = "LANE_CHANGE"
            lane_relative_rate = inputs.lateral_velocity_mps + inputs.speed_mps * inputs.road_heading_error_rad
            if (
                abs(inputs.road_d_m - hwa_target) <= 0.15
                and abs(lane_relative_rate) <= 0.1
                and abs(inputs.road_heading_error_rad) <= math.radians(0.5)
            ):
                self.current_lane_center_m = hwa_target
                self.lane_change_target_m = None
                self.lane_change_side = Side.NONE
                maneuver = LateralManeuver.LANE_CENTER
                self.lateral_state = LateralState.HWA_CENTER
                hwa_reason = "LANE_CHANGE_COMPLETED"
        elif hwa_available:
            # S3, S3b, S15: Offset reasoning for single or dual-sided obstacles & narrow lanes
            clr_left = min(inputs.adjacent_clearance_m if inputs.adjacent_hazard_side == Side.LEFT else 10.0, inputs.clearance_left_m)
            clr_right = min(inputs.adjacent_clearance_m if inputs.adjacent_hazard_side == Side.RIGHT else 10.0, inputs.clearance_right_m)
            thresh = float(self.cfg.get("offset_trigger_clearance_m", 1.2))
            hazard_left = clr_left < thresh or inputs.hazard_left_present
            hazard_right = clr_right < thresh or inputs.hazard_right_present

            if hazard_left and hazard_right:
                # S15: Hazards on both sides - never bias toward closer object; enforce DLC >= 0.3m both sides
                self.lateral_state = LateralState.HWA_CENTER
                hwa_target = self.current_lane_center_m
                maneuver = LateralManeuver.LANE_CENTER
                hwa_reason = "DUAL_HAZARD_STAY_CENTERED"
            elif hazard_left or hazard_right:
                # S3 & S3b: Offset away from obstacle, respecting opposite DLC >= 0.3m
                # Moving away from a left hazard means a positive (right) offset.
                direction = 1.0 if hazard_left else -1.0
                nominal_offset = float(self.cfg.get("offset_m", 0.25))
                # Check opposite side DLC buffer
                opposite_dlc = dlc_l if direction > 0 else dlc_r
                safe_offset = max(0.0, min(nominal_offset, opposite_dlc - 0.30))
                if safe_offset > 0.05:
                    hwa_target = self.current_lane_center_m + direction * safe_offset
                    maneuver = LateralManeuver.OFFSET_RIGHT if direction > 0 else LateralManeuver.OFFSET_LEFT
                    self.lateral_state = LateralState.HWA_OFFSET_RIGHT if direction > 0 else LateralState.HWA_OFFSET_LEFT
                    hwa_reason = "ADJACENT_OBJECT_OFFSET"
                else:
                    self.lateral_state = LateralState.HWA_CENTER
                    hwa_target = self.current_lane_center_m
                    maneuver = LateralManeuver.LANE_CENTER
                    hwa_reason = "NARROW_LANE_MAINTAIN_DLC"
            elif inputs.hwa_button:
                self.lateral_state = LateralState.HWA_CENTER
            else:
                self.lateral_state = LateralState.STANDBY
        else:
            self.lateral_state = LateralState.STANDBY

        requests: list[FunctionRequest] = []
        requests.append(FunctionRequest(
            Function.HWA, self._function_state(inputs.hwa_button, hwa_available, inputs.hwa_button and hwa_available),
            target_offset_m=hwa_target, maneuver=maneuver, reason=hwa_reason,
        ))
        requests.append(FunctionRequest(
            Function.LKA, self._function_state(
                inputs.lka_enabled,
                active_intervention_available if self.active_intervention == Function.LKA else intervention_available,
                lka_condition,
            ),
            side=side if lka_condition else Side.NONE, tlc_s=tlc,
            target_offset_m=self.current_lane_center_m,
            maneuver=LateralManeuver.RECOVERY_RIGHT if side == Side.RIGHT else LateralManeuver.RECOVERY_LEFT,
            reason="LANE_DEPARTURE",
        ))
        elk_reason = "ROAD_EDGE" if line_type in ("ROAD_EDGE", "BARRIER") else ("SOLID_LINE" if hard_boundary else "ADJACENT_VEHICLE")
        requests.append(FunctionRequest(
            Function.ELK, self._function_state(
                inputs.elk_enabled,
                active_intervention_available if self.active_intervention == Function.ELK else intervention_available,
                elk_condition,
            ),
            side=side if elk_condition else Side.NONE, tlc_s=tlc,
            target_offset_m=self.current_lane_center_m,
            maneuver=LateralManeuver.RECOVERY_RIGHT if side == Side.RIGHT else LateralManeuver.RECOVERY_LEFT,
            reason=elk_reason, elk_reason=elk_reason,
        ))
        return FunctionRequests(Header(0, 0), requests, override)

    def _longitudinal_region(self, inputs: BehaviorInputs) -> LongitudinalPlan:
        speed = inputs.speed_mps
        target = inputs.desired_speed_mps

        # S1, S1b, S14: Lateral-longitudinal coordination in sharp curves.
        # v_curve_max = sqrt(ays_max / |kappa|).  Use the peak curvature inside
        # the look-ahead window so the vehicle brakes *before* the curve rather
        # than only once it is already inside it.
        ays_max = float(self.cfg.get("ays_max_mps2", 2.5))
        kappa = max(abs(inputs.road_curvature_inv_m), abs(inputs.preview_curvature_inv_m))
        if kappa > 1e-6:
            eff_mu = max(0.2, min(1.0, inputs.road_friction_mu))
            curve_speed_limit = math.sqrt((eff_mu * ays_max) / kappa)
            target = min(target, curve_speed_limit)
        # S14: when the longitudinal planner subscribes to the lateral limit
        # exchange, the speed target cannot exceed the planner's v_max.
        if inputs.reads_lateral_limits:
            target = min(target, max(0.0, inputs.lateral_v_max_mps))

        desired_gap = max(float(self.cfg.get("minimum_gap_m", 5.0)), speed * float(self.cfg.get("time_gap_s", 1.8)))
        stopping_distance = speed * speed / (2.0 * abs(float(self.cfg.get("emergency_decel_mps2", -6.0))))
        reason = "SET_SPEED"

        if inputs.obstacle_distance_m <= stopping_distance + 5.0:
            mode, accel, reason = LongitudinalMode.EMERGENCY_BRAKE, float(self.cfg.get("emergency_decel_mps2", -6.0)), "IMMINENT_COLLISION"
        elif inputs.lead_gap_m < desired_gap:
            closing = max(0.0, speed - inputs.lead_speed_mps)
            accel = max(float(self.cfg.get("comfortable_decel_mps2", -2.5)), -0.35 * (desired_gap - inputs.lead_gap_m) - 0.5 * closing)
            mode, reason = LongitudinalMode.BRAKE, "LEAD_TOO_CLOSE"
        elif inputs.lead_gap_m < 1e5:
            target = min(target, inputs.lead_speed_mps)
            accel = max(-1.5, min(1.0, 0.5 * (target - speed)))
            mode, reason = LongitudinalMode.FOLLOW, "FOLLOW_LEAD"
        elif speed < target - 0.2:
            mode, accel = LongitudinalMode.ACCELERATE, min(float(self.cfg.get("max_accel_mps2", 1.5)), 0.6 * (target - speed))
        elif speed > target + 0.2:
            mode, accel = LongitudinalMode.BRAKE, max(float(self.cfg.get("comfortable_decel_mps2", -2.5)), 0.6 * (target - speed))
        else:
            mode, accel = LongitudinalMode.CRUISE, 0.0

        self.longitudinal_mode = mode
        return LongitudinalPlan(Header(0, 0), mode, target, float(accel), inputs.lead_gap_m, reason)
