import json
import math
from pathlib import Path

import pytest

from l2_path_planner.behavior import (
    BehaviorInputs, HierarchicalSFM, LateralState, SystemState,
)
from l2_path_planner.messages import (
    Function, Header, InterventionPhase, LaneLine, LaneModel, LateralManeuver,
    LongitudinalMode, PathStatus, RequestState, Side,
)

ROOT = Path(__file__).parents[1]


def make_lane(right_type="DASHED", left_type="DASHED", width=3.5):
    return LaneModel(
        Header(0, 0),
        # Sheet frame: the left line has the smaller (more negative) offset.
        LaneLine(-width / 2.0, 0, 0, 0, line_class=left_type),
        LaneLine(width / 2.0, 0, 0, 0, line_class=right_type),
        width,
    )


def make_sfm():
    cfg = json.loads((ROOT / "configs/baseline.json").read_text())
    return HierarchicalSFM(cfg["behavior"], cfg["vehicle"])


def default_inputs(**kwargs):
    base = dict(
        now_s=1.0, speed_mps=20.0, road_d_m=0.0, road_heading_error_rad=0.0,
        lateral_velocity_mps=0.0, yaw_rate_rps=0.0, lane=make_lane(),
        hwa_button=True, acc_active=True, desired_speed_mps=20.0,
        hands_on=True, inputs_valid=True,
    )
    base.update(kwargs)
    return BehaviorInputs(**base)


# =========================================================================
# S1 & S1b: HWA curve following & sharp curve speed adaptation
# =========================================================================
def test_s1_and_s1b_hwa_curve_and_curvature_speed_adaptation():
    sfm = make_sfm()
    # S1: Normal curve R = 1000m (kappa = 0.001) at 130 km/h (36.1 m/s)
    out_s1 = sfm.step(default_inputs(speed_mps=36.1, road_curvature_inv_m=0.001, desired_speed_mps=36.1))
    assert out_s1.requests.request_for(Function.HWA).state == RequestState.REQUEST
    assert out_s1.lateral_state == LateralState.HWA_CENTER

    # S1b: Sharp curve R = 200m (kappa = 0.005) -> speed adaptation kicks in
    # ays_max = 2.5 m/s^2, kappa = 0.005 -> v_limit = sqrt(2.5 / 0.005) = 22.36 m/s
    out_s1b = sfm.step(default_inputs(speed_mps=25.0, road_curvature_inv_m=0.005, desired_speed_mps=25.0))
    assert out_s1b.longitudinal.target_speed_mps < 23.0
    assert out_s1b.longitudinal.mode == LongitudinalMode.BRAKE


# =========================================================================
# S2: HWA activation with initial lateral offset & heading error
# =========================================================================
def test_s2_hwa_offset_entry():
    sfm = make_sfm()
    # Initial offset d = 0.4 m, heading error = 1 deg
    inp = default_inputs(road_d_m=0.4, road_heading_error_rad=math.radians(1.0))
    out = sfm.step(inp)
    assert out.requests.request_for(Function.HWA).state == RequestState.REQUEST
    assert out.lateral_state == LateralState.HWA_CENTER


# =========================================================================
# S3, S3b, S15: Offset with truck, rear overtaking, dual-side hazards & narrow lanes
# =========================================================================
def test_s3_s3b_s15_truck_offset_and_narrow_lane_scenarios():
    sfm = make_sfm()
    # S3: Single truck on left -> offsets right by <= 0.3 m
    out_s3 = sfm.step(default_inputs(adjacent_hazard_side=Side.LEFT, adjacent_clearance_m=0.8))
    hwa_s3 = out_s3.requests.request_for(Function.HWA)
    assert hwa_s3.maneuver == LateralManeuver.OFFSET_RIGHT
    assert 0.15 <= hwa_s3.target_offset_m <= 0.30

    # S3b: Narrow lane (3.0 m) with truck on left -> offset clamped so right DLC >= 0.3 m
    narrow_lane = make_lane(width=3.0)
    out_s3b = sfm.step(default_inputs(lane=narrow_lane, adjacent_hazard_side=Side.LEFT, adjacent_clearance_m=0.8))
    hwa_s3b = out_s3b.requests.request_for(Function.HWA)
    # Target offset must preserve DLC >= 0.3m on the right
    lane_margin_right = 3.0 / 2.0 - 1.9 / 2.0  # 0.55 m
    assert (lane_margin_right - hwa_s3b.target_offset_m) >= 0.29

    # S15: Obstacles on BOTH sides -> MUST STAY CENTERED (never push toward either obstacle)
    out_s15 = sfm.step(default_inputs(
        hazard_left_present=True, clearance_left_m=0.8,
        hazard_right_present=True, clearance_right_m=0.8,
    ))
    hwa_s15 = out_s15.requests.request_for(Function.HWA)
    assert hwa_s15.maneuver == LateralManeuver.LANE_CENTER
    assert hwa_s15.target_offset_m == 0.0
    assert hwa_s15.reason == "DUAL_HAZARD_STAY_CENTERED"


# =========================================================================
# S4 & S4b: LKA lane departure on straight and curved roads
# =========================================================================
def test_s4_and_s4b_lka_departure():
    sfm = make_sfm()
    # Frame 1: drift right (dlc low, tlc low)
    inp1 = default_inputs(road_d_m=0.72, lateral_velocity_mps=0.25)
    out1 = sfm.step(inp1)
    assert out1.requests.request_for(Function.LKA).state != RequestState.REQUEST  # Debounce

    # Frame 2: confirmed departure
    inp2 = default_inputs(now_s=1.025, road_d_m=0.72, lateral_velocity_mps=0.25)
    out2 = sfm.step(inp2)
    assert out2.requests.request_for(Function.LKA).state == RequestState.REQUEST
    assert out2.lateral_state == LateralState.LKA_RIGHT

    # S4b: Curved road drift outward
    sfm_curve = make_sfm()
    sfm_curve.step(default_inputs(road_d_m=0.72, lateral_velocity_mps=0.25, road_curvature_inv_m=0.002))
    out_curve = sfm_curve.step(default_inputs(now_s=1.025, road_d_m=0.72, lateral_velocity_mps=0.25, road_curvature_inv_m=0.002))
    assert out_curve.requests.request_for(Function.LKA).state == RequestState.REQUEST


# =========================================================================
# S5 & S6: ELK for solid line and road edge / barrier
# =========================================================================
def test_s5_and_s6_elk_solid_line_and_road_edge():
    # S5: Solid line
    sfm_s5 = make_sfm()
    lane_solid = make_lane(right_type="SOLID")
    out_s5 = sfm_s5.step(default_inputs(lane=lane_solid, road_d_m=0.75, lateral_velocity_mps=0.35))
    assert out_s5.requests.request_for(Function.ELK).state == RequestState.REQUEST
    assert out_s5.lateral_state == LateralState.ELK_RIGHT

    # S6: Road edge / barrier
    sfm_s6 = make_sfm()
    lane_edge = make_lane(right_type="ROAD_EDGE")
    out_s6 = sfm_s6.step(default_inputs(lane=lane_edge, road_d_m=0.75, lateral_velocity_mps=0.35))
    assert out_s6.requests.request_for(Function.ELK).state == RequestState.REQUEST
    assert out_s6.requests.request_for(Function.ELK).elk_reason == "ROAD_EDGE"


# =========================================================================
# S6b: Abnormal lane expansion (exit ramp / shoulder)
# =========================================================================
def test_s6b_abnormal_lane_expansion():
    sfm = make_sfm()
    # Lane suddenly expands to 5.0m
    expanded_lane = make_lane(width=5.0)
    out = sfm.step(default_inputs(lane=expanded_lane, lane_expansion_detected=True, road_d_m=0.0))
    # Path center stays clamped to nominal 0.0 without drifting into expanded shoulder
    hwa = out.requests.request_for(Function.HWA)
    assert abs(hwa.target_offset_m) <= 0.10


# =========================================================================
# S7 & S8: ELK oncoming and overtaking (blind spot risk even with turn signal)
# =========================================================================
def test_s7_and_s8_elk_oncoming_and_blind_spot():
    # S7: Oncoming vehicle risk
    sfm_s7 = make_sfm()
    out_s7 = sfm_s7.step(default_inputs(
        road_d_m=-0.75, lateral_velocity_mps=-0.35,
        adjacent_hazard_side=Side.LEFT, adjacent_lane_safe=False,
    ))
    assert out_s7.requests.request_for(Function.ELK).state == RequestState.REQUEST

    # S8: Blind spot collision risk WITH TURN SIGNAL ON -> MUST STILL INTERVENE
    sfm_s8 = make_sfm()
    out_s8 = sfm_s8.step(default_inputs(
        road_d_m=-0.72, lateral_velocity_mps=-0.35,
        turn_signal=Side.LEFT,  # Driver signalling left!
        adjacent_hazard_side=Side.LEFT, adjacent_lane_safe=False,  # But car in blind spot
    ))
    assert out_s8.requests.request_for(Function.ELK).state == RequestState.REQUEST
    assert out_s8.lateral_state == LateralState.ELK_LEFT


# =========================================================================
# S8b: Lane change risk assessment with rear approaching vehicle
# =========================================================================
def test_s8b_lane_change_rear_approaching_vehicle():
    sfm = make_sfm()
    # Safe rear vehicle: TTC = 2.5s, a_req = 0.24 m/s^2 -> Lane change allowed
    out_safe = sfm.step(default_inputs(
        lane_change_request=Side.LEFT,
        rear_approaching_vehicle_ttc_s=2.5,
        rear_approaching_vehicle_req_decel_mps2=0.24,
    ))
    assert out_safe.requests.request_for(Function.HWA).maneuver == LateralManeuver.LANE_CHANGE_LEFT
    assert out_safe.requests.request_for(Function.ELK).state != RequestState.REQUEST

    # Dangerous rear vehicle: TTC = 1.2s -> Lane change blocked & ELK ready
    sfm_danger = make_sfm()
    out_danger = sfm_danger.step(default_inputs(
        road_d_m=-0.72, lateral_velocity_mps=-0.35,
        lane_change_request=Side.LEFT,
        rear_approaching_vehicle_ttc_s=1.2,
        rear_approaching_vehicle_req_decel_mps2=3.5,
    ))
    assert out_danger.requests.request_for(Function.ELK).state == RequestState.REQUEST


# =========================================================================
# S8c: Evasive driver steering (rapid swerve without turn signal)
# =========================================================================
def test_s8c_evasive_driver_steering():
    sfm = make_sfm()
    # Driver rapidly swerves (v_lat = 1.2 m/s) to dodge obstacle without turn signal
    out = sfm.step(default_inputs(road_d_m=0.5, lateral_velocity_mps=1.2))
    # Must immediately yield to driver (DRIVER_OVERRIDE), not fight driver
    assert out.driver_override is True
    assert out.lateral_state == LateralState.DRIVER_OVERRIDE
    assert out.requests.request_for(Function.LKA).state != RequestState.REQUEST


# =========================================================================
# S9: Smooth mode transitions (LKA -> ELK -> HWA)
# =========================================================================
def test_s9_mode_transitions():
    sfm = make_sfm()
    # Trigger LKA
    sfm.step(default_inputs(road_d_m=0.72, lateral_velocity_mps=0.25))
    out_lka = sfm.step(default_inputs(now_s=1.025, road_d_m=0.72, lateral_velocity_mps=0.25))
    assert out_lka.lateral_state == LateralState.LKA_RIGHT

    # Escalate to ELK mid-intervention
    lane_solid = make_lane(right_type="SOLID")
    out_elk = sfm.step(default_inputs(now_s=1.05, lane=lane_solid, road_d_m=0.80, lateral_velocity_mps=0.35))
    assert out_elk.lateral_state == LateralState.ELK_RIGHT

    # Complete ELK and transition back to HWA
    status_done = PathStatus(
        Header(int(round(1.075 * 1e6)), 1), True, Function.ELK,
        InterventionPhase.COMPLETED, False, 25.0,
    )
    out_done = sfm.step(default_inputs(now_s=1.075, path_status=status_done, road_d_m=0.0, lateral_velocity_mps=0.0))
    assert out_done.lateral_state == LateralState.HWA_CENTER


# =========================================================================
# S10: Driving near line without drift (Zero False Positive)
# =========================================================================
def test_s10_near_boundary_without_drift():
    sfm = make_sfm()
    # DLC = 0.2m, but lateral velocity = 0.0 m/s (cutting curve or hugging edge)
    out = sfm.step(default_inputs(road_d_m=0.60, lateral_velocity_mps=0.0))
    assert out.requests.request_for(Function.LKA).state != RequestState.REQUEST
    assert out.requests.request_for(Function.ELK).state != RequestState.REQUEST


# =========================================================================
# S11: Highway Traffic Jam / Stop & Go (0 - 60 km/h)
# =========================================================================
def test_s11_stop_and_go_speed_band():
    sfm = make_sfm()
    # Standstill at 0 km/h with ACC active
    out_stop = sfm.step(default_inputs(speed_mps=0.0, desired_speed_mps=0.0))
    # HWA is ACTIVE and ready in stop & go
    assert out_stop.requests.request_for(Function.HWA).state == RequestState.REQUEST
    assert out_stop.lateral_state == LateralState.HWA_CENTER
    # LKA and ELK must be UNAVAILABLE at 0 km/h (minimum 65 km/h)
    assert out_stop.requests.request_for(Function.LKA).state == RequestState.UNAVAILABLE
    assert out_stop.requests.request_for(Function.ELK).state == RequestState.UNAVAILABLE


# =========================================================================
# S12 & S13: Friction and actuator robustness
# =========================================================================
def test_s12_s13_friction_and_delay_robustness():
    sfm = make_sfm()
    # Low friction mu = 0.5 in curve
    out = sfm.step(default_inputs(
        speed_mps=20.0, road_curvature_inv_m=0.003,
        road_friction_mu=0.5, desired_speed_mps=20.0,
    ))
    # Lower target speed to maintain safety margin in rain
    assert out.longitudinal.target_speed_mps < 21.0


# =========================================================================
# S14: Lateral-Longitudinal constraint exchange
# =========================================================================
def test_s14_lateral_longitudinal_coordination():
    sfm = make_sfm()
    out = sfm.step(default_inputs(speed_mps=25.0, road_curvature_inv_m=0.004, desired_speed_mps=25.0))
    assert out.longitudinal.mode == LongitudinalMode.BRAKE
    assert out.longitudinal.acceleration_mps2 < 0.0


# =========================================================================
# S16: Repeated intervention cooldown & re-arming
# =========================================================================
def test_s16_repeated_interventions():
    sfm = make_sfm()
    # First LKA completed
    status_done = PathStatus(
        Header(1000000, 1), True, Function.LKA,
        InterventionPhase.COMPLETED, False, 25.0,
    )
    sfm.step(default_inputs(now_s=1.0, path_status=status_done, road_d_m=0.0))
    # Drift again 0.5s later (within cooldown period 2.0s) -> suppressed
    out_cooldown = sfm.step(default_inputs(now_s=1.5, road_d_m=0.72, lateral_velocity_mps=0.25))
    assert out_cooldown.requests.request_for(Function.LKA).state != RequestState.REQUEST

    # Drift again 2.5s later (after cooldown period) -> re-arms and intervenes
    sfm.step(default_inputs(now_s=3.5, road_d_m=0.72, lateral_velocity_mps=0.25))
    out_rearmed = sfm.step(default_inputs(now_s=3.525, road_d_m=0.72, lateral_velocity_mps=0.25))
    assert out_rearmed.requests.request_for(Function.LKA).state == RequestState.REQUEST


# =========================================================================
# S17: Driver lane change during HWA
# =========================================================================
def test_s17_driver_lane_change_during_hwa():
    sfm = make_sfm()
    # Driver indicates left and turns wheel to change lane
    sfm.step(default_inputs(now_s=1.0, driver_torque_nm=3.5, turn_signal=Side.LEFT))
    out_override = sfm.step(default_inputs(now_s=1.15, driver_torque_nm=3.5, turn_signal=Side.LEFT, lane_change_completed_by_driver=True))
    assert out_override.driver_override is True

    # Driver releases wheel after moving into left lane
    out_resume = sfm.step(default_inputs(now_s=1.25, driver_torque_nm=0.0, road_d_m=0.0, turn_signal=Side.NONE))
    assert out_resume.lateral_state == LateralState.HWA_CENTER
    # Lane center updated by -3.5m (left lane is negative in the sheet frame)
    assert sfm.current_lane_center_m == -3.5
