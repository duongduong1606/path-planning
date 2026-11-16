import json
from pathlib import Path

from l2_path_planner.behavior import BehaviorInputs, HierarchicalSFM, LateralState
from l2_path_planner.messages import (
    Function, Header, LaneLine, LaneModel, LateralManeuver, LongitudinalMode,
    RequestState, Side,
)


ROOT = Path(__file__).parents[1]


def lane(right_solid=False):
    return LaneModel(
        Header(0, 0), LaneLine(-1.75, 0, 0, 0),
        LaneLine(1.75, 0, 0, 0, line_class="SOLID" if right_solid else "DASHED"), 3.5,
    )


def make_inputs(**kwargs):
    base = dict(
        now_s=1.0, speed_mps=20.0, road_d_m=0.0, road_heading_error_rad=0.0,
        lateral_velocity_mps=0.0, yaw_rate_rps=0.0, lane=lane(),
        hwa_button=True, acc_active=True, desired_speed_mps=20.0,
    )
    base.update(kwargs)
    return BehaviorInputs(**base)


def sfm():
    cfg = json.loads((ROOT / "configs/baseline.json").read_text())
    return HierarchicalSFM(cfg["behavior"], cfg["vehicle"])


def test_hwa_center_offset_and_lane_change_substates():
    model = sfm()
    center = model.step(make_inputs())
    assert center.requests.request_for(Function.HWA).state == RequestState.REQUEST
    offset = model.step(make_inputs(now_s=1.1, adjacent_hazard_side=Side.LEFT, adjacent_clearance_m=0.6))
    hwa = offset.requests.request_for(Function.HWA)
    assert hwa.maneuver == LateralManeuver.OFFSET_RIGHT and hwa.target_offset_m > 0
    change = model.step(make_inputs(now_s=1.2, lane_change_request=Side.LEFT, adjacent_lane_safe=True))
    hwa = change.requests.request_for(Function.HWA)
    assert hwa.maneuver == LateralManeuver.LANE_CHANGE_LEFT and hwa.target_offset_m == -3.5


def test_lka_debounce_elk_priority_and_longitudinal_states():
    model = sfm()
    drifting = make_inputs(road_d_m=0.72, lateral_velocity_mps=0.25)
    first = model.step(drifting)
    second = model.step(make_inputs(now_s=1.025, road_d_m=0.72, lateral_velocity_mps=0.25))
    assert first.requests.request_for(Function.LKA).state != RequestState.REQUEST
    assert second.requests.request_for(Function.LKA).state == RequestState.REQUEST
    emergency_model = sfm()
    elk = emergency_model.step(make_inputs(lane=lane(True), road_d_m=0.72, lateral_velocity_mps=0.35))
    assert elk.requests.request_for(Function.ELK).state == RequestState.REQUEST
    assert elk.lateral_state == LateralState.ELK_RIGHT
    accelerate = sfm().step(make_inputs(speed_mps=15.0, desired_speed_mps=20.0))
    assert accelerate.longitudinal.mode == LongitudinalMode.ACCELERATE
    follow = sfm().step(make_inputs(lead_gap_m=50.0, lead_speed_mps=18.0))
    assert follow.longitudinal.mode == LongitudinalMode.FOLLOW
    brake = sfm().step(make_inputs(lead_gap_m=10.0, lead_speed_mps=12.0))
    assert brake.longitudinal.mode == LongitudinalMode.BRAKE
    aeb = sfm().step(make_inputs(obstacle_distance_m=5.0))
    assert aeb.longitudinal.mode == LongitudinalMode.EMERGENCY_BRAKE


def test_lane_change_latches_until_target_and_updates_lane_center():
    model = sfm()
    started = model.step(make_inputs(lane_change_request=Side.LEFT))
    assert started.requests.request_for(Function.HWA).target_offset_m == -3.5
    latched = model.step(make_inputs(now_s=1.025, lane_change_request=Side.NONE, road_d_m=-1.0))
    assert latched.requests.request_for(Function.HWA).maneuver == LateralManeuver.LANE_CHANGE_LEFT
    assert latched.requests.request_for(Function.LKA).state != RequestState.REQUEST
    completed = model.step(make_inputs(
        now_s=1.05, lane_change_request=Side.NONE, road_d_m=-3.5,
        lateral_velocity_mps=0.0,
    ))
    assert model.current_lane_center_m == -3.5
    assert completed.requests.request_for(Function.HWA).maneuver == LateralManeuver.LANE_CENTER


def test_driver_override_debounce_and_active_lka_speed_hold():
    override_model = sfm()
    assert not override_model.step(make_inputs(now_s=1.0, driver_torque_nm=3.2)).driver_override
    assert override_model.step(make_inputs(now_s=1.11, driver_torque_nm=3.2)).driver_override

    model = sfm()
    model.step(make_inputs(road_d_m=0.72, lateral_velocity_mps=0.25))
    active = model.step(make_inputs(now_s=1.025, road_d_m=0.72, lateral_velocity_mps=0.25))
    assert active.requests.request_for(Function.LKA).state == RequestState.REQUEST
    held = model.step(make_inputs(
        now_s=1.05, speed_mps=17.5, road_d_m=0.4, lateral_velocity_mps=0.1,
    ))
    assert held.requests.request_for(Function.LKA).state == RequestState.REQUEST
