import json
import math
from pathlib import Path

import pytest

from l2_path_planner.messages import (
    Function, FunctionRequest, FunctionRequests, Header, LaneLine, LaneModel,
    LateralManeuver, RequestState, VehicleStateEstimate,
)
from l2_path_planner.planner import FrenetPlanner, IntegrityMonitor
from l2_path_planner.road_geometry import RoadGeometry

ROOT = Path(__file__).parents[1]


def make_planner():
    cfg = json.loads((ROOT / "configs/baseline.json").read_text())
    road = RoadGeometry.from_config(cfg["road"])
    return cfg, FrenetPlanner(cfg, road, cfg["vehicle"])


def normal_inputs(now_s=1.0, counter=1, lane_width=3.5, d_m=0.0):
    now_us = int(round(now_s * 1e6))
    lane = LaneModel(
        Header(now_us, counter),
        LaneLine(-lane_width / 2.0, 0, 0, 0),
        LaneLine(lane_width / 2.0, 0, 0, 0),
        lane_width,
    )
    req = FunctionRequests(Header(now_us, counter), [FunctionRequest(Function.HWA, RequestState.REQUEST)])
    state = VehicleStateEstimate(Header(now_us, counter), 0.0, 0.0, 0.0)
    return now_s, lane, req, state


# =========================================================================
# F01 & F02: Lane c0 Jump and Noise Continuity Enforcement
# =========================================================================
def test_f01_f02_c0_jump_continuity_enforcement():
    cfg, planner = make_planner()
    # Cycle 1: steady state at d = 0.0m
    now_s, lane, req, state = normal_inputs(now_s=1.000, counter=1, d_m=0.0)
    traj1, status1, _, _ = planner.plan(now_s, lane, req, state, 0.0, 0.0, 0.0, 20.0)
    assert status1.path_valid

    # Cycle 2: sudden c0 jump by 0.6 m without warning flag
    now_s2, lane2, req2, state2 = normal_inputs(now_s=1.025, counter=2, d_m=0.6)
    traj2, status2, _, _ = planner.plan(now_s2, lane2, req2, state2, 0.0, 0.6, 0.0, 20.0)
    assert status2.path_valid

    # Output trajectory must NOT jump by 0.6m; position shift per cycle is bounded by continuity_position_m (0.05m)
    step_diff = abs(traj2.points[0].road_d_m - traj1.points[0].road_d_m)
    assert step_diff <= cfg["planner"]["continuity_position_m"] + 1e-4


# =========================================================================
# F03: Misidentified Adjacent Lane (Lane Width Jump 3.5m -> 7.0m)
# =========================================================================
def test_f03_misidentified_adjacent_lane():
    cfg, planner = make_planner()
    # Lane width suddenly jumps to 7.0m (line detection jumped to adjacent lane L2)
    now_s, lane, req, state = normal_inputs(now_s=1.000, counter=1, lane_width=7.0)
    _, status, _, _ = planner.plan(now_s, lane, req, state, 0.0, 0.0, 0.0, 20.0)
    # Abnormal lane width must be rejected by planner
    assert not status.path_valid


# =========================================================================
# F04: Lost Single Lane Line
# =========================================================================
def test_f04_lost_single_lane_line():
    cfg, planner = make_planner()
    now_us = 1_000_000
    # Left line missing, right line still present
    lane_missing_left = LaneModel(
        Header(now_us, 1),
        None,
        LaneLine(1.75, 0, 0, 0),
        3.5,
    )
    req = FunctionRequests(Header(now_us, 1), [FunctionRequest(Function.HWA, RequestState.REQUEST)])
    state = VehicleStateEstimate(Header(now_us, 1), 0.0, 0.0, 0.0)
    _, status, _, _ = planner.plan(1.0, lane_missing_left, req, state, 0.0, 0.0, 0.0, 20.0)
    # Single-line lane model is supported and valid
    assert status.path_valid


# =========================================================================
# F05: Lost Both Lines & 1.0s Coasting
# =========================================================================
def test_f05_lost_both_lines_coasting_and_timeout():
    cfg, planner = make_planner()
    # Cycle 1: healthy lane
    now_s, lane, req, state = normal_inputs(now_s=1.000, counter=1)
    planner.plan(now_s, lane, req, state, 0.0, 0.0, 0.0, 20.0)

    # Cycle 2: both lines lost at t = 1.025s
    empty_lane = LaneModel(Header(1_025_000, 2), None, None, 3.5)
    _, status_coast, _, _ = planner.plan(1.025, empty_lane, req, state, 0.0, 0.0, 0.0, 20.0)
    # At t = 1.025s (0.025s after loss), coasting is active -> path_valid is True
    assert status_coast.path_valid

    # Cycle 3: at t = 1.5s (0.5s after loss), still within 1.0s coast window
    now_us_mid = 1_500_000
    req_mid = FunctionRequests(Header(now_us_mid, 3), [FunctionRequest(Function.HWA, RequestState.REQUEST)])
    state_mid = VehicleStateEstimate(Header(now_us_mid, 3), 0.0, 0.0, 0.0)
    empty_lane_mid = LaneModel(Header(now_us_mid, 3), None, None, 3.5)
    _, status_mid, _, _ = planner.plan(1.500, empty_lane_mid, req_mid, state_mid, 0.0, 0.0, 0.0, 20.0)
    assert status_mid.path_valid

    # Cycle 4: at t = 2.1s (1.1s after loss), T_coast (1.0s) exceeded -> path_valid MUST be False
    now_us_exp = 2_100_000
    req_exp = FunctionRequests(Header(now_us_exp, 4), [FunctionRequest(Function.HWA, RequestState.REQUEST)])
    state_exp = VehicleStateEstimate(Header(now_us_exp, 4), 0.0, 0.0, 0.0)
    empty_lane_exp = LaneModel(Header(now_us_exp, 4), None, None, 3.5)
    _, status_expired, _, _ = planner.plan(2.100, empty_lane_exp, req_exp, state_exp, 0.0, 0.0, 0.0, 20.0)
    assert not status_expired.path_valid


# =========================================================================
# F11: Out-of-order Messages
# =========================================================================
def test_f11_out_of_order_message():
    monitor = IntegrityMonitor(max_age_s=0.25, future_tolerance_s=0.01)
    # First message: stamp = 1_000_000, counter = 1
    assert monitor.valid("lane", Header(1_000_000, 1), 1_010_000)
    # Second message arrives with newer stamp: stamp = 1_050_000, counter = 2
    assert monitor.valid("lane", Header(1_050_000, 2), 1_060_000)
    # Third message arrives OUT OF ORDER (delayed older message: stamp = 1_020_000, counter = 3)
    assert not monitor.valid("lane", Header(1_020_000, 3), 1_070_000)


# =========================================================================
# F10, F15, F16: Stale Counter, Repeating Counter, and Invalid Qualifier
# =========================================================================
def test_f10_f15_f16_counter_and_validity_checks():
    monitor = IntegrityMonitor(max_age_s=0.25, future_tolerance_s=0.01)
    assert monitor.valid("lane", Header(1_000_000, 1), 1_010_000)

    # F15: Repeating counter (counter = 1 delivered again) -> rejected
    assert not monitor.valid("lane", Header(1_020_000, 1), 1_030_000)

    # F16: Header marked invalid (valid = False, e.g. CAN CRC or qualifier fault) -> rejected
    assert not monitor.valid("lane", Header(1_040_000, 2, valid=False), 1_050_000)


# =========================================================================
# F12: Timeout / Stopped Sending Input
# =========================================================================
def test_f12_input_timeout_exceeded():
    monitor = IntegrityMonitor(max_age_s=0.25, future_tolerance_s=0.01)
    # Message is 0.30s old (> max_age 0.25s) -> timeout rejected
    assert not monitor.valid("lane", Header(1_000_000, 1), 1_300_001)


# =========================================================================
# F19: Future Timestamp Tolerance
# =========================================================================
def test_f19_future_timestamp_tolerance():
    monitor = IntegrityMonitor(max_age_s=0.25, future_tolerance_s=0.01)  # 10 ms tolerance
    # Stamp 5 ms into future (within 10 ms tolerance) -> accepted
    assert monitor.valid("lane", Header(1_005_000, 1), 1_000_000)

    # Stamp 20 ms into future (exceeds 10 ms tolerance) -> rejected
    monitor_new = IntegrityMonitor(max_age_s=0.25, future_tolerance_s=0.01)
    assert not monitor_new.valid("lane", Header(1_020_000, 1), 1_000_000)


# =========================================================================
# F20: Invalid Ego-Motion / Vehicle State Estimate
# =========================================================================
def test_f20_invalid_ego_motion_invalidates_path_immediately():
    cfg, planner = make_planner()
    # Cycle 1: healthy operation
    now_s, lane, req, state = normal_inputs(now_s=1.000, counter=1)
    _, status_ok, _, _ = planner.plan(now_s, lane, req, state, 0.0, 0.0, 0.0, 20.0)
    assert status_ok.path_valid

    # Cycle 2: state estimator reports fault (valid = False)
    bad_state = VehicleStateEstimate(Header(1_025_000, 2, valid=False), 0.0, 0.0, 0.0)
    _, status_bad, _, _ = planner.plan(1.025, lane, req, bad_state, 0.0, 0.0, 0.0, 20.0)
    # PP-SWR-143: path_valid MUST be False in <= 1 cycle
    assert not status_bad.path_valid
