import json
from pathlib import Path

from l2_path_planner.messages import (
    Function, FunctionRequest, FunctionRequests, Header, LaneLine, LaneModel,
    LateralManeuver, RequestState, VehicleStateEstimate,
)
from l2_path_planner.planner import FrenetPlanner
from l2_path_planner.road_geometry import RoadGeometry


ROOT = Path(__file__).parents[1]


def make_planner():
    cfg = json.loads((ROOT / "configs/baseline.json").read_text())
    road = RoadGeometry.from_config(cfg["road"])
    return cfg, FrenetPlanner(cfg, road, cfg["vehicle"])


def test_planner_emits_exactly_31_points_and_rejects_aggressive_candidate():
    cfg, planner = make_planner()
    lane = LaneModel(Header(0, 0), LaneLine(-1.75, 0, 0, 0), LaneLine(1.75, 0, 0, 0), 3.5)
    req = FunctionRequests(Header(0, 0), [FunctionRequest(Function.HWA, RequestState.REQUEST)])
    state = VehicleStateEstimate(Header(0, 0), 0.0, 0.0, 0.0)
    trajectory, status, _, diag = planner.plan(0.0, lane, req, state, 0, 0.2, 0, 20)
    assert status.path_valid
    assert len(trajectory.points) == 31
    assert trajectory.points[0].time_from_start_s == 0.0
    assert trajectory.points[-1].time_from_start_s == 3.0
    candidate = planner._build_candidate(Function.ELK, 0.0, 20.0, (0.7, 3.0, 0.0), 0.4)
    assert not planner._feasible(candidate, Function.ELK, 20.0)
    assert diag["candidate_count"] == 3


def test_single_line_lane_is_valid():
    _, planner = make_planner()
    lane = LaneModel(Header(0, 0), LaneLine(-1.75, 0, 0, 0), None, 3.5)
    req = FunctionRequests(Header(0, 0), [FunctionRequest(Function.HWA, RequestState.REQUEST)])
    state = VehicleStateEstimate(Header(0, 0), 0.0, 0.0, 0.0)
    _, status, _, _ = planner.plan(0.0, lane, req, state, 0, 0, 0, 20)
    assert status.path_valid


def test_quintic_candidate_reaches_offset_and_lane_change_uses_wider_corridor():
    _, planner = make_planner()
    offset = planner._build_candidate(Function.HWA, 0.0, 20.0, (0.0, 0.0, 0.0), 3.0, 0.25)
    assert abs(offset.d[-1] - 0.25) < 1e-12
    change = planner._build_candidate(Function.HWA, 0.0, 20.0, (0.0, 0.0, 0.0), 11.0, -3.5)
    assert planner._feasible(change, Function.HWA, 20.0, -3.5, LateralManeuver.LANE_CHANGE_LEFT)


def test_planner_completion_uses_measured_added_acceleration():
    _, planner = make_planner()
    lane = LaneModel(Header(0, 0), LaneLine(-1.75, 0, 0, 0), LaneLine(1.75, 0, 0, 0), 3.5)
    req = FunctionRequests(Header(0, 0), [FunctionRequest(Function.LKA, RequestState.REQUEST)])
    state = VehicleStateEstimate(Header(0, 0), 0.0, 0.0, 0.0)
    _, _, _, diag = planner.plan(
        0.0, lane, req, state, 0.0, 0.0, 0.0, 20.0,
        lateral_acceleration_mps2=0.65,
    )
    assert abs(diag["completion_ay_added_mps2"] - 0.65) < 1e-12

