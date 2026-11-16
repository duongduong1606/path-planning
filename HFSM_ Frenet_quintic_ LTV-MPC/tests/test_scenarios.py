"""Library-level tests for the S1-S17 scenario set (schema v2).

The scenario files are the contract between the benchmark and the planner, so
they are checked in three layers:

* every family file loads, validates and expands without simulation,
* a representative subset is actually driven through the closed loop and must
  satisfy its own ``expectations`` block,
* the exported UI payload carries the channels the static viewer needs.

The full 72-scenario sweep takes about a minute, so it only runs when
``L2_FULL_SCENARIO_SWEEP=1`` is set.
"""

import copy
import json
import os
import tempfile
from pathlib import Path

import pytest

from l2_path_planner.scenario import (
    DEFAULT_SCENARIO_DIR, ScenarioError, list_scenarios, load_all, registry, validate,
)
from l2_path_planner.lane_model import LaneModelSynthesizer
from l2_path_planner.road_geometry import RoadGeometry
from l2_path_planner.simulation import ClosedLoopSimulation
from l2_path_planner.ui_export import export_ui_data


EXPECTED_FAMILIES = {
    "S1", "S1b", "S2", "S3", "S3b", "S4", "S4b", "S5", "S6", "S6b", "S7",
    "S8", "S8b", "S8c", "S9", "S10", "S11", "S12", "S13", "S14", "S15",
    "S16", "S17",
}

#: One representative scenario per behaviour, kept short enough for the default
#: test run.  The full sweep is covered by ``run_all_scenarios.py``.
SMOKE_IDS = ("S1-a", "S4-a", "S6-a", "S14-a")


def _run(scenario, duration_s=None):
    cfg = copy.deepcopy(scenario.runtime)
    if duration_s is not None:
        cfg["duration_s"] = float(duration_s)
    sim = ClosedLoopSimulation(cfg, controller_backend="feedback")
    summary = sim.run(write_artifacts=False)
    return sim, summary


def test_library_covers_every_family_and_validates():
    assert DEFAULT_SCENARIO_DIR.exists(), "scenarios/ directory is missing"
    scenarios = registry()
    assert len(scenarios) == 72
    assert {scenario.family for scenario in scenarios.values()} == EXPECTED_FAMILIES

    for scenario in scenarios.values():
        # validate() raises ScenarioError with the scenario id in the message.
        validate(scenario.runtime)
        assert scenario.duration_s > 0.0
        assert scenario.runtime["ego"]["speed_mps"] >= 0.0
        assert scenario.expectations, f"{scenario.id} declares no expectations"


def test_family_files_expand_to_unique_ids():
    seen = {}
    for path in sorted(DEFAULT_SCENARIO_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        for scenario in load_all(path):
            assert scenario.id not in seen, f"{scenario.id} duplicated in {path.name}"
            seen[scenario.id] = path.name
    assert len(seen) == 72


def test_scenario_listing_is_planner_independent():
    listing = list_scenarios()
    assert len(listing) == 72
    first = listing[0]
    assert set(first) == {
        "id", "family", "title", "category", "duration_s", "seed", "requirements",
    }
    # The selector must be ordered by id so the UI groups families predictably.
    assert [item["id"] for item in listing] == sorted(item["id"] for item in listing)


def test_invalid_documents_are_rejected():
    scenario = registry()["S4-a"]
    broken = copy.deepcopy(scenario.runtime)
    broken["road"]["centerline"] = [{"s_m": 0.0, "type": "spiral", "length_m": 10.0}]
    with pytest.raises(ScenarioError):
        validate(broken)

    broken = copy.deepcopy(scenario.runtime)
    broken["expectations"] = {"must_activate": ["HWA", "ALC"]}
    with pytest.raises(ScenarioError):
        validate(broken)


@pytest.mark.parametrize("scenario_id", SMOKE_IDS)
def test_smoke_scenarios_meet_their_expectations(scenario_id):
    sim, summary = _run(registry()[scenario_id])
    result = summary["expectations"]
    failed = [name for name, check in result["checks"].items() if not check["passed"]]
    assert result["passed"], f"{scenario_id} failed {failed}"
    assert summary["max_abs_lateral_error_m"] < 5.0


def test_smoke_scenarios_activate_the_expected_function():
    expected = {"S1-a": "HWA", "S4-a": "LKA", "S6-a": "ELK", "S14-a": "HWA"}
    for scenario_id, function in expected.items():
        sim, _ = _run(registry()[scenario_id])
        modes = {
            event["detail"] for event in sim.events if event["event"] == "MODE"
        }
        assert function in modes, f"{scenario_id} never entered {function}"


def test_same_seed_reproduces_identical_timeseries():
    scenario = registry()["S4-a"]
    first, _ = _run(scenario)
    second, _ = _run(scenario)

    assert len(first.timeseries) == len(second.timeseries)
    for a, b in zip(first.timeseries, second.timeseries):
        for key, value in a.items():
            if key.endswith("_ms"):  # wall-clock diagnostics are not physical state
                continue
            assert b[key] == value, key


def test_ui_export_carries_playback_channels():
    sim, _ = _run(registry()["S4-a"])
    with tempfile.TemporaryDirectory() as tmp:
        path = export_ui_data(sim, Path(tmp) / "ui_data.json")
        payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["scenario"]["id"] == "S4-a"
    assert payload["result"]["passed"] is True

    ts = payload["timeseries"]
    assert len(ts["time_s"]) == len(ts["yaw_rad"]) == len(payload["ego"])
    assert {"x_m", "y_m", "yaw_rad", "speed_mps", "road_d_m"} <= set(ts)

    ego = payload["ego"]
    assert ego[0]["t"] == pytest.approx(0.0)
    assert ego[0]["yaw"] == pytest.approx(0.0, abs=1e-6)
    assert ego[0]["mode"] in {"NONE", "HWA", "LKA", "ELK"}

    road = payload["road"]
    assert len(road["s"]) == len(road["left"]) == len(road["right"])
    assert road["markings"], "road markings are not exported"
    assert len(road["s"]) > 10


# --------------------------------------------------------------- lane layout

def _road(scenario_id):
    return RoadGeometry.from_config(registry()[scenario_id].runtime["road"])


def test_three_lane_road_geometry():
    road = _road("S1-a")
    assert road.lane_count == 3
    assert road.lane_widths() == pytest.approx([3.5, 3.5, 3.5])
    # The ego lane is the middle lane and keeps its centre at d = 0.
    assert road.lane_centers() == pytest.approx([-3.5, 0.0, 3.5])
    assert [entry["d"] for entry in road.lane_boundaries()] == pytest.approx(
        [-5.25, -1.75, 1.75, 5.25]
    )
    assert road.carriageway_extent() == pytest.approx((-5.25, 5.25))
    # The planner's view of the road must not move.
    assert road.width_at(0.0) == pytest.approx(3.5)
    assert road.next_lane_offset("LEFT") == pytest.approx(-5.25)
    assert road.next_lane_offset("RIGHT") == pytest.approx(5.25)


def test_lane_model_uses_the_real_adjacent_lane_widths():
    road = RoadGeometry.from_config({
        "lane_width_m": 3.5,
        "centerline": [{"s_m": 0.0, "type": "straight", "length_m": 100.0}],
        "lanes": {"ego_index": 1, "widths_m": [2.75, 3.5, 3.25]},
    })
    model = LaneModelSynthesizer(road).build(0.0, 1, 0.0, 0.0, 0.0, 0.0)
    assert model.next_left.c0 == pytest.approx(-(3.5 / 2.0 + 2.75))
    assert model.next_right.c0 == pytest.approx(3.5 / 2.0 + 3.25)
    assert road.lane_centers() == pytest.approx(
        [-(3.5 / 2.0 + 2.75 / 2.0), 0.0, 3.5 / 2.0 + 3.25 / 2.0]
    )


def test_single_lane_road_keeps_the_legacy_lane_model():
    road = RoadGeometry.from_config({
        "lane_width_m": 3.5,
        "centerline": [{"s_m": 0.0, "type": "straight", "length_m": 100.0}],
    })
    assert road.lane_count == 1
    assert road.lane_centers() == [0.0]
    model = LaneModelSynthesizer(road).build(0.0, 1, 0.0, 0.0, 0.0, 0.0)
    assert model.next_left.c0 == pytest.approx(-5.25)
    assert model.next_right.c0 == pytest.approx(5.25)


def test_opposing_lane_layout():
    road = _road("S7-a")
    assert road.has_opposing
    assert road.lane_count == 1
    assert road.opposing_lane_centers() == pytest.approx([-3.5])
    # The next line to the left is the centre line, not another driving lane.
    assert road.next_lane_offset("LEFT") == pytest.approx(-1.75)


@pytest.mark.parametrize("patch", [
    {"ego_index": 5},
    {"widths_m": [3.5, 3.0, 3.5]},
    {"widths_m": []},
    {"widths_m": [3.5, 3.5, 3.5], "roles": ["CAR", "BUS"]},
    {"widths_m": [3.5, 3.5, 3.5], "opposing": {"separator": "RAINBOW"}},
])
def test_lane_layout_validation(patch):
    broken = copy.deepcopy(registry()["S1-a"].runtime)
    broken["road"]["lanes"].update(patch)
    with pytest.raises(ScenarioError):
        validate(broken)


def test_ui_export_draws_three_lanes():
    sim, _ = _run(registry()["S1-a"])
    with tempfile.TemporaryDirectory() as tmp:
        path = export_ui_data(sim, Path(tmp) / "ui_data.json")
        payload = json.loads(path.read_text(encoding="utf-8"))

    road = payload["road"]
    lanes = road["lanes"]
    assert lanes["ego_index"] == 1
    assert lanes["widths"] == pytest.approx([3.5, 3.5, 3.5])
    assert [line["index"] for line in lanes["lines"]] == [0, 1, 2, 3]
    kinds = {(style["index"], style["kind"]) for style in lanes["styles"]}
    assert (0, "WHITE_SOLID") in kinds
    assert (1, "WHITE_DASHED") in kinds
    assert (2, "WHITE_DASHED") in kinds
    assert (3, "WHITE_SOLID") in kinds
    assert lanes["carriageway"]["width_m"] == pytest.approx(10.5, abs=1e-3)
    assert len(lanes["carriageway"]["left"]) == len(road["s"])
    # The legacy single-lane channels stay in the payload for older viewers.
    assert len(road["left"]) == len(road["s"])


def test_ui_export_marks_the_opposing_lane_yellow():
    sim, _ = _run(registry()["S7-a"])
    with tempfile.TemporaryDirectory() as tmp:
        path = export_ui_data(sim, Path(tmp) / "ui_data.json")
        payload = json.loads(path.read_text(encoding="utf-8"))

    lanes = payload["road"]["lanes"]
    assert lanes["opposing"]["separator"] == "DOUBLE_YELLOW"
    assert any(style["kind"] == "DOUBLE_YELLOW" for style in lanes["styles"])
    assert lanes["carriageway"]["width_m"] == pytest.approx(7.0, abs=1e-3)
    assert [line["role"] for line in lanes["lines"]][-1] == "OPPOSING_OUTER"


@pytest.mark.skipif(
    os.environ.get("L2_FULL_SCENARIO_SWEEP") != "1",
    reason="set L2_FULL_SCENARIO_SWEEP=1 to run all 72 scenarios (~70 s)",
)
def test_full_library_meets_expectations():
    failures = []
    for scenario_id, scenario in sorted(registry().items()):
        _, summary = _run(scenario)
        result = summary["expectations"]
        if not result["passed"]:
            failed = [name for name, check in result["checks"].items() if not check["passed"]]
            failures.append((scenario_id, failed))
    assert not failures, failures
