"""Frame-convention regression: the code is mirror-symmetric about the x axis.

The signal list puts ``+y`` to the right.  Mirroring every lateral scenario
input (road apex, wind/bank direction, adjacent-lane side, explicit object
offsets) must therefore negate every lateral output and leave every
longitudinal/invariant output unchanged.

This is the guard against a partially applied sign flip: if one module kept the
old left-positive convention, the mirrored run would no longer be the exact
mirror image of the nominal one.
"""

import copy
import json
import math
from pathlib import Path

from l2_path_planner.simulation import ClosedLoopSimulation


ROOT = Path(__file__).parents[1]

#: Outputs that follow the lateral axis: ``mirrored = -nominal``.
LATERAL_COLUMNS = (
    "y_m",
    "road_d_m",
    "heading_error_rad",
    "lateral_velocity_mps",
    "estimated_lateral_velocity_mps",
    "estimated_offset_m",
    "yaw_rate_rps",
    "lateral_acceleration_mps2",
    "lateral_jerk_mps3",
    "road_lateral_acceleration_mps2",
    "commanded_added_acceleration_mps2",
    "steering_command_rad",
    "steering_angle_rad",
    "yaw_rate_raw_degps",
    "yaw_rate_filtered_degps",
    "ay_raw_mps2",
    "ay_filtered_mps2",
    "swa_deg",
    "sws_degps",
    "crosswind_mps",
    "aerodynamic_force_n",
    "aerodynamic_yaw_moment_nm",
    "road_bank_angle_deg",
)

#: Line offsets are ego-relative, so a mirrored scenario swaps the left and
#: right line and negates it: ``left_mirrored == -right_nominal``.
LINE_PAIR_COLUMNS = (
    ("lane_left_c0_m", "lane_right_c0_m"),
    ("lane_next_left_c0_m", "lane_next_right_c0_m"),
)

#: Distances are unsigned, so the mirrored run simply swaps the left/right pair.
SWAP_COLUMNS = (
    ("dlc_left_m", "dlc_right_m"),
    ("tlc_left_s", "tlc_right_s"),
)

#: Outputs that do not depend on the lateral sign.
INVARIANT_COLUMNS = (
    "time_s",
    "x_m",
    "speed_mps",
    "road_s_m",
    "ax_raw_mps2",
    "lane_width_derived_m",
    "lane_margin_m",
    "continuity_position_m",
    "continuity_heading_rad",
    "continuity_curvature_inv_m",
    "friction_coefficient",
    "active_function",
)


def _flip_side(value: str) -> str:
    return "RIGHT" if str(value).upper() != "RIGHT" else "LEFT"


def _mirror_segment(segment: dict) -> None:
    kind = str(segment.get("type", "straight")).lower()
    if kind == "arc":
        if "radius_m" in segment:
            segment["radius_m"] = -float(segment["radius_m"])
        if "curvature_inv_m" in segment:
            segment["curvature_inv_m"] = -float(segment["curvature_inv_m"])
    elif kind == "clothoid":
        for key in ("start_curvature_inv_m", "end_curvature_inv_m"):
            if key in segment:
                segment[key] = -float(segment[key])
    elif kind == "cosine":
        if "amplitude_m" in segment:
            segment["amplitude_m"] = -float(segment["amplitude_m"])


def _mirror_config(cfg: dict) -> dict:
    """Return the same physical scenario described in the mirrored frame."""
    out = copy.deepcopy(cfg)
    for segment in out.get("road", {}).get("centerline", []):
        _mirror_segment(segment)
    for entry in out.get("road", {}).get("markings", []):
        entry["left"], entry["right"] = entry.get("right", "DASHED"), entry.get("left", "DASHED")
        for prefix in ("confidence", "view_range_m"):
            left, right = f"left_{prefix}", f"right_{prefix}"
            if left in entry or right in entry:
                entry[left], entry[right] = entry.get(right), entry.get(left)
    for edge in out.get("road", {}).get("edges", []):
        edge["side"] = _flip_side(edge.get("side", "RIGHT"))
    for profile in out.get("environment", []) + out.get("disturbances", []):
        profile["direction"] = -float(profile.get("direction", 1.0))
    initial = out.get("ego", {}).get("initial", {})
    for key in ("y_m", "yaw_rad", "v_lat_mps"):
        if key in initial:
            initial[key] = -float(initial[key])
    for actor in out.get("actors", []):
        for point in actor.get("trajectory", []):
            point["d"] = -float(point.get("d", 0.0))
    for entry in out.get("adas", []):
        for key in ("turn_signal", "lane_change_request"):
            if key in entry and str(entry[key]).upper() in ("LEFT", "RIGHT"):
                entry[key] = _flip_side(entry[key])
        for key in ("driver_torque_nm", "driver_rate_dps"):
            if key in entry:
                entry[key] = -float(entry[key])
    return out


def _run(cfg: dict) -> list[dict]:
    sim = ClosedLoopSimulation(copy.deepcopy(cfg), controller_backend="feedback")
    sim.run(write_artifacts=False)
    return sim.timeseries


def test_mirroring_the_scenario_negates_only_the_lateral_channels():
    cfg = json.loads((ROOT / "configs/baseline.json").read_text())
    # Long enough to cross the curve start (80 m), the truck (2.5-4.5 s) and
    # the first crosswind (5.0 s), short enough to keep the test fast.
    cfg["duration_s"] = 6.0

    nominal = _run(cfg)
    mirrored = _run(_mirror_config(cfg))
    assert len(nominal) == len(mirrored)

    for column in LATERAL_COLUMNS:
        for a, b in zip(nominal, mirrored):
            assert math.isclose(float(b[column]), -float(a[column]), abs_tol=1e-9), column

    for left, right in LINE_PAIR_COLUMNS:
        for a, b in zip(nominal, mirrored):
            assert math.isclose(float(b[left]), -float(a[right]), abs_tol=1e-9), left
            assert math.isclose(float(b[right]), -float(a[left]), abs_tol=1e-9), right

    for left, right in SWAP_COLUMNS:
        for a, b in zip(nominal, mirrored):
            assert math.isclose(float(b[left]), float(a[right]), abs_tol=1e-9), left
            assert math.isclose(float(b[right]), float(a[left]), abs_tol=1e-9), right

    for column in INVARIANT_COLUMNS:
        for a, b in zip(nominal, mirrored):
            assert math.isclose(float(b[column]), float(a[column]), abs_tol=1e-9), column

    # The DLC/TLC values are unsigned distances, so they swap in the mirrored
    # scenario rather than being equal or negated.
    for i, (left, right) in enumerate(SWAP_COLUMNS):
        for a, b in zip(nominal, mirrored):
            assert math.isclose(float(b[left]), float(a[right]), abs_tol=1e-9), left
            assert math.isclose(float(b[right]), float(a[left]), abs_tol=1e-9), right

    # The scenario must actually exercise the lateral channels, otherwise the
    # invariance above would be vacuous.
    assert max(abs(float(row["road_d_m"])) for row in nominal) > 0.05
    assert max(abs(float(row["steering_command_rad"])) for row in nominal) > 1e-3
