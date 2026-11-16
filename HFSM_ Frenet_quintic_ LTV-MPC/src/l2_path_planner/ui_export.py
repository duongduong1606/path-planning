"""Export a finished run to the static UI data format."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .behavior import LateralState
from .messages import Function


FUNCTION_NAMES = {0: "NONE", 1: "HWA", 2: "LKA", 3: "ELK"}

# Scenario markings use the old vocabulary; the canvas renderer uses the
# Vietnamese road-marking vocabulary (QCVN 41 white 2.1 / 2.2 / yellow 1.3).
LEGACY_LINE_KINDS = {
    "DASHED": "WHITE_DASHED",
    "SOLID": "WHITE_SOLID",
    "DOUBLE": "WHITE_DOUBLE",
    "ROAD_EDGE": "WHITE_SOLID",
    "BARRIER": "WHITE_SOLID",
    "LOST": None,
    "NONE": None,
}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _round(value, digits: int = 4) -> float:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return 0.0
    return round(float(value), digits)


def _polyline(road, s_values, d_values):
    x, y, _, _ = road.frenet_to_global(s_values, d_values, np.zeros_like(s_values), 1.0)
    return [[_round(a), _round(b)] for a, b in zip(x, y)]


def _lanes_payload(road, s_values, max_s: float) -> dict | None:
    """Lane bands and their boundary lines, ready for the canvas renderer.

    ``lines`` carry one polyline per boundary; ``styles`` say how to paint each
    boundary over an ``s`` range (later entries win, so scenario ``markings``
    can override the derived default on the ego lane).
    """
    if not road.lanes:
        return None
    ego = int(road.lanes["ego_index"])
    widths = list(road.lanes["widths_m"])
    centers = road.lane_centers()
    count = len(widths)
    opposing = road.lanes.get("opposing")

    def offset_series(index: int):
        values = np.empty_like(s_values)
        if index <= count:
            for i, s in enumerate(s_values):
                values[i] = road.lane_boundaries(float(s))[index]["d"]
        else:
            # Far edge of the opposite carriageway.
            step = opposing["width_m"] * (index - count)
            for i, s in enumerate(s_values):
                values[i] = road.lane_boundaries(float(s))[0]["d"] - step
        return values

    def role_of(index: int) -> str:
        if index > count:
            return "OPPOSING_OUTER"
        if index == 0:
            return "OUTER_LEFT"
        if index == count:
            return "OUTER_RIGHT"
        return "INNER"

    lines = []
    last = count + 1 if opposing else count
    for index in range(last + 1):
        lines.append({
            "index": index,
            "role": role_of(index),
            "points": _polyline(road, s_values, offset_series(index)),
        })

    separator = str(opposing.get("separator", "DOUBLE_YELLOW")).upper() if opposing else None
    default_kind = {}
    for index in range(last + 1):
        if index > count:
            default_kind[index] = "WHITE_SOLID"
        elif index == 0 and opposing:
            default_kind[index] = separator
        elif index == 0 or index == count:
            default_kind[index] = "WHITE_SOLID"
        else:
            default_kind[index] = "WHITE_DASHED"

    overrides: dict[int, list[tuple[float, float, str]]] = {}
    for entry in road.markings:
        start = max(0.0, float(entry.get("s_start_m", 0.0)))
        end = min(max_s, float(entry.get("s_end_m", max_s)))
        if end <= start:
            continue
        for boundary_index, key in ((ego, "left"), (ego + 1, "right")):
            if boundary_index > count or (boundary_index == 0 and opposing):
                continue
            raw = str(entry.get(key, "")).upper()
            if not raw:
                continue
            overrides.setdefault(boundary_index, []).append(
                (start, end, LEGACY_LINE_KINDS.get(raw, raw))
            )

    styles = []
    for index in range(last + 1):
        base = default_kind[index]
        events = overrides.get(index, [])
        cuts = {0.0, float(max_s)}
        for start, end, _ in events:
            cuts.add(start)
            cuts.add(end)
        ordered = sorted(cuts)
        for a, b in zip(ordered, ordered[1:]):
            if b - a < 1e-9:
                continue
            mid = 0.5 * (a + b)
            kind = base
            for start, end, candidate in events:
                if start <= mid < end:
                    kind = candidate
            kind = kind or "NONE"
            previous = styles[-1] if styles else None
            if (
                previous is not None
                and previous["index"] == index
                and previous["kind"] == kind
                and abs(previous["s_end"] - a) < 1e-9
            ):
                previous["s_end"] = _round(b)
            else:
                styles.append({
                    "index": index, "kind": kind,
                    "s_start": _round(a), "s_end": _round(b),
                })

    carriageway_left = offset_series(count + 1 if opposing else 0)
    carriageway_right = offset_series(count)
    return {
        "ego_index": ego,
        "widths": [_round(value) for value in widths],
        "centers": [_round(value) for value in centers],
        "roles": list(road.lanes.get("roles") or ["CAR"] * count),
        "opposing": None if not opposing else {
            "count": int(opposing["count"]),
            "width_m": _round(opposing["width_m"]),
            "separator": separator,
        },
        "lines": lines,
        "styles": styles,
        "carriageway": {
            "left": _polyline(road, s_values, carriageway_left),
            "right": _polyline(road, s_values, carriageway_right),
            "width_m": _round(float(np.max(carriageway_right - carriageway_left))),
        },
    }


def _road_payload(sim, max_s: float) -> dict:
    road = sim.road
    step = 2.0
    s_values = np.arange(0.0, max(max_s, 50.0) + step, step)
    x, y, yaw, kappa = road.frenet_to_global(
        s_values, np.zeros_like(s_values), np.zeros_like(s_values), 1.0,
    )
    widths = np.array([road.width_at(float(value)) for value in s_values])
    left = road.frenet_to_global(s_values, -widths / 2.0, np.zeros_like(s_values), 1.0)
    right = road.frenet_to_global(s_values, widths / 2.0, np.zeros_like(s_values), 1.0)
    markings = []
    for entry in road.markings:
        start = max(0.0, float(entry.get("s_start_m", 0.0)))
        end = min(max_s, float(entry.get("s_end_m", max_s)))
        if end <= start:
            continue
        markings.append({
            "s_start_m": start, "s_end_m": end,
            "left": str(entry.get("left", "DASHED")).upper(),
            "right": str(entry.get("right", "DASHED")).upper(),
        })
    edges = []
    for entry in road.edges:
        side = str(entry.get("side", "RIGHT")).upper()
        offset = float(entry.get("offset_m", road.lane_width_m / 2.0))
        if side == "LEFT":
            offset = -abs(offset)
        start = max(0.0, float(entry.get("s_start_m", 0.0)))
        end = min(max_s, float(entry.get("s_end_m", max_s)))
        if end <= start:
            continue
        sub_s = np.arange(start, end + 1.0, 2.0)
        ex, ey, _, _ = road.frenet_to_global(sub_s, np.full_like(sub_s, offset), np.zeros_like(sub_s), 1.0)
        edges.append({
            "side": side, "type": str(entry.get("type", "BARRIER")).upper(),
            "points": [[_round(a), _round(b)] for a, b in zip(ex, ey)],
        })
    return {
        "s": [_round(value) for value in s_values],
        "x": [_round(value) for value in x],
        "y": [_round(value) for value in y],
        "yaw": [_round(value, 5) for value in yaw],
        "curvature": [_round(value, 6) for value in kappa],
        "width": [_round(value) for value in widths],
        "left": [[_round(a), _round(b)] for a, b in zip(left[0], left[1])],
        "right": [[_round(a), _round(b)] for a, b in zip(right[0], right[1])],
        "markings": markings,
        "edges": edges,
        "lane_width_m": _round(road.lane_width_m),
        "lanes": _lanes_payload(road, s_values, max_s),
    }


def export_ui_data(sim, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = sim.timeseries
    if not ts:
        path.write_text("{}", encoding="utf-8")
        return path
    stride = max(1, int(round(0.02 / sim.dt)))
    rows = ts[::stride]
    if rows[-1] is not ts[-1]:
        rows = rows + [ts[-1]]
    keys = [
        "time_s", "x_m", "y_m", "yaw_rad", "speed_mps", "speed_kmh", "road_s_m", "road_d_m",
        "heading_error_rad", "lateral_acceleration_mps2", "lateral_jerk_mps3",
        "steering_command_rad", "steering_angle_rad", "active_function",
        "sfm_lateral_state", "sfm_longitudinal_mode", "dlc_left_m", "dlc_right_m",
        "tlc_left_s", "tlc_right_s", "lane_width_derived_m", "lane_margin_m",
        "target_speed_mps", "path_valid", "clearance_left_m", "clearance_right_m",
        "hazard_left_present", "hazard_right_present", "turn_signal",
        "lane_center_m", "road_curvature_inv_m", "friction_coefficient",
        "crosswind_mps", "lateral_v_max_mps",
    ]
    timeseries = {key: [row[key] for row in rows] for key in keys if key in rows[0]}
    ego = [
        {
            "t": _round(row["time_s"]),
            "x": _round(row["x_m"]),
            "y": _round(row["y_m"]),
            "yaw": _round(row["yaw_rad"], 5),
            "d": _round(row["road_d_m"]),
            "s": _round(row["road_s_m"]),
            "speed": _round(row["speed_mps"]),
            "mode": FUNCTION_NAMES.get(int(row["active_function"]), "NONE"),
            "lateral_state": LateralState(int(row["sfm_lateral_state"])).name,
            "turn_signal": int(row["turn_signal"]),
            "lane_center": _round(row["lane_center_m"]),
        }
        for row in rows
    ]
    trajectories: dict[str, dict] = {}
    for point in sim.trajectories:
        tick = str(point["planner_tick"])
        entry = trajectories.setdefault(tick, {"stamp": point["stamp_s"], "points": []})
        entry["points"].append({
            "t": _round(point["time_from_start_s"], 3),
            "x": _round(point["global_x_m"]),
            "y": _round(point["global_y_m"]),
            "d": _round(point["road_d_m"]),
        })
    trajectory_list = [
        {"tick": int(tick), "stamp": _round(entry["stamp"], 3), "points": entry["points"]}
        for tick, entry in sorted(trajectories.items(), key=lambda item: int(item[0]))
    ]
    actor_series: dict[str, dict] = {}
    for row in rows:
        t = float(row["time_s"])
        for state in sim.actor_model.states(t):
            entry = actor_series.setdefault(state.id, {"id": state.id, "class": state.object_class, "points": []})
            entry["points"].append({
                "t": _round(t), "x": _round(state.x_m), "y": _round(state.y_m),
                "s": _round(state.s_m), "d": _round(state.d_m), "v": _round(state.speed_mps),
            })
    max_s = max(float(row["road_s_m"]) for row in ts) + 120.0
    scenario = sim.scenario
    result = sim.evaluate_expectations()
    payload = {
        "scenario": {
            "id": scenario.id, "family": scenario.family, "title": scenario.title,
            "category": scenario.category, "requirements": scenario.requirements,
            "seed": scenario.seed, "duration_s": scenario.duration_s,
            "expectations": scenario.expectations,
        },
        "result": result,
        "road": _road_payload(sim, max_s),
        "ego": ego,
        "timeseries": timeseries,
        "trajectories": trajectory_list,
        "actors": list(actor_series.values()),
        "events": [
            {"t": _round(event["time_s"], 3), "event": event["event"], "detail": event["detail"]}
            for event in sim.events
        ],
    }
    path.write_text(
        json.dumps(payload, separators=(",", ":"), default=_json_default),
        encoding="utf-8",
    )
    return path
