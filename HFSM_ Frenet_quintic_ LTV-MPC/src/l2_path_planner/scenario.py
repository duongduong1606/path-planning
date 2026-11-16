"""Scenario schema v2: loader, inheritance, variant expansion and validation.

A scenario file is a JSON object.  It may ``extend`` another file (path relative
to the file that declares it); dictionaries are merged recursively while lists
and scalars replace.  A file may also declare a ``variants`` list; every variant
is the base document plus the variant's own overrides, which keeps a family of
3-5 runs in one readable file.

The loader is deliberately independent from the planner: it only produces the
runtime configuration dictionary consumed by
:class:`l2_path_planner.simulation.ClosedLoopSimulation`.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 2
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO_DIR = ROOT / "scenarios"

REQUIRED_KEYS = (
    "id", "seed", "duration_s", "ego", "road", "vehicle", "behavior",
    "planner", "controller",
)
VALID_FUNCTIONS = {"HWA", "LKA", "ELK"}
VALID_LANE_ROLES = {"CAR", "TRUCK", "BUS", "MOTORBIKE"}
VALID_SEPARATORS = {"DOUBLE_YELLOW", "SINGLE_YELLOW", "NONE"}


class ScenarioError(ValueError):
    """Raised when a scenario document is malformed."""


def _deep_merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for key, value in override.items():
            merged[key] = _deep_merge(base.get(key), value) if key in base else copy.deepcopy(value)
        return merged
    if isinstance(override, list):
        return copy.deepcopy(override)
    return copy.deepcopy(override)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ScenarioError(f"scenario file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ScenarioError(f"invalid JSON in {path}: {exc}") from exc


def _resolve(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ScenarioError(f"cyclic extends chain: {path}")
    document = _read_json(path)
    parent = document.get("extends")
    document = {key: value for key, value in document.items() if key != "extends"}
    if parent:
        parent_path = (path.parent / parent).resolve()
        base = _resolve(parent_path, seen + (path,))
        document = _deep_merge(base, document)
    return document


def _variant_documents(document: dict) -> list[dict]:
    variants = document.get("variants")
    base = {key: value for key, value in document.items() if key not in ("variants", "base")}
    if "base" in document:
        base = _deep_merge(base, document["base"])
    if not variants:
        return [base]
    expanded: list[dict] = []
    for variant in variants:
        overrides = variant.get("overrides", {})
        metadata = {key: value for key, value in variant.items() if key not in ("overrides", "id", "title")}
        merged = _deep_merge(_deep_merge(base, metadata), overrides)
        if "id" in variant:
            merged["id"] = variant["id"]
        if "title" in variant:
            merged["title"] = variant["title"]
        expanded.append(merged)
    return expanded


def _normalise_environment(environment: list[dict], duration_s: float) -> list[dict]:
    """Map the readable ``environment`` block to disturbance profiles."""
    profiles: list[dict] = []
    for entry in environment or []:
        kind = str(entry.get("type", "")).lower()
        if kind == "crosswind":
            profiles.append({
                "type": "crosswind",
                "start_s": float(entry.get("start_s", 0.0)),
                "duration_s": float(entry.get("duration_s", 1.0)),
                "peak_wind_mps": float(entry.get("peak_mps", entry.get("peak_wind_mps", 10.0))),
                "direction": float(entry.get("direction", 1.0)),
                "ramp_fraction": float(entry.get("ramp_fraction", 0.25)),
                "turbulence_intensity": float(entry.get("turbulence_intensity", 0.0)),
                "center_of_pressure_x_m": float(entry.get("center_of_pressure_x_m", 0.35)),
            })
        elif kind == "road_bank":
            profiles.append({
                "type": "road_bank",
                "start_s": float(entry.get("start_s", 0.0)),
                "duration_s": float(entry.get("duration_s", 1.0)),
                "peak_angle_deg": float(entry.get("peak_angle_deg", 1.0)),
                "direction": float(entry.get("direction", 1.0)),
                "ramp_fraction": float(entry.get("ramp_fraction", 0.25)),
            })
        elif kind in ("rain", "wet", "friction_patch", "low_mu"):
            start = float(entry.get("start_s", 0.0))
            duration = float(entry.get("duration_s", max(duration_s - start, 0.0)))
            profiles.append({
                "type": "friction_patch",
                "start_s": start,
                "duration_s": duration,
                "minimum_mu": float(entry.get("mu", entry.get("minimum_mu", 0.5))),
                "ramp_fraction": float(entry.get("ramp_fraction", 0.15)),
            })
    return profiles


def _normalise_runtime(document: dict) -> dict:
    runtime = copy.deepcopy(document)
    runtime.setdefault("schema_version", SCHEMA_VERSION)
    ego = runtime.setdefault("ego", {})
    ego.setdefault("initial", {})
    vehicle = runtime.setdefault("vehicle", {})
    if "speed_mps" not in ego:
        ego["speed_mps"] = float(vehicle.get("speed_mps", 20.0))
    ego.setdefault("desired_speed_mps", ego["speed_mps"])
    vehicle["speed_mps"] = float(ego["speed_mps"])
    runtime.setdefault("actors", [])
    runtime.setdefault("adas", [])
    runtime.setdefault("expectations", {})
    runtime.setdefault("environment", [])
    if not any("desired_speed_mps" in entry for entry in runtime["adas"]):
        runtime["adas"] = [{"t": 0.0, "desired_speed_mps": ego["desired_speed_mps"]}] + list(runtime["adas"])
    runtime["disturbances"] = _normalise_environment(
        runtime["environment"], float(runtime["duration_s"]),
    )
    runtime.setdefault("output_dir", f"outputs/{runtime['id']}")
    runtime["scenario_id"] = runtime["id"]
    runtime.setdefault("family", str(runtime["id"]).split("-")[0])
    return runtime


@dataclass(slots=True)
class Scenario:
    id: str
    family: str
    title: str
    category: str
    requirements: list[str]
    seed: int
    duration_s: float
    runtime: dict
    source: Path | None = None

    @property
    def expectations(self) -> dict:
        return self.runtime.get("expectations", {})


def _validate_lanes(scenario_id: str, road: dict) -> None:
    """Check the optional ``road.lanes`` block.

    ``lanes`` is opt-in: a road without it keeps the legacy single-lane
    behaviour.  When present, ``widths_m[ego_index]`` is the lane the planner
    sees, so it must agree with ``lane_width_m``.
    """
    lanes = road.get("lanes")
    if lanes is None:
        return
    if not isinstance(lanes, dict):
        raise ScenarioError(f"{scenario_id}: road.lanes must be an object")
    widths = lanes.get("widths_m")
    if not isinstance(widths, list) or not widths:
        raise ScenarioError(f"{scenario_id}: road.lanes.widths_m must be a non-empty list")
    if len(widths) > 8:
        raise ScenarioError(f"{scenario_id}: road.lanes.widths_m has too many lanes")
    for index, value in enumerate(widths):
        try:
            width = float(value)
        except (TypeError, ValueError):
            raise ScenarioError(f"{scenario_id}: lane {index} width is not a number")
        if width <= 0.5 or width > 7.0:
            raise ScenarioError(
                f"{scenario_id}: lane {index} width {width} m outside 0.5-7.0 m"
            )
    ego_index = int(lanes.get("ego_index", 0))
    if not 0 <= ego_index < len(widths):
        raise ScenarioError(f"{scenario_id}: road.lanes.ego_index out of range")
    roles = lanes.get("roles")
    if roles is not None:
        if not isinstance(roles, list) or len(roles) != len(widths):
            raise ScenarioError(f"{scenario_id}: road.lanes.roles must match widths_m")
        unknown = {str(role).upper() for role in roles} - VALID_LANE_ROLES
        if unknown:
            raise ScenarioError(f"{scenario_id}: unknown lane roles {sorted(unknown)}")
    lane_width = float(road.get("lane_width_m", float(widths[ego_index])))
    if abs(lane_width - float(widths[ego_index])) > 1e-6:
        raise ScenarioError(
            f"{scenario_id}: lane_width_m {lane_width} does not match the ego lane "
            f"width {float(widths[ego_index])}"
        )
    opposing = lanes.get("opposing")
    if opposing is not None:
        if not isinstance(opposing, dict):
            raise ScenarioError(f"{scenario_id}: road.lanes.opposing must be an object")
        count = int(opposing.get("count", 1))
        if count < 1:
            raise ScenarioError(f"{scenario_id}: opposing.count must be positive")
        width = float(opposing.get("width_m", lane_width))
        if width <= 0.5 or width > 7.0:
            raise ScenarioError(f"{scenario_id}: opposing lane width out of range")
        separator = str(opposing.get("separator", "DOUBLE_YELLOW")).upper()
        if separator not in VALID_SEPARATORS:
            raise ScenarioError(f"{scenario_id}: unknown separator {separator!r}")


def validate(document: dict) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in document]
    if missing:
        raise ScenarioError(f"{document.get('id', '<unknown>')}: missing keys {missing}")
    if not str(document["id"]):
        raise ScenarioError("scenario id must be a non-empty string")
    if float(document["duration_s"]) <= 0.0:
        raise ScenarioError(f"{document['id']}: duration_s must be positive")
    road = document["road"]
    for segment in road.get("centerline", []):
        kind = str(segment.get("type", "straight")).lower()
        if kind not in {"straight", "arc", "clothoid", "cosine"}:
            raise ScenarioError(f"{document['id']}: unknown segment type {kind!r}")
        if kind == "arc" and abs(float(segment.get("radius_m", 1.0))) < 1e-6:
            raise ScenarioError(f"{document['id']}: arc radius must be non-zero")
    for entry in road.get("markings", []):
        if float(entry.get("s_end_m", 1e9)) <= float(entry.get("s_start_m", 0.0)):
            raise ScenarioError(f"{document['id']}: marking s_end_m must exceed s_start_m")
    _validate_lanes(str(document["id"]), road)
    for actor in document.get("actors", []):
        trajectory = actor.get("trajectory", [])
        if not trajectory:
            raise ScenarioError(f"{document['id']}: actor {actor.get('id')} has no trajectory")
        times = [float(point["t"]) for point in trajectory]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ScenarioError(f"{document['id']}: actor {actor.get('id')} times must increase")
    expectations = document.get("expectations", {})
    for key in ("must_activate", "must_not_activate"):
        unknown = set(expectations.get(key, [])) - VALID_FUNCTIONS
        if unknown:
            raise ScenarioError(f"{document['id']}: unknown functions in {key}: {sorted(unknown)}")
    window = expectations.get("activation_window_s")
    if window and float(window[1]) > float(document["duration_s"]) + 1e-6:
        raise ScenarioError(f"{document['id']}: activation window exceeds duration")


def load_dict(document: dict, source: Path | None = None) -> Scenario:
    validate(document)
    runtime = _normalise_runtime(document)
    return Scenario(
        id=str(runtime["id"]),
        family=str(runtime.get("family", str(runtime["id"]).split("-")[0])),
        title=str(runtime.get("title", runtime["id"])),
        category=str(runtime.get("category", "")),
        requirements=list(runtime.get("requirements", [])),
        seed=int(runtime["seed"]),
        duration_s=float(runtime["duration_s"]),
        runtime=runtime,
        source=source,
    )


def load_all(path: str | Path) -> list[Scenario]:
    resolved = Path(path).resolve()
    document = _resolve(resolved)
    return [load_dict(item, resolved) for item in _variant_documents(document)]


def load(path: str | Path, scenario_id: str | None = None) -> Scenario:
    scenarios = load_all(path)
    if scenario_id is None:
        return scenarios[0]
    for scenario in scenarios:
        if scenario.id == scenario_id:
            return scenario
    raise ScenarioError(f"{scenario_id!r} not found in {path}")


def from_legacy(config: dict) -> Scenario:
    """Wrap an old flat config so stray callers keep working."""
    config = copy.deepcopy(config)
    stimuli = config.get("behavior_stimuli", {})
    road = dict(config.get("road", {}))
    actors: list[dict] = []
    if "truck_start_s" in stimuli:
        side = str(stimuli.get("truck_side", "LEFT")).upper()
        d = -float(road.get("lane_width_m", 3.5)) if side != "RIGHT" else float(road.get("lane_width_m", 3.5))
        actors.append({
            "id": "truck", "class": "TRUCK",
            "trajectory": [
                {"t": float(stimuli.get("truck_start_s", 0.0)),
                 "s": float(stimuli.get("truck_x_m", 20.0)), "d": float(stimuli.get("truck_y_m", d)),
                 "v": float(config.get("vehicle", {}).get("speed_mps", 20.0)) - float(stimuli.get("truck_delta_speed_mps", 0.0))},
                {"t": float(stimuli.get("truck_end_s", 0.0)),
                 "s": float(stimuli.get("truck_x_m", 20.0)) + 30.0, "d": float(stimuli.get("truck_y_m", d)),
                 "v": float(config.get("vehicle", {}).get("speed_mps", 20.0)) - float(stimuli.get("truck_delta_speed_mps", 0.0))},
            ],
        })
    if "lead_start_s" in stimuli:
        actors.append({
            "id": "lead", "class": "CAR",
            "trajectory": [
                {"t": float(stimuli.get("lead_start_s", 0.0)), "s": float(stimuli.get("lead_gap_m", 30.0)),
                 "d": 0.0, "v": float(stimuli.get("lead_speed_mps", 17.0))},
                {"t": float(stimuli.get("lead_end_s", 0.0)), "s": float(stimuli.get("lead_gap_m", 30.0)) + 20.0,
                 "d": 0.0, "v": float(stimuli.get("lead_speed_mps", 17.0))},
            ],
        })
    document = {
        "schema_version": SCHEMA_VERSION,
        "id": str(config.get("id", "legacy")),
        "family": "legacy",
        "title": str(config.get("title", "legacy config")),
        "seed": int(config.get("seed", 7)),
        "duration_s": float(config.get("duration_s", 18.0)),
        "simulation_dt_s": float(config.get("simulation_dt_s", 0.005)),
        "planner_dt_s": float(config.get("planner_dt_s", 0.025)),
        "lane_model_dt_s": float(config.get("lane_model_dt_s", 0.10)),
        "lane_capture_to_send_s": float(config.get("lane_capture_to_send_s", 0.03)),
        "state_estimator_dt_s": float(config.get("state_estimator_dt_s", 0.01)),
        "path_point_dt_s": float(config.get("path_point_dt_s", 0.1)),
        "path_horizon_s": float(config.get("path_horizon_s", 3.0)),
        "ego": {
            "speed_mps": float(config.get("vehicle", {}).get("speed_mps", 20.0)),
            "desired_speed_mps": float(stimuli.get("desired_speed_mps", config.get("vehicle", {}).get("speed_mps", 20.0))),
            "initial": {},
        },
        "road": road,
        "actors": actors,
        "environment": config.get("environment", config.get("disturbances", [])),
        "adas": [{"t": float(stimuli.get("hwa_button_time_s", 1.0)), "hwa": True}],
        "vehicle": config.get("vehicle", {}),
        "behavior": config.get("behavior", {}),
        "planner": config.get("planner", {}),
        "controller": config.get("controller", {}),
        "expectations": {},
    }
    return load_dict(document)


def coerce(config: dict | Scenario) -> Scenario:
    if isinstance(config, Scenario):
        return config
    if int(config.get("schema_version", 0)) == SCHEMA_VERSION:
        return load_dict(config)
    return from_legacy(config)


def registry(root: str | Path | None = None) -> dict[str, Scenario]:
    root = Path(root) if root else DEFAULT_SCENARIO_DIR
    found: dict[str, Scenario] = {}
    if not root.exists():
        return found
    for path in sorted(root.rglob("*.json")):
        if "_base" in path.parts or path.name in ("catalog.json",) or path.name.startswith("_"):
            continue
        for scenario in load_all(path):
            if scenario.id in found:
                raise ScenarioError(f"duplicate scenario id {scenario.id!r}")
            found[scenario.id] = scenario
    return found


def find(scenario_id: str, root: str | Path | None = None) -> Scenario:
    scenarios = registry(root)
    if scenario_id not in scenarios:
        raise ScenarioError(f"scenario {scenario_id!r} is not in the library")
    return scenarios[scenario_id]


def list_scenarios(root: str | Path | None = None) -> list[dict]:
    return [
        {
            "id": scenario.id,
            "family": scenario.family,
            "title": scenario.title,
            "category": scenario.category,
            "duration_s": scenario.duration_s,
            "seed": scenario.seed,
            "requirements": scenario.requirements,
        }
        for scenario in sorted(registry(root).values(), key=lambda item: item.id)
    ]
