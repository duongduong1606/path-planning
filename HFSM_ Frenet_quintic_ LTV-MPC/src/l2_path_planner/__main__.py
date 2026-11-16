from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .scenario import ScenarioError, list_scenarios, load_all, registry
from .simulation import ClosedLoopSimulation

ROOT = Path(__file__).resolve().parents[2]
UI_DIR = ROOT / "ui"


def _dump(payload) -> str:
    import numpy as np

    def default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    return json.dumps(payload, indent=2, default=default)


def _resolve_scenario(args) -> tuple[object, Path | None]:
    if getattr(args, "scenario", None):
        scenarios = registry()
        if args.scenario not in scenarios:
            raise ScenarioError(f"unknown scenario id {args.scenario!r}")
        return scenarios[args.scenario], None
    path = Path(getattr(args, "config", "configs/baseline.json"))
    scenarios = load_all(path)
    return scenarios[0], path


def _run(args) -> None:
    scenario, config_path = _resolve_scenario(args)
    simulation = ClosedLoopSimulation(scenario, config_path, args.controller)
    if args.duration is not None:
        simulation.cfg["duration_s"] = args.duration
    summary = simulation.run(args.output, write_artifacts=True)
    print(_dump(summary))


def _scenarios(args) -> None:
    if args.action == "list":
        print(_dump(list_scenarios()))
        return
    scenarios = registry()
    failures = []
    for scenario in scenarios.values():
        try:
            from .scenario import validate

            validate(scenario.runtime)
        except ScenarioError as exc:  # pragma: no cover - defensive
            failures.append({"id": scenario.id, "error": str(exc)})
    payload = {"count": len(scenarios), "failures": failures}
    print(_dump(payload))
    if failures:
        raise SystemExit(1)


def _ui(args) -> None:
    UI_DIR.mkdir(parents=True, exist_ok=True)
    library = json.dumps(list_scenarios(), separators=(",", ":"))
    (UI_DIR / "scenarios.js").write_text(
        f"window.SCENARIO_LIBRARY = {library};\n", encoding="utf-8",
    )

    if getattr(args, "all", False):
        from .ui_export import export_ui_data

        data_dir = UI_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        exported: list[str] = []
        for sid, scenario in sorted(registry().items()):
            sim = ClosedLoopSimulation(scenario, None, args.controller)
            sim.run(write_artifacts=False)
            tmp = data_dir / f"{sid}.json"
            export_ui_data(sim, tmp)
            payload = tmp.read_text(encoding="utf-8")
            (data_dir / f"{sid}.js").write_text(
                f"window.SCENARIO_DATA = {payload};\n", encoding="utf-8",
            )
            tmp.unlink()
            exported.append(sid)
        if exported:
            default_id = args.scenario if args.scenario in exported else exported[0]
            shutil.copyfile(data_dir / f"{default_id}.js", UI_DIR / "data.js")
        print(_dump({"exported": len(exported), "ui": str(UI_DIR / "index.html")}))
        return

    scenario, config_path = _resolve_scenario(args)
    output_dir = Path(args.output or f"outputs/{scenario.id}")
    simulation = ClosedLoopSimulation(scenario, config_path, args.controller)
    simulation.run(output_dir, write_artifacts=True)
    data = (output_dir / "ui_data.json").read_text(encoding="utf-8")
    (UI_DIR / "data.js").write_text(f"window.SCENARIO_DATA = {data};\n", encoding="utf-8")
    print(_dump({
        "scenario": scenario.id,
        "output_dir": str(output_dir),
        "ui": str(UI_DIR / "index.html"),
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description="L2 lateral closed-loop simulation")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("--config", default="configs/baseline.json")
    run.add_argument("--scenario", default=None, help="Scenario id from the library, e.g. S4-a")
    run.add_argument("--output", default=None)
    run.add_argument("--duration", type=float, default=None, help="Override simulation duration in seconds")
    run.add_argument("--controller", choices=["auto", "osqp_native", "cvxpy", "feedback"], default="auto")
    run.set_defaults(func=_run)

    scenarios = sub.add_parser("scenarios")
    scenarios.add_argument("action", choices=["list", "validate"])
    scenarios.set_defaults(func=_scenarios)

    ui = sub.add_parser("ui")
    ui.add_argument("--config", default="configs/baseline.json")
    ui.add_argument("--scenario", default=None)
    ui.add_argument("--output", default=None)
    ui.add_argument("--all", action="store_true", help="Export every scenario into ui/data/<id>.js for the selector")
    ui.add_argument("--controller", choices=["auto", "osqp_native", "cvxpy", "feedback"], default="feedback")
    ui.set_defaults(func=_ui)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
