import json
from pathlib import Path

from l2_path_planner.simulation import ClosedLoopSimulation


ROOT = Path(__file__).parents[1]


def run_short():
    cfg = json.loads((ROOT / "configs/baseline.json").read_text())
    cfg["duration_s"] = 8.0
    sim = ClosedLoopSimulation(cfg, controller_backend="feedback")
    summary = sim.run(write_artifacts=False)
    return sim, summary


def test_closed_loop_is_deterministic_and_exercises_hwa_lka():
    first, summary_a = run_short()
    second, summary_b = run_short()
    assert summary_a["max_abs_lateral_error_m"] == summary_b["max_abs_lateral_error_m"]
    modes = {row["detail"] for row in first.events if row["event"] == "MODE"}
    assert {"HWA", "LKA"}.issubset(modes)
    assert all(len(first.trajectory.points) == 31 for _ in [0])
    assert summary_a["max_abs_lateral_error_m"] < 0.75
    assert summary_a["supervisor_intervention_fraction"] <= 0.10

