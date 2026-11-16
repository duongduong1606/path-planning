import json
import math
from pathlib import Path

from l2_path_planner.controller import LTVMPCController
from l2_path_planner.messages import Function


ROOT = Path(__file__).parents[1]


def test_supervisor_preserves_twenty_percent_added_acceleration_reserve():
    cfg = json.loads((ROOT / "configs/baseline.json").read_text())
    controller = LTVMPCController(cfg["controller"], cfg["vehicle"], 0.01, "feedback")
    command, diagnostics = controller._supervise(1.0, Function.HWA, 0.0, 20.0)
    predicted_added_ay = 20.0**2 * math.tan(command) / cfg["vehicle"]["wheelbase_m"]
    assert predicted_added_ay <= 0.4
    assert diagnostics["supervisor_limited"]


def test_constraint_aware_feedback_does_not_need_supervisor_clipping():
    cfg = json.loads((ROOT / "configs/baseline.json").read_text())
    controller = LTVMPCController(cfg["controller"], cfg["vehicle"], 0.01, "feedback")
    from l2_path_planner.messages import Header, Trajectory, TrajectoryPoint
    trajectory = Trajectory(Header(0, 0), [
        TrajectoryPoint(0.1 * i, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for i in range(31)
    ])
    _, diagnostics = controller.command(
        trajectory, 1.0, 0.2, 0.5, 0.2, 20.0, Function.HWA, 0.0,
    )
    assert diagnostics["controller_constraint_active"]
    assert not diagnostics["supervisor_limited"]

