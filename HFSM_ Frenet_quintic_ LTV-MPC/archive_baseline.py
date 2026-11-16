"""Archive baseline timeseries before mirror transformation."""
import json
from pathlib import Path
from l2_path_planner.simulation import ClosedLoopSimulation

ROOT = Path("C:\\Users\\PC\\Path_planning\\HFSM_ Frenet_quintic_ LTV-MPC")

cfg = json.loads((ROOT / "configs/baseline.json").read_text())
cfg["duration_s"] = 8.0

sim = ClosedLoopSimulation(cfg.copy(), controller_backend="feedback")
sim.run(write_artifacts=True)

# Save baseline timeseries for comparison
baseline_path = ROOT / "baseline" / "timeseries.csv"
baseline_path.parent.mkdir(parents=True, exist_ok=True)
sim.simulation.logger._timeseries.to_csv(baseline_path, index=False)

print(f"Baseline saved to: {baseline_path}")
print(f"Timeseries shape: {sim.timeseries.shape}")
print(f"Columns: {list(sim.timeseries.columns)[:10]}...")
