"""Experimental closed-loop test of TCN40HzAdapter inside HFSM_Frenet_quintic_LTV-MPC."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Callable

import numpy as np

HFSM_ROOT = Path(r"C:\Users\PC\Path_planning\HFSM_ Frenet_quintic_ LTV-MPC")
sys.path.insert(0, str(HFSM_ROOT / "src"))

from l2_path_planner.messages import Function, LateralManeuver
from l2_path_planner.planner import Candidate, FrenetPlanner, Quintic
from l2_path_planner.scenario import load_all
from l2_path_planner.simulation import ClosedLoopSimulation

from tcn_runtime import TCN40HzAdapter


class TCNHybridPlanner(FrenetPlanner):
    """FrenetPlanner augmented with a 40 Hz TCN candidate in HWA LANE_CENTER mode."""

    def __init__(
        self,
        config: dict,
        road,
        vehicle_config: dict,
        bundle_path: Path,
        steering_getter: Callable[[], float],
        guarded: bool = True,
    ) -> None:
        super().__init__(config, road, vehicle_config)
        self.tcn = TCN40HzAdapter(
            bundle_path=bundle_path,
            steering_ratio=float(vehicle_config.get("steering_ratio", 15.0)),
            lane_width_m=float(road.lane_width_m),
            infer_every_n_steps=2,  # 20 Hz inference inside 40 Hz loop
        )
        self.steering_getter = steering_getter
        self.guarded = guarded
        self._last_yaw_rate = 0.0
        self.tcn_used_steps = 0
        self.tcn_feasible_steps = 0
        self.hwa_steps = 0

    def update_estimation(self, lane, speed_mps: float, yaw_rate_rps: float, lateral_acceleration_mps2: float) -> None:
        self._last_yaw_rate = float(yaw_rate_rps)
        super().update_estimation(lane, speed_mps, yaw_rate_rps, lateral_acceleration_mps2)

    def _build_tcn_candidate(
        self,
        now_s: float,
        lane,
        road_s: float,
        speed: float,
        initial: tuple[float, float, float],
        ego_x_m: float,
        ego_y_m: float,
        ego_yaw_rad: float,
        lateral_accel_mps2: float,
    ) -> Candidate:
        pred = self.tcn.step(
            stamp_s=now_s,
            left_line=lane.left,
            right_line=lane.right,
            speed_mps=speed,
            yaw_rate_rps=self._last_yaw_rate,
            lateral_accel_mps2=lateral_accel_mps2,
            front_wheel_angle_rad=self.steering_getter(),
        )
        y_T, y_dot_T, y_ddot_T, _ = [float(v) for v in pred]
        T = 2.5
        # Convert ego-frame terminal (speed*T, y_T) to world -> Frenet d_T
        cos_y, sin_y = math.cos(ego_yaw_rad), math.sin(ego_yaw_rad)
        x_long = speed * T
        wx = ego_x_m + x_long * cos_y - y_T * sin_y
        wy = ego_y_m + x_long * sin_y + y_T * cos_y
        _, d_T, _, _ = self.road.project(wx, wy)

        poly = Quintic.from_boundary(*initial, d_T, y_dot_T, 0.0, T)
        eval_t = np.minimum(self.t, T)
        d, d_dot, d_ddot, jerk = poly.evaluate(eval_t)
        after = self.t >= T
        d[after], d_dot[after], d_ddot[after], jerk[after] = d_T, 0.0, 0.0, 0.0
        s = road_s + speed * self.t
        x, y, yaw, kappa_center = self.road.frenet_to_global(s, d, d_dot, speed)
        curvature = kappa_center + d_ddot / max(speed * speed, 1.0)
        front_angle = np.arctan(self.vehicle["wheelbase_m"] * curvature)
        cost = 8.0 * float(np.mean(jerk * jerk)) + 50.0 * (d[-1] ** 2) + 0.03 * T
        return Candidate(T, self.t.copy(), s, d, d_dot, d_ddot, jerk, x, y, yaw, curvature, front_angle, cost)

    def plan(self, now_s, lane, requests, state, ego_x_m, ego_y_m, ego_yaw_rad, speed_mps, lateral_acceleration_mps2=0.0):
        # Always keep TCN history buffer warm at 40 Hz, then optionally override HWA
        road_s, _, _, _ = self.road.project(ego_x_m, ego_y_m)
        traj, status, limits, diag = super().plan(
            now_s, lane, requests, state, ego_x_m, ego_y_m, ego_yaw_rad, speed_mps, lateral_acceleration_mps2
        )
        req = requests.request_for(status.active_function)
        maneuver = req.maneuver if req is not None else LateralManeuver.NONE
        if status.active_function == Function.HWA and maneuver in (LateralManeuver.NONE, LateralManeuver.LANE_CENTER):
            self.hwa_steps += 1
            initial = self._initial_state(road_s, diag["road_d_m"], diag["heading_error_rad"], state, Function.HWA)
            tcn_cand = self._build_tcn_candidate(
                now_s, lane, road_s, speed_mps, initial, ego_x_m, ego_y_m, ego_yaw_rad, lateral_acceleration_mps2
            )
            is_feas = self._feasible(tcn_cand, Function.HWA, speed_mps, diag["target_d_m"], maneuver)
            if is_feas:
                self.tcn_feasible_steps += 1
            if (not self.guarded) or is_feas:
                self.tcn_used_steps += 1
                selected = self._continuity_blend(tcn_cand)
                from l2_path_planner.messages import Trajectory, TrajectoryPoint
                from l2_path_planner.road import wrap_angle

                points = [
                    TrajectoryPoint(
                        time_from_start_s=float(selected.t[i]),
                        x_m=float(selected.x[i] - ego_x_m),
                        y_m=float(selected.y[i] - ego_y_m),
                        yaw_rad=float(wrap_angle(np.array([selected.yaw[i] - ego_yaw_rad]))[0]),
                        longitudinal_velocity_mps=speed_mps,
                        lateral_velocity_mps=float(selected.d_dot[i]),
                        acceleration_mps2=0.0,
                        heading_rate_rps=float(speed_mps * selected.curvature[i]),
                        curvature_inv_m=float(selected.curvature[i]),
                        front_wheel_angle_rad=float(selected.front_angle[i]),
                        road_s_m=float(selected.s[i]),
                        road_d_m=float(selected.d[i]),
                    )
                    for i in range(len(self.t))
                ]
                traj = Trajectory(traj.header, points)
                self.last_trajectory = traj
        else:
            # Keep 40 Hz buffer updated even during LKA/ELK
            self.tcn.step(
                stamp_s=now_s,
                left_line=lane.left,
                right_line=lane.right,
                speed_mps=speed_mps,
                yaw_rate_rps=self._last_yaw_rate,
                lateral_accel_mps2=lateral_acceleration_mps2,
                front_wheel_angle_rad=self.steering_getter(),
            )
        return traj, status, limits, diag


def run_one(
    scenario_path: Path,
    mode: str,
    bundle_path: Path | None = None,
    guarded: bool = True,
    output_dir: Path | None = None,
    export_ui: bool = False,
    controller_backend: str = "auto",
) -> dict:
    sc = load_all(scenario_path)[0]
    sim = ClosedLoopSimulation(sc, scenario_path, controller_backend=controller_backend)
    if bundle_path is not None:
        sim.planner = TCNHybridPlanner(
            sim.cfg, sim.road, sim.vehicle_cfg,
            bundle_path=bundle_path,
            steering_getter=lambda: sim.plant.state.steering_angle_rad,
            guarded=guarded,
        )
    write_artifacts = (output_dir is not None) or export_ui
    if write_artifacts and output_dir is None:
        output_dir = HFSM_ROOT / "outputs" / f"{scenario_path.stem}_{mode}"
    summary = sim.run(output_dir=output_dir, write_artifacts=write_artifacts)
    if export_ui and output_dir is not None:
        ui_dir = HFSM_ROOT / "ui"
        ui_dir.mkdir(parents=True, exist_ok=True)
        data = (Path(output_dir) / "ui_data.json").read_text(encoding="utf-8")
        (ui_dir / "data.js").write_text(f"window.SCENARIO_DATA = {data};\n", encoding="utf-8")

    d = np.array([r["road_d_m"] for r in sim.timeseries], dtype=float)
    ay = np.array([r["lateral_acceleration_mps2"] for r in sim.timeseries], dtype=float)
    margins = np.array([r["lane_margin_m"] - abs(r["road_d_m"]) for r in sim.timeseries], dtype=float)
    res = {
        "scenario": scenario_path.stem,
        "mode": mode,
        "max_err": round(float(summary.get("max_abs_lateral_error_m", np.max(np.abs(d)) if len(d) else 0.0)), 4),
        "rms_err": round(float(summary.get("rms_lateral_error_m", np.sqrt(np.mean(d * d)) if len(d) else 0.0)), 4),
        "max_ay": round(float(summary.get("max_abs_lateral_acceleration_mps2", np.max(np.abs(ay)) if len(ay) else 0.0)), 3),
        "sup_int": round(float(summary.get("supervisor_intervention_fraction", 0.0)) * 100, 2),
        "depart": bool(summary.get("lane_departure", bool(np.any(margins < 0.0)) if len(margins) else False)),
        "passed": summary.get("expectations", {}).get("passed"),
    }
    if isinstance(sim.planner, TCNHybridPlanner):
        res["hwa_steps"] = sim.planner.hwa_steps
        res["tcn_feas"] = sim.planner.tcn_feasible_steps
        res["tcn_used"] = sim.planner.tcn_used_steps
    if output_dir is not None:
        res["output_dir"] = str(output_dir)
    if export_ui:
        res["ui"] = str(HFSM_ROOT / "ui" / "index.html")
    return res


def _resolve_scenario_path(name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p
    if name == "baseline":
        return HFSM_ROOT / "configs" / "baseline.json"
    candidate = HFSM_ROOT / "scenarios" / f"{name}.json"
    if candidate.exists():
        return candidate
    for f in (HFSM_ROOT / "scenarios").glob("*.json"):
        if name.lower() in f.stem.lower():
            return f
    raise FileNotFoundError(f"Cannot resolve scenario: {name}")


def main() -> None:
    import argparse
    import shutil
    from l2_path_planner.scenario import list_scenarios, registry
    from l2_path_planner.ui_export import export_ui_data

    parser = argparse.ArgumentParser(description="Evaluate or demo TCNHybridPlanner inside HFSM_Frenet_quintic_LTV-MPC")
    parser.add_argument("--scenario", default=None, help="Scenario name/file (e.g. baseline, s01_hwa_curve, s04_lka_drift)")
    parser.add_argument(
        "--mode",
        choices=["frenet_baseline", "tcn_safe_guarded", "tcn_safe_raw_hwa", "tcn_exp_raw_hwa"],
        default="tcn_safe_guarded",
        help="Planner mode when running scenario/UI export",
    )
    parser.add_argument("--ui", action="store_true", help="Export dashboard PNGs and update HFSM ui/data.js for browser demo")
    parser.add_argument("--all", action="store_true", help="Run and export ALL scenarios into ui/data/<id>.js for the Web UI dropdown")
    parser.add_argument("--output", default=None, help="Optional output directory for artifacts")
    parser.add_argument("--controller", choices=["auto", "osqp_native", "cvxpy", "feedback"], default="auto")
    args = parser.parse_args()

    b_safe = Path("artifacts/tcn_lane_safe/lateral_tcn_runtime.npz")
    b_exp = Path("artifacts/tcn_frenet_expert_v2/lateral_tcn_runtime.npz")
    mode_map = {
        "frenet_baseline": (None, True),
        "tcn_safe_guarded": (b_safe, True),
        "tcn_safe_raw_hwa": (b_safe, False),
        "tcn_exp_raw_hwa": (b_exp, False),
    }

    if args.all:
        bundle, guarded = mode_map[args.mode]
        ui_dir = HFSM_ROOT / "ui"
        ui_dir.mkdir(parents=True, exist_ok=True)
        library = json.dumps(list_scenarios(), separators=(",", ":"))
        (ui_dir / "scenarios.js").write_text(
            f"window.SCENARIO_LIBRARY = {library};\n", encoding="utf-8",
        )
        data_dir = ui_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        exported: list[str] = []
        passed_count = 0
        for sid, scenario in sorted(registry().items()):
            sim = ClosedLoopSimulation(scenario, None, args.controller)
            if bundle is not None:
                sim.planner = TCNHybridPlanner(
                    sim.cfg, sim.road, sim.vehicle_cfg,
                    bundle_path=bundle,
                    steering_getter=lambda: sim.plant.state.steering_angle_rad,
                    guarded=guarded,
                )
            summary = sim.run(write_artifacts=False)
            if summary.get("expectations", {}).get("passed"):
                passed_count += 1
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
            shutil.copyfile(data_dir / f"{default_id}.js", ui_dir / "data.js")
        print(json.dumps({
            "mode": args.mode,
            "controller": args.controller,
            "exported": len(exported),
            "passed": passed_count,
            "ui": str(ui_dir / "index.html"),
        }, indent=2))
        return

    if args.scenario or args.ui:
        sc_path = _resolve_scenario_path(args.scenario or "baseline")
        bundle, guarded = mode_map[args.mode]
        out_dir = Path(args.output) if args.output else None
        res = run_one(
            sc_path,
            args.mode,
            bundle_path=bundle,
            guarded=guarded,
            output_dir=out_dir,
            export_ui=args.ui,
            controller_backend=args.controller,
        )
        print(json.dumps(res, indent=2))
        return

    scenarios = [
        HFSM_ROOT / "configs" / "baseline.json",
        HFSM_ROOT / "scenarios" / "s01_hwa_curve.json",
        HFSM_ROOT / "scenarios" / "s01b_sharp_curve.json",
        HFSM_ROOT / "scenarios" / "s04_lka_drift.json",
    ]

    rows = []
    for sc in scenarios:
        for mode_name, (bundle, guarded) in mode_map.items():
            rows.append(run_one(sc, mode_name, bundle, guarded=guarded, controller_backend=args.controller))

    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
