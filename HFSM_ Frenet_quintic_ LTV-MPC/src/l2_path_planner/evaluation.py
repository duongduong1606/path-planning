from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _json_default(value):
    """Convert numpy scalars/arrays so summaries are always JSON-safe."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_outputs(
    output_dir: Path,
    timeseries: list[dict],
    trajectories: list[dict],
    events: list[dict],
    extra: dict | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.DataFrame(timeseries)
    tr = pd.DataFrame(trajectories)
    ev = pd.DataFrame(events, columns=["time_s", "event", "detail"])
    ts.to_csv(output_dir / "timeseries.csv", index=False)
    tr.to_csv(output_dir / "trajectory_points.csv", index=False)
    ev.to_csv(output_dir / "events.csv", index=False)
    original = {1: {"ay": 2.5, "added": 0.5, "jerk": 5.0}, 2: {"ay": 3.0, "added": 1.5, "jerk": 5.0}, 3: {"ay": 4.0, "added": 3.0, "jerk": 10.0}}
    quality = {}
    steering_rate = np.abs(np.gradient(ts.steering_angle_rad, ts.time_s)) * 15.0
    for mode, name in ((1, "HWA"), (2, "LKA"), (3, "ELK")):
        part = ts[ts.active_function == mode]
        if part.empty:
            continue
        observed = {
            "ay_mps2": float(part.lateral_acceleration_mps2.abs().max()),
            "commanded_ay_added_mps2": float(part.commanded_added_acceleration_mps2.abs().max()),
            "raw_jerk_mps3": float(part.lateral_jerk_mps3.abs().max()),
        }
        reserves = {
            "ay": 1.0 - observed["ay_mps2"] / original[mode]["ay"],
            "ay_added": 1.0 - observed["commanded_ay_added_mps2"] / original[mode]["added"],
            "jerk": 1.0 - observed["raw_jerk_mps3"] / original[mode]["jerk"],
        }
        quality[name] = {**observed, "reserve_fraction": reserves, "reserve_at_least_20_percent": min(reserves.values()) >= 0.2 - 1e-6}
    steering_reserve = 1.0 - float(np.max(steering_rate)) / 6.0
    controller_deadline_reserve = 1.0 - float(np.percentile(ts.controller_time_ms[ts.controller_time_ms > 0], 99)) / 10.0
    lane_reserve = 1.0 - float(ts.road_d_m.abs().max() / ts.lane_margin_m.iloc[0])
    summary = {
        "samples": int(len(ts)),
        "duration_s": float(ts.time_s.iloc[-1]) if len(ts) else 0.0,
        "max_abs_lateral_error_m": float(ts.road_d_m.abs().max()),
        "rms_lateral_error_m": float(np.sqrt(np.mean(ts.road_d_m**2))),
        "max_abs_lateral_acceleration_mps2": float(ts.lateral_acceleration_mps2.abs().max()),
        "max_abs_front_wheel_angle_rad": float(ts.steering_angle_rad.abs().max()),
        "planner_p99_ms": float(np.percentile(ts.planner_time_ms[ts.planner_time_ms > 0], 99)) if (ts.planner_time_ms > 0).any() else 0.0,
        "controller_p99_ms": float(np.percentile(ts.controller_time_ms[ts.controller_time_ms > 0], 99)) if (ts.controller_time_ms > 0).any() else 0.0,
        "supervisor_intervention_fraction": float(ts.supervisor_limited.mean()),
        "supervisor_below_10_percent": bool(ts.supervisor_limited.mean() <= 0.1),
        "controller_constraint_active_fraction": float(ts.controller_constraint_active.mean()),
        "lane_departure": bool((ts.road_d_m.abs() > ts.lane_margin_m).any()),
        "completed_functions": ev.loc[ev.event == "COMPLETED", "detail"].tolist(),
        "quality_reserve": quality,
        "steering_rate_reserve_fraction": steering_reserve,
        "controller_deadline_reserve_fraction": controller_deadline_reserve,
        "lane_geometric_reserve_fraction": lane_reserve,
        "all_quality_reserves_at_least_20_percent": bool(
            all(x["reserve_at_least_20_percent"] for x in quality.values())
            and steering_reserve >= 0.2 and controller_deadline_reserve >= 0.2
            and lane_reserve >= 0.2
        ),
    }
    if extra:
        summary = {**summary, **extra}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8",
    )
    _plot(output_dir, ts, tr)
    _plot_disturbances(output_dir, ts)
    return summary


def _plot(output_dir: Path, ts: pd.DataFrame, tr: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(ts.x_m, ts.y_m, label="actual", linewidth=2)
    if len(tr):
        last = tr[tr.planner_tick == tr.planner_tick.max()]
        ax.plot(last.global_x_m, last.global_y_m, "--", label="last command")
    ax.set(title="Scene and paths", xlabel="x [m]", ylabel="y [m]"); ax.axis("equal"); ax.legend()
    axes[0, 1].plot(ts.time_s, ts.road_d_m, label="lateral error")
    axes[0, 1].step(ts.time_s, ts.active_function, where="post", label="mode")
    axes[0, 1].set(title="Error and HFSM mode", xlabel="t [s]"); axes[0, 1].legend()
    axes[1, 0].plot(ts.time_s, ts.lateral_acceleration_mps2, label="ay")
    axes[1, 0].plot(ts.time_s, ts.lateral_jerk_mps3, label="jerk")
    axes[1, 0].set(title="Lateral dynamics", xlabel="t [s]"); axes[1, 0].legend()
    axes[1, 1].plot(ts.time_s, ts.steering_command_rad, label="command")
    axes[1, 1].plot(ts.time_s, ts.steering_angle_rad, label="actual")
    axes[1, 1].set(title="Front-wheel steering", xlabel="t [s]"); axes[1, 1].legend()
    axes[2, 0].plot(ts.time_s, ts.planner_time_ms, label="planner")
    axes[2, 0].plot(ts.time_s, ts.controller_time_ms, label="controller")
    axes[2, 0].set(title="Compute time", xlabel="t [s]", ylabel="ms"); axes[2, 0].legend()
    axes[2, 1].plot(ts.time_s, ts.continuity_position_m, label="position")
    axes[2, 1].plot(ts.time_s, ts.continuity_heading_rad, label="heading")
    axes[2, 1].plot(ts.time_s, ts.continuity_curvature_inv_m, label="curvature")
    axes[2, 1].set(title="Trajectory continuity", xlabel="t [s]"); axes[2, 1].legend()
    fig.savefig(output_dir / "kpi_dashboard.png", dpi=150)
    plt.close(fig)


def _plot_disturbances(output_dir: Path, ts: pd.DataFrame) -> None:
    required = {"crosswind_mps", "aerodynamic_force_n", "aerodynamic_yaw_moment_nm", "road_bank_angle_deg", "friction_coefficient"}
    if not required.issubset(ts.columns):
        return
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(ts.time_s, ts.crosswind_mps, label="crosswind [m/s]")
    force_ax = axes[0].twinx()
    force_ax.plot(ts.time_s, ts.aerodynamic_force_n, color="tab:red", alpha=0.75, label="side force [N]")
    axes[0].set_title("Aerodynamic crosswind load")
    axes[0].legend(loc="upper left"); force_ax.legend(loc="upper right")
    axes[1].plot(ts.time_s, ts.aerodynamic_yaw_moment_nm, label="yaw moment [Nm]")
    axes[1].plot(ts.time_s, ts.road_bank_angle_deg, label="road bank [deg]")
    axes[1].set_title("Yaw and road-bank disturbance"); axes[1].legend()
    axes[2].plot(ts.time_s, ts.friction_coefficient, label="friction coefficient μ")
    axes[2].set(title="Surface friction", xlabel="t [s]", ylim=(0.0, 1.05)); axes[2].legend()
    fig.savefig(output_dir / "disturbance_dashboard.png", dpi=150)
    plt.close(fig)

