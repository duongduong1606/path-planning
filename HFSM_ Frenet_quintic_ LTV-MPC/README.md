# L2/L2+ lateral path-planning prototype

Deterministic Python closed loop combining an HFSM (`ELK > LKA > HWA`), analytic Frenet-quintic sampling, an LTV dynamic-bicycle MPC, actuator delay, and a dynamic-bicycle plant.

## Summary

- **Goal** — a verifiable L2/L2+ lateral stack: HFSM decides *when and why*,
  Frenet quintic selects a feasible path, LTV-MPC tracks it, and every KPI is
  measured in a deterministic closed loop.
- **Approach** — 40 Hz hierarchical SFM (`OFF`, `STANDBY`, `ACTIVE`, `DEGRADED`,
  `FAULT`; parallel lateral/longitudinal regions) → analytic quintic candidates in
  Frenet coordinates → LTV dynamic-bicycle MPC (native OSQP warm start, CVXPY
  reference, feedback fallback) → actuator delay → bicycle plant, with crosswind,
  bank/camber and low-friction disturbances and an independent safety supervisor.
- **Progress** — baseline (18 s, 3,601 samples): max lateral error 0.55 m, RMS
  0.28 m, planner p99 1.05 ms, controller p99 1.61 ms, supervisor interventions
  0 %, all quality reserves ≥ 20 %. 23 deterministic scenario configs
  (HWA/LKA/ELK, faults, low friction, EPS delay, stop-and-go), static HTML/JS UI
  exporter, 12 pytest modules, and a C reference planner
  (`c_planner/pp_planner.c`) as a draft.
- **Next** — CI run over all 23 scenarios, perception/object inputs for lane
  changes and avoidance, and timing on target hardware.

## Run

```powershell
python -m pip install -e ".[test]"
python -m l2_path_planner run --config configs/baseline.json
pytest -q
```

The default backend is a preallocated native OSQP QP with warm start. Use
`--controller feedback` for a dependency-light smoke run or `--controller cvxpy`
for the reference formulation. A mode-aware safety supervisor derates lateral
acceleration, added acceleration, jerk, steering rate, and angle to retain at
least 20% of the configured design budgets. The baseline run writes CSV logs,
`summary.json`, `kpi_dashboard.png`, and `disturbance_dashboard.png` under
`outputs/baseline`. Environment loads include smooth turbulent crosswind,
aerodynamic yaw moment, road bank/camber, and a finite low-friction patch.
The same mode-aware steering corridor is enforced inside OSQP/CVXPY and the
feedback fallback. The external supervisor remains an independent backstop;
its intervention fraction is recorded as a KPI and must stay at or below 10%.

## Hierarchical SFM behavior planner

`HierarchicalSFM` replaces the scripted request source and runs at 40 Hz. Its
top-level system states are `OFF`, `STANDBY`, `ACTIVE`, `DEGRADED`, and `FAULT`.
Two parallel regions keep lateral and longitudinal decisions independent while
sharing validity, speed, lane, driver, and hazard inputs:

- Lateral: HWA lane centering, truck/object offset, latched lane change, LKA,
  ELK for solid line/road edge or adjacent object, driver override, cooldown,
  and `PathStatus` completion feedback. Arbitration remains in the path planner
  with `ELK > LKA > HWA`.
- Longitudinal: cruise, accelerate, lead-vehicle follow, comfortable braking,
  and emergency braking. It publishes `LongitudinalPlan`; the lateral planner
  still only owns lateral trajectory generation.

Offset, lane change, LKA, and ELK publish a target and maneuver through the same
`FunctionRequests[3]` interface. The Frenet planner therefore draws every
lateral maneuver with analytic quintic candidates instead of applying a direct
steering command. The SFM decides *when and why*; Frenet selects a feasible
path; LTV-MPC tracks it.

The baseline stimulus contains ordinary environmental inputs (HWA button,
nearby truck, lead vehicle, lane markings, and physical disturbances), not a
time table of function requests. It can be extended with perception and HMI
sources without changing the SFM/planner interfaces.

