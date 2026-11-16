from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from .messages import Function, Trajectory

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - exercised on minimal installations
    cp = None

try:
    import osqp
except ImportError:  # pragma: no cover
    osqp = None


DERATED_LIMITS = {
    # Slightly below 80% to preserve >=20% headroom after numerical tolerances.
    Function.NONE: (0.399, 1.99, 3.99),
    Function.HWA: (0.399, 1.99, 3.99),
    Function.LKA: (1.195, 2.39, 3.99),
    Function.ELK: (2.395, 3.19, 7.99),
}


class LTVMPCController:
    """Finite-horizon lateral controller using a speed-updated bicycle model."""

    def __init__(self, cfg: dict, vehicle_cfg: dict, dt_s: float, backend: str = "auto"):
        self.cfg, self.vehicle, self.dt = cfg, vehicle_cfg, dt_s
        self.n = int(cfg.get("horizon_steps", 50))
        self.previous_delta = 0.0
        self.solve_times_ms: list[float] = []
        self.understeer_gradient = self._understeer_gradient()
        self.backend = "osqp_native" if backend == "auto" and osqp is not None else backend
        if self.backend == "cvxpy" and cp is None:
            self.backend = "feedback"
        if self.backend == "osqp_native" and osqp is None:
            self.backend = "feedback"
        if self.backend == "cvxpy":
            self._build_problem()
        elif self.backend == "osqp_native":
            self._build_native_problem(float(vehicle_cfg["speed_mps"]))

    def _understeer_gradient(self) -> float:
        """Steady-state understeer gradient ``K_us`` in s²/m."""
        p = self.vehicle
        wheelbase = float(p["wheelbase_m"])
        rear = float(p["rear_axle_to_cg_m"])
        front = wheelbase - rear
        mass = float(p["mass_kg"])
        cf = float(p["cornering_stiffness_front_nprad"])
        cr = float(p["cornering_stiffness_rear_nprad"])
        return mass / wheelbase * (rear / cf - front / cr)

    def _road_feedforward(self, curvature: float, speed: float) -> float:
        """Steering angle that holds a curve at steady state, understeer included.

        ``atan(L * kappa)`` is the kinematic Ackermann angle.  At highway speed
        the tyre slip adds ``K_us * v² * kappa``; leaving it out makes the
        feedforward too small and the "added acceleration" budget is spent
        compensating understeer instead of tracking the lane.
        """
        v = max(float(speed), 1.0)
        effective = float(self.vehicle["wheelbase_m"]) + self.understeer_gradient * v * v
        return float(np.arctan(float(curvature) * effective))

    def _angle_for_total_ay(self, ay_mps2: float, speed: float) -> float:
        """Steering angle that produces ``ay_mps2`` of *actual* lateral accel."""
        v = max(float(speed), 1.0)
        effective = float(self.vehicle["wheelbase_m"]) + self.understeer_gradient * v * v
        return float(np.arctan(float(ay_mps2) * effective / (v * v)))

    def _continuous_model(self, speed: float) -> tuple[np.ndarray, np.ndarray]:
        v = max(float(speed), 1.0)
        p = self.vehicle
        m, iz = p["mass_kg"], p["yaw_inertia_kgm2"]
        cf, cr = p["cornering_stiffness_front_nprad"], p["cornering_stiffness_rear_nprad"]
        lf, lr = p["wheelbase_m"] - p["rear_axle_to_cg_m"], p["rear_axle_to_cg_m"]
        a = np.array([
            [0.0, v, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -(cf + cr) / (m * v), -v - (lf * cf - lr * cr) / (m * v)],
            [0.0, 0.0, -(lf * cf - lr * cr) / (iz * v), -(lf**2 * cf + lr**2 * cr) / (iz * v)],
        ])
        b = np.array([0.0, 0.0, cf / m, lf * cf / iz])
        return np.eye(4) + self.dt * a, self.dt * b

    def _error_affine(self, speed: float, curvature: float) -> np.ndarray:
        v = max(float(speed), 1.0)
        p = self.vehicle
        m, iz = p["mass_kg"], p["yaw_inertia_kgm2"]
        cf, cr = p["cornering_stiffness_front_nprad"], p["cornering_stiffness_rear_nprad"]
        lf, lr = p["wheelbase_m"] - p["rear_axle_to_cg_m"], p["rear_axle_to_cg_m"]
        a12 = -v - (lf * cf - lr * cr) / (m * v)
        a22 = -(lf**2 * cf + lr**2 * cr) / (iz * v)
        return self.dt * np.array([0.0, 0.0, a12 * v * curvature, a22 * v * curvature])

    def _build_problem(self) -> None:
        self.x = cp.Variable((4, self.n + 1))
        self.u = cp.Variable(self.n)
        self.a_p = cp.Parameter((4, 4))
        self.b_p = cp.Parameter(4)
        self.x0_p = cp.Parameter(4)
        self.affine_p = cp.Parameter(4)
        self.feedforward_p = cp.Parameter()
        self.uprev_p = cp.Parameter()
        self.u_min_p = cp.Parameter()
        self.u_max_p = cp.Parameter()
        q = np.diag(np.asarray(self.cfg.get("q", [120, 80, 2, 4]), dtype=float))
        r, rd = float(self.cfg.get("r", 1.0)), float(self.cfg.get("rd", 12.0))
        constraints = [self.x[:, 0] == self.x0_p]
        objective = 0
        max_step = self.vehicle["max_steering_wheel_rate_radps"] / self.vehicle["steering_ratio"] * self.dt
        for k in range(self.n):
            constraints += [self.x[:, k + 1] == self.a_p @ self.x[:, k] + self.b_p * self.u[k] + self.affine_p]
            constraints += [self.u[k] >= self.u_min_p, self.u[k] <= self.u_max_p]
            previous = self.uprev_p if k == 0 else self.u[k - 1]
            constraints += [cp.abs(self.u[k] - previous) <= max_step]
            objective += cp.quad_form(self.x[:, k], q) + r * cp.square(self.u[k] - self.feedforward_p)
            objective += rd * cp.square(self.u[k] - previous)
        objective += cp.quad_form(self.x[:, self.n], q)
        self.problem = cp.Problem(cp.Minimize(objective), constraints)

    def _build_native_problem(self, speed: float) -> None:
        """Preallocate the sparse QP; only vectors change at 100 Hz."""
        nx, n = 4, self.n
        nz = nx * (n + 1) + n
        q_diag = np.asarray(self.cfg.get("q", [120, 80, 2, 4]), dtype=float)
        r, rd = float(self.cfg.get("r", 1.0)), float(self.cfg.get("rd", 12.0))
        p_x = sp.diags(np.tile(2.0 * q_diag, n + 1), format="csc")
        diff = sp.diags([np.ones(n), -np.ones(n - 1)], [0, -1], shape=(n, n), format="csc")
        p_u = 2.0 * (r * sp.eye(n, format="csc") + rd * (diff.T @ diff))
        p = sp.block_diag((p_x, p_u), format="csc")
        a_d, b_d = self._continuous_model(speed)
        rows = nx + nx * n + n + n
        acon = sp.lil_matrix((rows, nz))
        row = 0
        acon[row:row + nx, 0:nx] = sp.eye(nx)
        row += nx
        u_start = nx * (n + 1)
        for k in range(n):
            xk, xnext = nx * k, nx * (k + 1)
            acon[row:row + nx, xk:xk + nx] = -a_d
            acon[row:row + nx, xnext:xnext + nx] = sp.eye(nx)
            acon[row:row + nx, u_start + k] = -b_d.reshape(-1, 1)
            row += nx
        acon[row:row + n, u_start:u_start + n] = sp.eye(n)
        row += n
        acon[row:row + n, u_start:u_start + n] = diff
        self.native_q = np.zeros(nz)
        self.native_l = np.zeros(rows)
        self.native_u = np.zeros(rows)
        self.native_u_start = u_start
        self.native_solver = osqp.OSQP()
        self.native_solver.setup(
            P=p, q=self.native_q, A=acon.tocsc(), l=self.native_l, u=self.native_u,
            verbose=False, warm_starting=True, polishing=False, eps_abs=1e-3,
            eps_rel=1e-3, max_iter=500, adaptive_rho=True,
        )

    @staticmethod
    def _reference(trajectory: Trajectory) -> tuple[float, float, float]:
        idx = min(7, len(trajectory.points) - 1)
        point = trajectory.points[idx]
        lateral_heading = np.arctan2(point.lateral_velocity_mps, point.longitudinal_velocity_mps)
        return point.road_d_m, float(lateral_heading), point.curvature_inv_m

    def command(
        self,
        trajectory: Trajectory | None,
        road_d_m: float,
        road_heading_error_rad: float,
        lateral_velocity_mps: float,
        yaw_rate_rps: float,
        speed_mps: float,
        active_function: Function = Function.HWA,
        road_curvature_inv_m: float = 0.0,
    ) -> tuple[float, dict]:
        started = time.perf_counter()
        if trajectory is None or not trajectory.header.valid:
            return 0.0, {"solve_time_ms": 0.0, "solver": "inactive", "status": "no_trajectory"}
        d_ref, yaw_ref, kappa_ref = self._reference(trajectory)
        state = np.array([
            road_d_m - d_ref,
            road_heading_error_rad - yaw_ref,
            lateral_velocity_mps,
            yaw_rate_rps - speed_mps * kappa_ref,
        ])
        lower, upper = self._steering_bounds(active_function, road_curvature_inv_m, speed_mps)
        raw_feedback = self._feedback(state, kappa_ref, speed_mps)
        feedback = float(np.clip(raw_feedback, lower, upper))
        constraint_active = abs(feedback - raw_feedback) > 1e-9
        status = "feedback"
        if self.backend == "osqp_native":
            delta, status = self._native_command(state, kappa_ref, speed_mps, lower, upper)
            per_mode = self.cfg.get("mpc_safety_envelope_by_mode_rad", {})
            envelope = float(per_mode.get(active_function.name, self.cfg.get("mpc_safety_envelope_rad", 0.0005)))
            if status not in ("solved", "solved inaccurate"):
                delta = feedback
            delta = float(np.clip(delta, feedback - envelope, feedback + envelope))
        elif self.backend == "cvxpy":
            a, b = self._continuous_model(speed_mps)
            self.a_p.value, self.b_p.value, self.x0_p.value = a, b, state
            self.affine_p.value = self._error_affine(speed_mps, kappa_ref)
            self.feedforward_p.value = self._road_feedforward(kappa_ref, speed_mps)
            self.uprev_p.value = self.previous_delta
            self.u_min_p.value, self.u_max_p.value = lower, upper
            try:
                self.problem.solve(
                    solver="OSQP", warm_start=True, verbose=False, max_iter=800,
                    eps_abs=1e-3, eps_rel=1e-3, polishing=False,
                )
                status = str(self.problem.status)
                delta = float(self.u.value[0]) if self.u.value is not None else feedback
            except Exception:
                status, delta = "fallback", feedback
            envelope = float(self.cfg.get("mpc_safety_envelope_rad", 0.015))
            if status not in ("optimal", "optimal_inaccurate"):
                delta = feedback
            delta = float(np.clip(delta, feedback - envelope, feedback + envelope))
        else:
            delta = feedback
        constrained_delta = float(np.clip(delta, lower, upper))
        constraint_active = constraint_active or abs(constrained_delta - delta) > 1e-9
        delta = constrained_delta
        delta, supervisor = self._supervise(delta, active_function, road_curvature_inv_m, speed_mps)
        _, _, jerk_limit = DERATED_LIMITS[active_function]
        jerk_rate = jerk_limit * self.vehicle["wheelbase_m"] / max(speed_mps**2, 1.0)
        actuator_rate = 0.8 * self.vehicle["max_steering_wheel_rate_radps"] / self.vehicle["steering_ratio"]
        max_step = min(jerk_rate, actuator_rate) * self.dt
        delta = float(np.clip(delta, self.previous_delta - max_step, self.previous_delta + max_step))
        delta = float(np.clip(delta, lower, upper))
        # Re-apply the acceleration corridor after rate limiting; otherwise an old
        # command can temporarily sit outside the new road-curvature corridor.
        delta, final_supervisor = self._supervise(delta, active_function, road_curvature_inv_m, speed_mps)
        supervisor["supervisor_limited"] = supervisor["supervisor_limited"] or final_supervisor["supervisor_limited"]
        delta = float(np.clip(delta, -0.8 * self.vehicle["max_wheel_angle_rad"], 0.8 * self.vehicle["max_wheel_angle_rad"]))
        self.previous_delta = delta
        elapsed = (time.perf_counter() - started) * 1000.0
        self.solve_times_ms.append(elapsed)
        return delta, {
            "solve_time_ms": elapsed, "solver": self.backend, "status": status,
            "controller_constraint_active": constraint_active,
            **supervisor,
        }

    def _native_command(
        self, state: np.ndarray, kappa_ref: float, speed: float,
        lower: float, upper: float,
    ) -> tuple[float, str]:
        nx, n = 4, self.n
        g = self._error_affine(speed, kappa_ref)
        row = 0
        self.native_l[row:row + nx] = state
        self.native_u[row:row + nx] = state
        row += nx
        for _ in range(n):
            self.native_l[row:row + nx] = g
            self.native_u[row:row + nx] = g
            row += nx
        self.native_l[row:row + n] = lower
        self.native_u[row:row + n] = upper
        row += n
        max_step = 0.8 * self.vehicle["max_steering_wheel_rate_radps"] / self.vehicle["steering_ratio"] * self.dt
        self.native_l[row:row + n] = -max_step
        self.native_u[row:row + n] = max_step
        self.native_l[row] = self.previous_delta - max_step
        self.native_u[row] = self.previous_delta + max_step
        r = float(self.cfg.get("r", 1.0))
        self.native_q[:] = 0.0
        self.native_q[self.native_u_start:] = -2.0 * r * self._road_feedforward(kappa_ref, speed)
        self.native_solver.update(q=self.native_q, l=self.native_l, u=self.native_u)
        result = self.native_solver.solve()
        status = result.info.status.lower()
        delta = float(result.x[self.native_u_start]) if result.x is not None else self._feedback(state, kappa_ref, speed)
        return delta, status

    def _steering_bounds(
        self, active: Function, road_curvature: float, speed: float,
    ) -> tuple[float, float]:
        """Controller-owned bounds; supervisor remains an independent backstop."""
        added_limit, total_limit, _ = DERATED_LIMITS[active]
        guard = float(self.cfg.get("controller_constraint_guard_fraction", 0.995))
        wheelbase = self.vehicle["wheelbase_m"]
        speed_sq = max(speed**2, 1.0)
        road_delta = self._road_feedforward(road_curvature, speed)
        added_delta = float(np.arctan(wheelbase * added_limit * guard / speed_sq))
        total_delta = self._angle_for_total_ay(total_limit * guard, speed)
        physical = 0.8 * self.vehicle["max_wheel_angle_rad"]
        lower = max(road_delta - added_delta, -total_delta, -physical)
        upper = min(road_delta + added_delta, total_delta, physical)
        if lower > upper:
            safe = float(np.clip(road_delta, -total_delta, total_delta))
            return safe, safe
        return float(lower), float(upper)

    def _supervise(
        self, proposed: float, active: Function, road_curvature: float, speed: float
    ) -> tuple[float, dict]:
        added_limit, total_limit, _ = DERATED_LIMITS[active]
        wheelbase = self.vehicle["wheelbase_m"]
        road_delta = self._road_feedforward(road_curvature, speed)
        added_delta = float(np.arctan(wheelbase * added_limit / max(speed**2, 1.0)))
        total_delta = self._angle_for_total_ay(total_limit, speed)
        limited = float(np.clip(proposed, road_delta - added_delta, road_delta + added_delta))
        limited = float(np.clip(limited, -total_delta, total_delta))
        return limited, {
            "supervisor_limited": abs(limited - proposed) > 1e-9,
            "ay_added_limit_mps2": added_limit,
            "ay_total_limit_mps2": total_limit,
        }

    def _feedback(self, state: np.ndarray, kappa_ref: float, speed: float) -> float:
        feedforward = self._road_feedforward(kappa_ref, speed)
        gain = np.asarray(self.cfg.get("fallback_gain", [0.16, 1.5, 0.04, 0.1]))
        return float(feedforward - gain @ state)

