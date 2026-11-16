from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .disturbance import DisturbanceSample


@dataclass(slots=True)
class VehicleState:
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0
    lateral_velocity_mps: float = 0.0
    yaw_rate_rps: float = 0.0
    steering_angle_rad: float = 0.0
    #: Body-frame lateral acceleration at the CG: ``a_y = v_dot_y + v_x * r``.
    #: This is what an IMU publishes; it is not the centripetal ``v * r`` term.
    lateral_acceleration_mps2: float = 0.0


class DynamicBicycle:
    def __init__(self, cfg: dict, dt_s: float):
        self.cfg = cfg
        self.dt = dt_s
        self.state = VehicleState()
        delay_steps = max(0, int(round(cfg["actuator_delay_s"] / dt_s)))
        self.delay = deque([0.0] * (delay_steps + 1), maxlen=delay_steps + 1)

    def _lateral_matrices(self, speed: float) -> tuple[np.ndarray, np.ndarray]:
        m = self.cfg["mass_kg"]
        iz = self.cfg["yaw_inertia_kgm2"]
        cf = self.cfg["cornering_stiffness_front_nprad"]
        cr = self.cfg["cornering_stiffness_rear_nprad"]
        lf = self.cfg["wheelbase_m"] - self.cfg["rear_axle_to_cg_m"]
        lr = self.cfg["rear_axle_to_cg_m"]
        v = max(speed, 1.0)
        a = np.array(
            [
                [-(cf + cr) / (m * v), -v - (lf * cf - lr * cr) / (m * v)],
                [-(lf * cf - lr * cr) / (iz * v), -(lf * lf * cf + lr * lr * cr) / (iz * v)],
            ]
        )
        b = np.array([cf / m, lf * cf / iz])
        return a, b

    def step(
        self, steering_command_rad: float, speed_mps: float,
        disturbance: DisturbanceSample | None = None,
    ) -> VehicleState:
        self.delay.append(float(steering_command_rad))
        delayed = self.delay[0]
        max_rate = self.cfg["max_steering_wheel_rate_radps"] / self.cfg["steering_ratio"]
        delta_change = np.clip(delayed - self.state.steering_angle_rad, -max_rate * self.dt, max_rate * self.dt)
        self.state.steering_angle_rad = float(np.clip(
            self.state.steering_angle_rad + delta_change,
            -self.cfg["max_wheel_angle_rad"], self.cfg["max_wheel_angle_rad"]
        ))
        disturbance = disturbance or DisturbanceSample()
        p = self.cfg
        m, iz = p["mass_kg"], p["yaw_inertia_kgm2"]
        lf, lr = p["wheelbase_m"] - p["rear_axle_to_cg_m"], p["rear_axle_to_cg_m"]
        cf, cr = p["cornering_stiffness_front_nprad"], p["cornering_stiffness_rear_nprad"]
        v = max(speed_mps, 1.0)
        vy, yaw_rate = self.state.lateral_velocity_mps, self.state.yaw_rate_rps
        alpha_f = self.state.steering_angle_rad - (vy + lf * yaw_rate) / v
        alpha_r = -(vy - lr * yaw_rate) / v
        fy_front = cf * alpha_f
        fy_rear = cr * alpha_r
        gravity = 9.81
        front_load = m * gravity * lr / (lf + lr)
        rear_load = m * gravity * lf / (lf + lr)
        fy_front = float(np.clip(fy_front, -disturbance.friction_coefficient * front_load, disturbance.friction_coefficient * front_load))
        fy_rear = float(np.clip(fy_rear, -disturbance.friction_coefficient * rear_load, disturbance.friction_coefficient * rear_load))
        bank_acceleration = gravity * np.sin(disturbance.bank_angle_rad)
        vy_dot = (fy_front + fy_rear + disturbance.aerodynamic_force_n) / m - v * yaw_rate + bank_acceleration
        yaw_acceleration = (lf * fy_front - lr * fy_rear + disturbance.aerodynamic_yaw_moment_nm) / iz
        lat = np.array([vy + self.dt * vy_dot, yaw_rate + self.dt * yaw_acceleration])
        self.state.lateral_velocity_mps = float(lat[0])
        self.state.yaw_rate_rps = float(lat[1])
        # Measured lateral acceleration (IMU channel), not the centripetal v*r.
        self.state.lateral_acceleration_mps2 = float(vy_dot + speed_mps * lat[1])
        self.state.x_m += self.dt * (speed_mps * np.cos(self.state.yaw_rad) - lat[0] * np.sin(self.state.yaw_rad))
        self.state.y_m += self.dt * (speed_mps * np.sin(self.state.yaw_rad) + lat[0] * np.cos(self.state.yaw_rad))
        self.state.yaw_rad += self.dt * lat[1]
        return self.state


