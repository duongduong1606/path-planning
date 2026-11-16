from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class DisturbanceSample:
    crosswind_mps: float = 0.0
    aerodynamic_force_n: float = 0.0
    aerodynamic_yaw_moment_nm: float = 0.0
    bank_angle_rad: float = 0.0
    friction_coefficient: float = 0.9


def _smooth_envelope(t: float, start: float, duration: float, ramp_fraction: float) -> float:
    """C1 half-cosine ramp, plateau, and ramp-down envelope."""
    local = t - start
    if local < 0.0 or local > duration:
        return 0.0
    ramp = max(1e-6, min(0.5, ramp_fraction) * duration)
    if local < ramp:
        return 0.5 - 0.5 * math.cos(math.pi * local / ramp)
    if local > duration - ramp:
        return 0.5 - 0.5 * math.cos(math.pi * (duration - local) / ramp)
    return 1.0


class DisturbanceModel:
    """Deterministic environment loads, kept outside all planner inputs."""

    def __init__(self, profiles: list[dict], vehicle_cfg: dict):
        self.profiles = profiles
        self.vehicle = vehicle_cfg
        self.rho = float(vehicle_cfg.get("air_density_kgpm3", 1.225))
        self.area = float(vehicle_cfg.get("side_area_m2", 4.2))
        self.cy = float(vehicle_cfg.get("crosswind_coefficient", 0.8))
        self.cp_x = float(vehicle_cfg.get("aero_center_to_cg_m", 0.35))
        self.base_mu = float(vehicle_cfg.get("road_friction_coefficient", 0.9))

    def sample(self, t: float, lateral_velocity_mps: float = 0.0) -> DisturbanceSample:
        wind = 0.0
        bank = 0.0
        mu = self.base_mu
        cp_weighted = 0.0
        force_weight = 0.0
        for profile in self.profiles:
            env = _smooth_envelope(
                t, float(profile["start_s"]), float(profile["duration_s"]),
                float(profile.get("ramp_fraction", 0.25)),
            )
            kind = profile["type"]
            direction = float(profile.get("direction", 1.0))
            if kind == "crosswind":
                local_t = max(0.0, t - float(profile["start_s"]))
                intensity = float(profile.get("turbulence_intensity", 0.0))
                f1 = float(profile.get("turbulence_frequency_1_hz", 1.1))
                f2 = float(profile.get("turbulence_frequency_2_hz", 2.7))
                fluctuation = 1.0 + intensity * (
                    0.6 * math.sin(2.0 * math.pi * f1 * local_t)
                    + 0.4 * math.sin(2.0 * math.pi * f2 * local_t + 0.7)
                )
                component = direction * float(profile["peak_wind_mps"]) * env * fluctuation
                wind += component
                cp = float(profile.get("center_of_pressure_x_m", self.cp_x))
                cp_weighted += abs(component) * cp
                force_weight += abs(component)
            elif kind == "road_bank":
                bank += direction * math.radians(float(profile["peak_angle_deg"])) * env
            elif kind == "friction_patch":
                target = float(profile["minimum_mu"])
                mu = min(mu, self.base_mu + (target - self.base_mu) * env)
        relative_wind = wind - lateral_velocity_mps
        force = 0.5 * self.rho * self.area * self.cy * relative_wind * abs(relative_wind)
        effective_cp = cp_weighted / force_weight if force_weight > 1e-9 else self.cp_x
        return DisturbanceSample(
            crosswind_mps=wind,
            aerodynamic_force_n=force,
            aerodynamic_yaw_moment_nm=force * effective_cp,
            bank_angle_rad=bank,
            friction_coefficient=max(0.1, mu),
        )

    def start_events(self) -> list[tuple[float, str]]:
        return [(float(p["start_s"]), p["type"]) for p in self.profiles]

