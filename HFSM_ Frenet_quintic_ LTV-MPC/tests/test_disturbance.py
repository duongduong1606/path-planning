import math

from l2_path_planner.disturbance import DisturbanceModel


def test_physical_disturbance_profiles_are_smooth_and_finite():
    vehicle = {
        "air_density_kgpm3": 1.225, "side_area_m2": 4.2,
        "crosswind_coefficient": 0.8, "aero_center_to_cg_m": 0.35,
        "road_friction_coefficient": 0.9,
    }
    profiles = [
        {"type": "crosswind", "start_s": 1.0, "duration_s": 2.0, "peak_wind_mps": 20.0, "ramp_fraction": 0.25, "turbulence_intensity": 0.1},
        {"type": "road_bank", "start_s": 1.0, "duration_s": 2.0, "peak_angle_deg": 3.0, "ramp_fraction": 0.25},
        {"type": "friction_patch", "start_s": 1.0, "duration_s": 2.0, "minimum_mu": 0.6, "ramp_fraction": 0.25},
    ]
    model = DisturbanceModel(profiles, vehicle)
    before, onset, middle, after = [model.sample(t) for t in (0.99, 1.0, 2.0, 3.01)]
    assert before.aerodynamic_force_n == onset.aerodynamic_force_n == 0.0
    assert middle.aerodynamic_force_n > 0.0
    assert middle.aerodynamic_yaw_moment_nm > 0.0
    assert math.isclose(middle.bank_angle_rad, math.radians(3.0))
    assert math.isclose(middle.friction_coefficient, 0.6)
    assert after.aerodynamic_force_n == 0.0
    assert after.friction_coefficient == 0.9
