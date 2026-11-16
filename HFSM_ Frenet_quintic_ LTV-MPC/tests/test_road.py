import numpy as np

from l2_path_planner.road import SimpleRoad


def test_frenet_cartesian_round_trip_near_centerline():
    road = SimpleRoad(3.5, 80.0, 160.0, 2.0)
    s = np.array([20.0, 100.0, 180.0])
    d = np.array([0.2, -0.3, 0.1])
    x, y, _, _ = road.frenet_to_global(s, d, np.zeros(3), 20.0)
    recovered = [road.project(float(xi), float(yi))[1] for xi, yi in zip(x, y)]
    np.testing.assert_allclose(recovered, d, atol=5e-3)

