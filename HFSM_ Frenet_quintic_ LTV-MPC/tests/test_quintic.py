import numpy as np

from l2_path_planner.planner import Quintic


def test_quintic_satisfies_all_six_boundary_conditions():
    q = Quintic.from_boundary(0.3, -0.2, 0.1, -0.4, 0.0, 0.0, 2.5)
    d, v, a, _ = q.evaluate(np.array([0.0, 2.5]))
    np.testing.assert_allclose([d[0], v[0], a[0]], [0.3, -0.2, 0.1], atol=1e-10)
    np.testing.assert_allclose([d[1], v[1], a[1]], [-0.4, 0.0, 0.0], atol=1e-9)

