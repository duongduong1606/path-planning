import numpy as np

from evaluation import (
    clustered_bootstrap, lane_behavior_metrics, lane_boundary_violation, quintic_trajectory,
    uncertainty_metrics,
)
from safety_shield import project_to_lane_corridor


def test_quintic_meets_boundary_conditions():
    terminal = np.array([1.2, -.3, .4, 0.])
    y, dy, ddy, _ = quintic_trajectory(terminal)
    np.testing.assert_allclose([y[0], dy[0], ddy[0]], 0., atol=1e-10)
    np.testing.assert_allclose([y[-1], dy[-1], ddy[-1]], terminal[:3], atol=1e-10)


def test_uncertainty_coverage():
    prediction = np.zeros((10, 4)); target = np.zeros((10, 3))
    result = uncertainty_metrics(prediction, target)
    assert result["coverage_90"] == 1.0


def test_bootstrap_is_reproducible():
    values = np.arange(6.); routes = np.array(["a", "a", "b", "b", "c", "c"])
    assert clustered_bootstrap(values, routes, repetitions=50) == clustered_bootstrap(values, routes, repetitions=50)


def test_safety_projection_moves_path_inside_lane():
    prediction = np.array([[3.0, 0.0, 0.0, -0.5]], np.float32)
    features = np.zeros((1, 16), np.float32)
    features[:, 0] = 1.8
    features[:, 4] = -1.8
    features[:, 8:10] = 1.0
    speed = np.array([10.0], np.float32)
    assert lane_boundary_violation(prediction, features, speed) == 1.0
    projected, stats = project_to_lane_corridor(prediction, features, speed)
    assert lane_boundary_violation(projected, features, speed) == 0.0
    assert projected[0, 3] == prediction[0, 3]
    assert stats["changed_fraction"] == 1.0


def test_lane_metric_respects_calibrated_forward_range():
    prediction = np.array([[0.0, 0.0, 0.0, 0.0]], np.float32)
    features = np.zeros((1, 16), np.float32)
    # The left cubic is nonsensical when extrapolated to x=0, but valid from
    # the declared calibration start x=5 m onward.
    features[0, :4] = [-10.0, 2.36, 0.0, 0.0]
    features[0, 4] = -1.8
    assert lane_boundary_violation(prediction, features, np.array([1.0])) == 0.0


def test_lane_behavior_separates_demonstrated_crossing():
    features = np.zeros((2, 16), np.float32)
    features[:, 0], features[:, 4] = 1.8, -1.8
    targets = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], np.float32)
    predictions = np.column_stack((targets, np.zeros(2, np.float32)))
    result = lane_behavior_metrics(predictions, targets, features, np.full(2, 10.0))
    assert result["truth_current_lane_crossing_fraction"] == 0.5
    assert result["stay_lane_boundary_violation"] == 0.0
    assert result["truth_crossing_recall"] == 1.0


def test_lane_loss_respects_calibrated_forward_range():
    import torch
    from models import lane_feasibility_loss

    prediction = torch.zeros((1, 4))
    features = torch.zeros((1, 16))
    features[0, :4] = torch.tensor([-10.0, 2.36, 0.0, 0.0])
    features[0, 4] = -1.8
    features[0, 8:10] = 1.0
    features[0, 10] = 1.0
    features[0, 15] = 1.0
    assert lane_feasibility_loss(prediction, features) == 0.0

