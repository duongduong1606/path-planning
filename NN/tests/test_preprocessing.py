import numpy as np

from pipeline_config import PipelineConfig
from preprocessing import (
    build_ego_truth, complete_lane_series, interpolate_continuous, previous_hold,
)


def test_interpolation_does_not_extrapolate():
    values, covered = interpolate_continuous(
        np.array([-1., .5, 2.]), np.array([0., 1.]), np.array([0., 2.])
    )
    assert np.isnan(values[[0, 2]]).all()
    assert values[1] == 1.
    assert covered.tolist() == [False, True, False]


def test_previous_hold_is_causal():
    values, covered = previous_hold(
        np.array([0.5, 1.5]), np.array([0., 1., 2.]), np.array([3, 4, 5])
    )
    assert values.tolist() == [3, 4]
    assert covered.all()


def test_ego_truth_terminal_derivatives():
    config = PipelineConfig()
    t = np.arange(120)/20
    y_left = .2*t**2
    positions = np.column_stack((10*t, -y_left, np.zeros_like(t)))
    orientations = np.tile([1., 0., 0., 0.], (len(t), 1))
    target, valid, future = build_ego_truth(t, positions, orientations, config)
    assert valid[0]
    np.testing.assert_allclose(target[0], [1.25, 1.0, .4], atol=2e-3)
    np.testing.assert_allclose(future[0, -1], 1.25, atol=1e-6)


def test_lane_completion_is_causal_and_bounded():
    config = PipelineConfig()
    left = np.full((13, 4), np.nan); right = np.full((13, 4), np.nan)
    left[0] = [1.85, 0, 0, 0]
    lanes = complete_lane_series(left, right, np.ones(13), np.ones(13), config)
    assert lanes.validity[0] == .5
    assert lanes.right[0, 0] == np.float32(-1.85)
    assert lanes.validity[config.lane_hold_steps] > 0
    assert lanes.validity[config.lane_hold_steps + 1] == 0

