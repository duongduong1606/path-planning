import numpy as np

from expert_teacher import ExpertConfig, build_constraints, project_human_target


def lane_feature() -> np.ndarray:
    feature = np.zeros(16, np.float32)
    feature[0] = 1.8
    feature[4] = -1.8
    feature[8:10] = 1.0
    feature[10] = 20.0
    feature[15] = 1.0
    return feature


def test_feasible_human_target_is_not_duplicated_as_teacher() -> None:
    target = np.zeros(3, np.float32)
    result, valid, changed, correction, _, reason = project_human_target(target, lane_feature())
    np.testing.assert_array_equal(result, target)
    assert not valid and not changed and correction == 0
    assert reason == "already_feasible"


def test_expert_projects_unsafe_target_into_all_constraints() -> None:
    config = ExpertConfig()
    target = np.array([3.0, 1.5, 1.0], np.float32)
    result, valid, changed, correction, _, reason = project_human_target(
        target, lane_feature(), config
    )
    assert valid and changed and correction > 0 and reason == "projected"
    rows, lower, upper = build_constraints(lane_feature(), config)
    values = rows @ result
    assert np.all(values >= lower-config.tolerance*2)
    assert np.all(values <= upper+config.tolerance*2)


def test_low_confidence_lane_disables_expert() -> None:
    feature = lane_feature()
    feature[8] = 0.2
    _, valid, changed, _, _, reason = project_human_target(
        np.array([3.0, 0.0, 0.0]), feature
    )
    assert not valid and not changed and reason == "lane_confidence"

