import numpy as np
import json
from pathlib import Path

from openpilot_teacher import (
    MEDMODEL_INTRINSICS, SOURCE_INTRINSICS, frame_to_model_tensor,
    rotation_from_euler, warp_matrix,
)
from prepare_teacher import cache_is_complete


def test_zero_calibration_warp_maps_model_principal_point_to_source() -> None:
    matrix = warp_matrix(SOURCE_INTRINSICS, MEDMODEL_INTRINSICS, np.zeros(3))
    point = matrix @ np.array([256.0, 47.6, 1.0])
    point /= point[2]
    np.testing.assert_allclose(point[:2], [582.0, 437.0], atol=1e-5)


def test_rotation_is_identity_at_zero() -> None:
    np.testing.assert_allclose(rotation_from_euler(0.0, 0.0, 0.0), np.eye(3))


def test_i420_pack_contract() -> None:
    image = np.zeros((256, 512, 3), np.uint8)
    image[:, :, 1] = np.arange(512, dtype=np.uint16)[None, :] % 256
    packed = frame_to_model_tensor(image, np.eye(3, dtype=np.float32))
    assert packed.shape == (6, 128, 256)
    assert packed.dtype == np.uint8
    assert np.isfinite(packed).all()


def test_partial_cache_is_not_complete_for_full_run() -> None:
    path = Path(__file__).with_name("_partial_teacher_cache_test.npz")
    try:
        metadata = {"decoded_frames": 100, "expected_frames": 1200}
        np.savez(path, timestamps=np.arange(100), metadata=np.array(json.dumps(metadata)))
        assert cache_is_complete(path, 80)
        assert not cache_is_complete(path, None)
    finally:
        path.unlink(missing_ok=True)

