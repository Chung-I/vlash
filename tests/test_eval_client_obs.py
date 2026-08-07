"""Observation conversion: LIBERO raw obs -> policy inputs.

Conventions (openpi LIBERO): both camera images are rendered upside-down and
must be rotated 180 degrees; state is eef_pos(3) + axis-angle(3) + gripper(2).
"""
import numpy as np

from benchmarks.libero.eval_client import libero_obs_to_arrays, quat2axisangle


def test_images_rotated_180_and_contiguous():
    obs = {
        "agentview_image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "robot0_eye_in_hand_image": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }
    arrays = libero_obs_to_arrays(obs)
    np.testing.assert_array_equal(
        arrays["image"], obs["agentview_image"][::-1, ::-1]
    )
    assert arrays["image"].flags["C_CONTIGUOUS"]
    assert arrays["state"].shape == (8,)
    assert arrays["state"].dtype == np.float32


def test_quat2axisangle_identity():
    np.testing.assert_allclose(
        quat2axisangle(np.array([0.0, 0.0, 0.0, 1.0])), np.zeros(3), atol=1e-7
    )


def test_quat2axisangle_90deg_z():
    s = np.sin(np.pi / 4)
    aa = quat2axisangle(np.array([0.0, 0.0, s, np.cos(np.pi / 4)]))
    np.testing.assert_allclose(aa, [0.0, 0.0, np.pi / 2], atol=1e-6)
