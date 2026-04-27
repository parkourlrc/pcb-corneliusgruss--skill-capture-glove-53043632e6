import torch

from glove.policy.train import _actions_to_relative_pose_mats
from glove.policy.transforms import matrix_to_6d, rot6d_to_matrix
from glove.policy.transforms import NormStats


def test_matrix_to_6d_identity_roundtrip():
    R = torch.eye(3)

    d6 = matrix_to_6d(R)
    R_rec = rot6d_to_matrix(d6)

    torch.testing.assert_close(d6, torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(R_rec, R, atol=1e-6, rtol=1e-6)


def test_matrix_to_6d_batched_roundtrip():
    R = torch.tensor([
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    ])

    R_rec = rot6d_to_matrix(matrix_to_6d(R))

    torch.testing.assert_close(R_rec, R, atol=1e-6, rtol=1e-6)


def test_delta_actions_chain_to_relative_poses():
    stats = NormStats(
        hall_mean=torch.zeros(12).numpy(),
        hall_std=torch.ones(12).numpy(),
        tof_mean=torch.zeros(2).numpy(),
        tof_std=torch.ones(2).numpy(),
        pose_pos_scale=1.0,
    )
    actions = torch.zeros(1, 3, 21)
    identity6 = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    actions[..., 3:9] = identity6
    actions[0, :, 0] = torch.tensor([0.0, 0.1, 0.2])

    T_rel = _actions_to_relative_pose_mats(actions, stats, "delta")

    torch.testing.assert_close(
        T_rel[0, :, :3, 3],
        torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.3, 0.0, 0.0]]),
    )
