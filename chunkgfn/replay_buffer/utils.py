import torch


def extend_trajectories(
    trajectories_a: torch.Tensor,
    trajectories_b: torch.Tensor,
    actions_a: torch.Tensor,
    actions_b: torch.Tensor,
    dones_a: torch.Tensor,
    dones_b: torch.Tensor,
) -> torch.Tensor:
    """Stitch trajectories, actions and dones together.
    When they don't have the same trajectory length, the one with the minimum length
    should be padded.
    Args:
        trajectories_a (torch.Tensor[batch_size, traj_len, *state_shape]): trajectories.
        trajectories_a (torch.Tensor[batch_size, traj_len, *state_shape]): trajectories.
        actions_a (torch.Tensor[batch_size, traj_len]): actions.
        actions_b (torch.Tensor[batch_size, traj_len]): actions.
        dones_a (torch.Tensor[batch_size, traj_len]): dones.
        dones_b (torch.Tensor[batch_size, traj_len]): dones.
    Return:
        stitched_trajectories (torch.Tensor[batch_size, traj_len, *state_shape]): Stitched trajectories.
        stitched_actions (torch.Tensor[batch_size, traj_len]): Stitched actions.
        stitched_dones (torch.Tensor[batch_size, traj_len]): Stitched dones.
    """

    if trajectories_a.shape[1] == trajectories_b.shape[1]:
        stitched_trajectories = torch.cat([trajectories_a, trajectories_b], dim=0)
        stitched_actions = torch.cat([actions_a, actions_b], dim=0)
        stitched_dones = torch.cat([dones_a, dones_b], dim=0)
    else:
        max_length = max(trajectories_a.shape[1], trajectories_b.shape[1])
        if trajectories_a.shape[1] == max_length:
            trajectories_b = torch.cat(
                [
                    trajectories_b,
                    torch.zeros(
                        trajectories_b.shape[0],
                        max_length - trajectories_b.shape[1],
                        *trajectories_b.shape[2:],
                    ).to(trajectories_b),
                ],
                dim=1,
            )

            actions_b = torch.cat(
                [
                    actions_b,
                    torch.zeros(
                        actions_b.shape[0], max_length - 1 - actions_b.shape[1]
                    ).to(actions_b),
                ],
                dim=1,
            )
            # We pad with ones and not zeros because the trajectory is over.
            dones_b = torch.cat(
                [
                    dones_b,
                    torch.ones(dones_b.shape[0], max_length - dones_b.shape[1]).to(
                        dones_b
                    ),
                ],
                dim=1,
            )
        else:
            trajectories_a = torch.cat(
                [
                    trajectories_a,
                    torch.zeros(
                        trajectories_a.shape[0],
                        max_length - trajectories_a.shape[1],
                        *trajectories_a.shape[2:],
                    ).to(trajectories_a),
                ],
                dim=1,
            )
            actions_a = torch.cat(
                [
                    actions_a,
                    torch.zeros(
                        actions_a.shape[0],
                        max_length - 1 - actions_a.shape[1],
                    ).to(actions_a),
                ],
                dim=1,
            )
            # We pad with ones and not zeros because the trajectory is over.
            dones_a = torch.cat(
                [
                    dones_a,
                    torch.ones(
                        dones_a.shape[0],
                        max_length - dones_a.shape[1],
                    ).to(dones_a),
                ],
                dim=1,
            )

        stitched_trajectories = torch.cat([trajectories_a, trajectories_b], dim=0)
        stitched_actions = torch.cat([actions_a, actions_b], dim=0)
        stitched_dones = torch.cat([dones_a, dones_b], dim=0)

    return stitched_trajectories, stitched_actions, stitched_dones
