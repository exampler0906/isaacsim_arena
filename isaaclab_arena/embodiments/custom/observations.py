# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def gripper_open_normalized(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["joint7", "joint8"], preserve_order=True
    ),
    open_command_expr: dict[str, float] | None = None,
) -> torch.Tensor:
    """Scalar in [0, 1]: 0 = fully closed, 1 = fully open (matches `open_command_expr` / close at 0)."""
    asset: Articulation = env.scene[asset_cfg.name]
    open_command_expr = open_command_expr or {"joint7": 0.035, "joint8": -0.035}
    _, joint_names = asset.find_joints(asset_cfg.joint_names, preserve_order=True)
    # FIXME:硬编码
    jp = asset.data.joint_pos[:, asset_cfg.joint_ids][...,-2:]
    index_list, _, value_list = string_utils.resolve_matching_names_values(open_command_expr, joint_names)
    open_t = torch.zeros(len(joint_names), device=asset.device, dtype=jp.dtype)
    open_t[index_list] = torch.tensor(value_list, device=asset.device, dtype=jp.dtype)
    open_t = open_t.unsqueeze(0).expand_as(jp)
    safe = torch.where(
        open_t.abs() > 1e-9,
        jp / open_t,
        torch.zeros_like(jp),
    )
    g = torch.clamp(safe.mean(dim=-1, keepdim=True), 0.0, 1.0)
    return g


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)
