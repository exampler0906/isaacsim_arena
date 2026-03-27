# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from isaaclab_arena.utils.pose import Pose
import sys
from pathlib import Path
import os
for _env_key in ("ISAACSIM_ARENA_COMMON_ROOT", "LEROBOT_CODE_ROOT"):
    _root = os.environ.get(_env_key, "").strip()
    if _root:
        _p = Path(_root).expanduser().resolve()
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
from isaacsim_arena_common.object_position import *

def set_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose: Pose,
) -> None:
    if env_ids is None:
        return
    # Grab the object
    asset = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    # Convert the pose to the env frame (Isaac sim buffers are float32)
    pose_t_xyz_q_wxyz = pose.to_tensor(device=env.device).repeat(num_envs, 1).float()
    origins = env.scene.env_origins[env_ids].float()
    pose_t_xyz_q_wxyz[:, :3] = pose_t_xyz_q_wxyz[:, :3] + origins
    # Set the pose and velocity
    asset.write_root_pose_to_sim(pose_t_xyz_q_wxyz, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=env_ids)


def set_random_object_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (0.6, 0.9),
    y_range: tuple[float, float] = (0.1, 0.4),
    z: float = 0.15,
) -> None:
    if env_ids is None:
        return
    # Grab the object
    asset = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    x = round(float(np.random.uniform(x_range[0], x_range[1])), 3)
    y = round(float(np.random.uniform(y_range[0], y_range[1])), 3)
    pose = Pose(
        position_xyz=(x, y, z),
        rotation_wxyz=object_rotation_dict[asset_cfg.name],
    )
    # 所有env的随机位置一致
    print("randomize_object_position: ", x, y, z)
    pose_t_xyz_q_wxyz = pose.to_tensor(device=env.device).repeat(num_envs, 1).float()
    origins = env.scene.env_origins[env_ids].float()
    pose_t_xyz_q_wxyz[:, :3] = pose_t_xyz_q_wxyz[:, :3] + origins
    asset.write_root_pose_to_sim(pose_t_xyz_q_wxyz, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=env_ids)

def set_object_pose_per_env(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_list: list[Pose],
) -> None:
    if env_ids is None:
        return

    # Grab the object
    asset = env.scene[asset_cfg.name]

    # Set the objects pose in each environment independently
    assert env_ids.ndim == 1
    for cur_env in env_ids.tolist():
        # Convert the pose to the env frame
        pose = pose_list[cur_env]
        pose_t_xyz_q_wxyz = pose.to_tensor(device=env.device).float()
        origin = env.scene.env_origins[cur_env, :].squeeze().float()
        pose_t_xyz_q_wxyz[:3] = pose_t_xyz_q_wxyz[:3] + origin
        # Set the pose and velocity
        asset.write_root_pose_to_sim(pose_t_xyz_q_wxyz, env_ids=torch.tensor([cur_env], device=env.device))
        asset.write_root_velocity_to_sim(
            torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
        )
