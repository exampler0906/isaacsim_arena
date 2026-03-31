# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Single scalar [0, 1] gripper action mapped to two finger joint position targets."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

from isaaclab.envs import ManagerBasedEnv

if TYPE_CHECKING:
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor


class GripperNormalizedAction(ActionTerm):
    """Maps a single normalized openness action in [0, 1] to two gripper joint position targets."""

    cfg: GripperNormalizedActionCfg
    _asset: Articulation
    _joint_ids: list[int] | slice
    _joint_names: list[str]
    _open_command: torch.Tensor
    _close_command: torch.Tensor
    _raw_actions: torch.Tensor
    _processed_actions: torch.Tensor

    def __init__(self, cfg: GripperNormalizedActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=True
        )
        self._num_joints = len(self._joint_ids)
        omni.log.info(
            f"Resolved joint names for GripperNormalizedAction: {self._joint_names} [{self._joint_ids}]"
        )

        self._open_command = torch.zeros(self._num_joints, device=self.device)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.open_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(
                f"open_command_expr: missing joints {set(self._joint_names) - set(name_list)}"
            )
        self._open_command[index_list] = torch.tensor(value_list, device=self.device)

        self._close_command = torch.zeros_like(self._open_command)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.close_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(
                f"close_command_expr: missing joints {set(self._joint_names) - set(name_list)}"
            )
        self._close_command[index_list] = torch.tensor(value_list, device=self.device, dtype=self._close_command.dtype)

        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "JointAction"
        self._IO_descriptor.joint_names = self._joint_names
        return self._IO_descriptor

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        g = torch.clamp(actions, 0.0, 1.0)
        self._processed_actions[:] = (1.0 - g) * self._close_command + g * self._open_command

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


@configclass
class GripperNormalizedActionCfg(ActionTermCfg):
    """Map one openness scalar in [0, 1] to gripper joint positions (closed at 0, open at 1)."""

    class_type: type[ActionTerm] = GripperNormalizedAction

    joint_names: list[str] = MISSING
    open_command_expr: dict[str, float] = MISSING
    close_command_expr: dict[str, float] = MISSING
