# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Sequence
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.utils.math as PoseUtils
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.assets.asset_base_cfg import AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab import sim as sim_utils
from dataclasses import MISSING
from isaaclab_tasks.manager_based.manipulation.stack.mdp.observations import ee_frame_pos, ee_frame_quat

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.mimic_utils import get_rigid_and_articulated_object_poses
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.custom.gripper_normalized_action import GripperNormalizedActionCfg
from isaaclab_arena.embodiments.custom.observations import gripper_open_normalized, gripper_pos
from isaaclab_arena.utils.pose import Pose

# 仅手臂 6 关节：与「全关节 joint_pos_rel 再取 [..., :-2]」等价，且由 joint_names 显式约束顺序
_ARM_JOINT_ASSET_CFG = SceneEntityCfg(
    "robot",
    joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    preserve_order=True,
)


def _joint_pos_round_2dp(t: torch.Tensor) -> torch.Tensor:
    """关节角保留两位小数（与物体位置 round(x, 2) 一致）。"""
    return (t * 100.0).round() / 100.0


def set_default_joint_pose(
    env,
    env_ids,
    default_pose: list[float],
    asset_cfg=None,
):
    """Set robot to default joint pose and write to sim (for 8-DOF custom arm: 6 arm + 2 gripper)."""
    if asset_cfg is None:
        robot = env.scene["robot"]
    else:
        robot = env.scene[asset_cfg.name]
    device = robot.device
    default_pose = torch.tensor(default_pose, device=device, dtype=torch.float32)
    num_joints = robot.num_joints
    if default_pose.numel() != num_joints:
        raise RuntimeError(
            f"default_pose length {default_pose.numel()} != robot joint count {num_joints}"
        )
    default_pose = default_pose.unsqueeze(0).repeat(len(env_ids), 1)
    default_pose = _joint_pos_round_2dp(default_pose)
    zero_vel = torch.zeros_like(default_pose)
    robot.write_joint_state_to_sim(position=default_pose, velocity=zero_vel, env_ids=env_ids)
    robot.set_joint_position_target(default_pose, env_ids=env_ids)


def randomize_joint_by_gaussian_offset(
    env,
    env_ids,
    mean: float,
    std: float,
    asset_cfg=None,
):
    """Add gaussian noise to arm joints only; gripper (last 2) unchanged. For 8-DOF custom arm."""
    if asset_cfg is None:
        robot = env.scene["robot"]
    else:
        robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[env_ids].clone()
    noise = torch.randn_like(joint_pos) * std + mean
    joint_pos = joint_pos + noise
    joint_pos[:, -2:] = robot.data.joint_pos[env_ids, -2:]  # keep gripper unchanged
    lower = robot.data.joint_limits[env_ids, :, 0]
    upper = robot.data.joint_limits[env_ids, :, 1]
    new_joint_pos = torch.clamp(joint_pos, lower, upper)
    new_joint_pos = _joint_pos_round_2dp(new_joint_pos)
    new_joint_pos = torch.clamp(new_joint_pos, lower, upper)
    zero_vel = torch.zeros_like(new_joint_pos)
    robot.write_joint_state_to_sim(position=new_joint_pos, velocity=zero_vel, env_ids=env_ids)
    robot.set_joint_position_target(new_joint_pos, env_ids=env_ids)


@register_asset
class CustomEmbodiment(EmbodimentBase):
    """Embodiment for the Custom robot."""

    name = "custom"

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config = CustomSceneCfg()
        self.action_config = CustomJointPositionGripperScalarActionsCfg()
        self.observation_config = CustomObservationsCfg()
        self.event_config = CustomEventCfg()
        self.mimic_env = CustomMimicEnv
        self.camera_config = CustomCameraCfg()

    def _update_scene_cfg_with_robot_initial_pose(self, scene_config: Any, pose: Pose) -> Any:
        # We override the default initial pose setting function in order to also set
        # the initial pose of the stand.
        scene_config = super()._update_scene_cfg_with_robot_initial_pose(scene_config, pose)
        if scene_config is None or not hasattr(scene_config, "robot"):
            raise RuntimeError("scene_config must be populated with a `robot` before calling `set_robot_initial_pose`.")
        scene_config.stand.init_state.pos = pose.position_xyz
        scene_config.stand.init_state.rot = pose.rotation_wxyz
        return scene_config


@register_asset
class BigCustomEmbodiment(CustomEmbodiment):
    """Embodiment for the Big Custom robot."""

    name = "big_custom"

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config = BigCustomSceneCfg()


# 腕部相机相对末端（与 Franka Libero 风格一致）
# 欧拉 X=0°, Y=-55°, Z=-90° -> wxyz 四元数
_WRIST_CAM_OFFSET = Pose(position_xyz=(0.0, 0.0, 0.03), rotation_wxyz=(0.6272, 0.3256, -0.3256, -0.6272))


@configclass
class CustomCameraCfg():
    """机械臂腕部相机（场景固定相机由 example environment 单独提供并合并）。"""

    wrist_cam: CameraCfg | TiledCameraCfg = MISSING

    def __post_init__(self):
        is_tiled_camera = getattr(self, "_is_tiled_camera", True)
        wrist_cam_offset = getattr(self, "_wrist_camera_offset", _WRIST_CAM_OFFSET)
        CameraClass = TiledCameraCfg if is_tiled_camera else CameraCfg
        OffsetClass = CameraClass.OffsetCfg
        wrist_cam_common_kwargs = dict(
            prim_path="{ENV_REGEX_NS}/Robot/piper_description/camera_link/WristCam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.01, 1.0e5)),
        )
        wrist_cam_offset_real = OffsetClass(
            pos=wrist_cam_offset.position_xyz,
            rot=wrist_cam_offset.rotation_wxyz,
            convention="opengl",
        )
        self.wrist_cam = CameraClass(offset=wrist_cam_offset_real, **wrist_cam_common_kwargs)


@configclass
class CustomSceneCfg:
    """Additions to the scene configuration coming from the Custom embodiment."""

    # The robot (link1-6: revolute arm, link7-8: gripper; aligned with Franka pattern)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],         # 世界坐标系位置
            rot=[1.0, 0.0, 0.0, 0.0],    # 四元数 wxyz
            joint_pos={".*": 0.0}               # 可选初始关节状态
        ),
        spawn=UsdFileCfg(
            usd_path="/home/weipeng/lerobot_code/usd/piper_description1/test.usd",
            scale=(1.0, 1.0, 1.0),       # 缩放
            activate_contact_sensors=False,
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
                stiffness=15000,
                # 略增阻尼：水平移动时减小腕部链对夹爪的激励
                damping=180,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["joint7", "joint8"],
                # 高刚度 + 原 damping=200 易在水平加速下欠阻尼“甩动”；提高 d 抑制振荡
                stiffness=38000,
                damping=1200,
                # 略增关节等效惯量，利于 PhysX 隐式求解稳定、抑制高频抖
                armature=0.02,
            ),
        }
    )

    # The stand for the custom
    # TODO(alexmillane, 2025-07-28): We probably want to make the stand an optional addition.
    stand: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot_Stand",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-0.05, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(1.2, 1.2, 1.7),
            activate_contact_sensors=False,
        ),
    )

    # 末端与夹爪帧（prim_path 与当前 USD 一致，勿改）
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/piper_description/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/piper_description/gripper_base",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/piper_description/link8",
                name="tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/piper_description/link7",
                name="tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )

    def __post_init__(self):
        # Add a marker to the end-effector frame
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame.visualizer_cfg = marker_cfg


@configclass
class BigCustomSceneCfg(CustomSceneCfg):
    """Additions to the scene configuration coming from the Custom embodiment."""

    # The robot (link1-6: revolute arm, link7-8: gripper; aligned with Franka pattern)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],         # 世界坐标系位置
            rot=[1.0, 0.0, 0.0, 0.0],    # 四元数 wxyz
            joint_pos={".*": 0.0}               # 可选初始关节状态
        ),
        spawn=UsdFileCfg(
            usd_path="/home/weipeng/lerobot_code/usd/piper_description1/test.usd",
            scale=(2.0, 2.0, 2.0),       # 缩放
            activate_contact_sensors=False,
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
                stiffness=15000,
                damping=180,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["joint7", "joint8"],
                stiffness=38000,
                damping=1200,
                armature=0.02,
            ),
        }
    )



# 抑制末端“前后摆动、按 S 画圈”：加大 IK 阻尼、减小每步位移、提高关节阻尼。
@configclass
class CustomActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        body_name="gripper_base",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls"
        ),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    )

    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint7", "joint8"],
        open_command_expr={"joint7": 0.035, "joint8": -0.035},
        close_command_expr={"joint7": 0.0, "joint8": 0.0},
    )


@configclass
class CustomJointPositionActionsCfg:
    """使用关节位置 (joint_pos) 直接控制的动作配置。

    这里将 8 个关节（6 个手臂关节 + 2 个夹爪关节）都作为 JointPositionActionCfg 的控制目标：
    - 前 6 维: joint1~joint6 关节位置
    - 后 2 维: joint7~joint8 关节位置（可由策略直接输出目标开合角度）

    这样一来，策略直接在 joint space 中工作，省去 IK 规划。
    """

    # 整个机械臂（含手臂+夹爪）都用关节位置控制
    joint_position_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    )

    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint7", "joint8"],
        open_command_expr={"joint7": 0.035, "joint8": -0.035},
        close_command_expr={"joint7": 0.0, "joint8": 0.0},
    )


@configclass
class CustomJointPositionGripperScalarActionsCfg:
    """6 维手臂关节位置 + 1 维夹爪开合 [0,1]（0=闭合，1=张开，与 Binary 配置的 open/close 一致）。"""

    joint_position_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    )
    gripper_open_action: ActionTermCfg = GripperNormalizedActionCfg(
        asset_name="robot",
        joint_names=["joint7", "joint8"],
        open_command_expr={"joint7": 0.035, "joint8": -0.035},
        close_command_expr={"joint7": 0.0, "joint8": 0.0},
    )


@configclass
class CustomObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel)
        # 不能对 ObsTerm 做切片；用 asset_cfg 限定关节即得到「去掉最后两维夹爪」的 6 维手臂
        joint_pos_less = ObsTerm(
            func=mdp_isaac_lab.joint_pos_rel,
            params={"asset_cfg": _ARM_JOINT_ASSET_CFG},
        )
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel)
        eef_pos = ObsTerm(func=ee_frame_pos)
        eef_quat = ObsTerm(func=ee_frame_quat)
        gripper_pos = ObsTerm(func=gripper_pos)
        gripper_open = ObsTerm(func=gripper_open_normalized)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()

@configclass
class CustomEventCfg:
    """Configuration for Custom (8-DOF: 6 arm + 2 gripper; uses local events, not franka_stack_events)."""

    init_custom_arm_pose = EventTerm(
        func=set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.035, -0.035],
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    randomize_custom_joint_state = EventTerm(
        func=randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.2,
            "std": 0.2,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# This is copied from FrankaCubeStackIKAbsMimicEnv in isaaclab_mimic.
# We copy it as we only need a few methods from it.
# The remaining ones belong to the task.
class CustomMimicEnv(ManagerBasedRLMimicEnv):
    """Configuration for Custom Mimic."""

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.
        Args:
            eef_name: Name of the end effector.
            env_ids: Environment indices to get the pose for. If None, all envs are considered.
        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        if env_ids is None:
            env_ids = slice(None)

        # Retrieve end effector pose from the observation buffer
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        # Quaternion format is w,x,y,z
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        noise: float | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector controller and returns an action
        (usually a normalized delta pose action) to try and achieve that target pose.
        Noise is added to the target pose action if specified.
        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            noise: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to get the action for.
        Returns:
            An action torch.Tensor that's compatible with env.step().
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        # target position and rotation
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # normalized delta position action
        delta_position = target_pos - curr_pos

        # normalized delta rotation action
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        # get gripper action for single eef
        (gripper_action,) = gripper_action_dict.values()

        # add noise to action
        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if noise is not None:
            noise = noise * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.
        Args:
            action: Environment action. Shape is (num_envs, action_dim)
        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # get pose target
        target_pos = curr_pos + delta_position

        # Convert delta_rotation to axis angle form
        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle

        # Handle invalid division for the case when delta_rotation_angle is close to zero
        is_close_to_zero_angle = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[is_close_to_zero_angle]

        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions (compatible with env.step).
        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).
        Returns:
            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
        """
        # last dimension is gripper action
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    # Implemented this to consider articulated objects as well
    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """
        Gets the pose of each object(rigid and articulated) in the current scene.
        Args:
            env_ids: Environment indices to get the pose for. If None, all envs are considered.
        Returns:
            A dictionary that maps object names to object pose matrix (4x4 torch.Tensor)
        """
        if env_ids is None:
            env_ids = slice(None)

        state = self.scene.get_state(is_relative=True)

        object_pose_matrix = get_rigid_and_articulated_object_poses(state, env_ids)

        return object_pose_matrix
