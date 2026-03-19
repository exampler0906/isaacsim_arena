from isaaclab.sensors.camera.camera_cfg import PinholeCameraCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm, ObservationGroupCfg as ObsGroup
from .franka import FrankaEmbodiment, FrankaObservationsCfg
from isaaclab.utils import configclass
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.assets.register import register_asset
import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab_tasks.manager_based.manipulation.stack.mdp.observations import ee_frame_pos, ee_frame_quat
from isaaclab_arena.embodiments.franka.observations import gripper_pos
from isaaclab.sensors import CameraCfg, TiledCameraCfg
import isaaclab.sim as sim_utils
from dataclasses import MISSING
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, BinaryJointPositionActionCfg
from isaaclab.managers import ActionTermCfg

# 默认夹爪相机偏移
_WRIST_CAM_OFFSET = Pose(position_xyz=(-0.03, 0.0, 0.05), rotation_wxyz=(0.0, 0.70710678, -0.70710678, 0.0))
# 默认场景相机偏移
_SCENE_CAM_OFFSET = Pose(position_xyz=(2.5, 0, 0.6), rotation_wxyz=(0.5, 0.5, 0.5, 0.5))

@register_asset
class FrankaLiberoEmbodiment(FrankaEmbodiment):
    """Franka embodiment with Libero-style dual cameras: wrist + scene."""

    name = "franka_libero"

    def __init__(self, enable_cameras: bool = True, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.camera_config = FrankaLiberoCameraCfg()
        self.event_config = FrankaLiberoEventCfg()
        #self.action_config = FrankaLiberoActionsCfg()

@configclass
class FrankaLiberoCameraCfg:
    """Configuration for cameras."""

    wrist_cam: CameraCfg | TiledCameraCfg = MISSING
    scene_cam: CameraCfg | TiledCameraCfg = MISSING

    def __post_init__(self):
        # Get configuration from private attributes set by embodiment constructor
        # These use getattr with defaults to avoid scene parser treating them as assets
        is_tiled_camera = getattr(self, "_is_tiled_camera", True)

        wrist_cam_offset = getattr(self, "_camera_offset", _WRIST_CAM_OFFSET)
        scene_cam_offset = getattr(self, "_camera_offset", _SCENE_CAM_OFFSET)

        CameraClass = TiledCameraCfg if is_tiled_camera else CameraCfg
        OffsetClass = CameraClass.OffsetCfg

        wrist_cam_common_kwargs = dict(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/WristCam",
            update_period=0.0,
            height=512,
            width=512,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=5, clipping_range=(0.01, 1.0e5)),
        )

        scene_cam_common_kwargs = dict(
            prim_path="{ENV_REGEX_NS}/SceneCam",
            update_period=0.0,
            height=512,
            width=512,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.01, 1.0e5)),
        )

        wrist_cam_offset_real = OffsetClass(
            pos=wrist_cam_offset.position_xyz,
            rot=wrist_cam_offset.rotation_wxyz,
            convention="opengl",
        )

        scene_cam_offset_real = OffsetClass(
            pos=scene_cam_offset.position_xyz,
            rot=scene_cam_offset.rotation_wxyz,
            convention="opengl",
        )

        self.wrist_cam = CameraClass(offset=wrist_cam_offset_real, **wrist_cam_common_kwargs)
        self.scene_cam = CameraClass(offset=scene_cam_offset_real, **scene_cam_common_kwargs)

@configclass
class FrankaLiberoEventCfg:
    """Configuration for Franek."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0, -0.785, 0.0, -2.356194, 0.0, 1.570796, 0.785, 0.0400, 0.0400],
        },
    )
    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# @configclass
# class FrankaLiberoActionsCfg:
#     """Action specifications for the MDP. 9-DOF joint position control."""
#     arm_action: ActionTermCfg = JointPositionActionCfg(
#         asset_name="robot",
#         joint_names=["panda_joint.*"],
#         scale=0.5,
#         use_default_offset=True,
#     )
#     gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
#         asset_name="robot",
#         joint_names=["panda_finger.*"],
#         open_command_expr={"panda_finger_.*": 0.04},
#         close_command_expr={"panda_finger_.*": 0.0},
#     )