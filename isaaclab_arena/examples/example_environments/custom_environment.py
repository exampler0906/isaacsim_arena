# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase


import sys
from pathlib import Path

# 指向包含 isaacsim_arena_common 这一目录的父路径，例如 /path/to/lerobot_code
# 用法: export ISAACSIM_ARENA_COMMON_ROOT=/path/to/lerobot_code
# 也可设 LEROBOT_CODE_ROOT（同上含义）
import os
for _env_key in ("ISAACSIM_ARENA_COMMON_ROOT", "LEROBOT_CODE_ROOT"):
    _root = os.environ.get(_env_key, "").strip()
    if _root:
        _p = Path(_root).expanduser().resolve()
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from isaacsim_arena_common.object_position import *


def _resolve_viewer_lookat_offset(args_cli: argparse.Namespace):
    """视口 eye = 物体初始位姿 + offset。CLI --viewer_offset 优先，否则按 background 查 viewer_lookat_offset。"""
    import numpy as np

    bg = getattr(args_cli, "background", None) or "packing_table"
    off = viewer_lookat_offset.get(bg)
    if off is not None:
        return np.array(off, dtype=np.float64)
    return None


def _scene_cam_profile_for_background(background: str) -> str:
    """选用 object_position 中已有 scene_cam_* 条目的 profile（与 --background 名称一致）。"""
    if background in scene_cam_position and background in scene_cam_rotation:
        return background
    return "office_204"


def build_custom_env_scene_camera_cfg(profile: str):
    """固定场景相机（非机械臂本体）：prim 与位姿来自 object_position.scene_cam_*。"""
    from dataclasses import MISSING

    from isaaclab import sim as sim_utils
    from isaaclab.sensors import CameraCfg, TiledCameraCfg
    from isaaclab.utils import configclass

    from isaaclab_arena.utils.pose import Pose

    if profile not in scene_cam_position or profile not in scene_cam_rotation:
        profile = "office_204"

    @configclass
    class _CustomEnvSceneCameraCfg:
        scene_cam: CameraCfg | TiledCameraCfg = MISSING

        def __post_init__(self):
            is_tiled_camera = True
            CameraClass = TiledCameraCfg if is_tiled_camera else CameraCfg
            OffsetClass = CameraClass.OffsetCfg
            scene_cam_offset = Pose(
                position_xyz=scene_cam_position[profile],
                rotation_wxyz=scene_cam_rotation[profile],
            )
            scene_prim = scene_cam_prim_path.get(profile, scene_cam_prim_path["office_204"])
            scene_cam_common_kwargs = dict(
                prim_path=f"{{ENV_REGEX_NS}}/{scene_prim}",
                update_period=0.0,
                height=1024,
                width=1024,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=scene_cam_focal_length[profile], clipping_range=(0.01, 1.0e5)),
            )
            scene_cam_offset_real = OffsetClass(
                pos=scene_cam_offset.position_xyz,
                rot=scene_cam_offset.rotation_wxyz,
                convention="opengl",
            )
            self.scene_cam = CameraClass(offset=scene_cam_offset_real, **scene_cam_common_kwargs)

    return _CustomEnvSceneCameraCfg()


# NOTE(alexmillane, 2025.09.04): There is an issue with type annotation in this file.
# We cannot annotate types which require the simulation app to be started in order to
# import, because this file is used to retrieve CLI arguments, so it must be imported
# before the simulation app is started.
# TODO(alexmillane, 2025.09.04): Fix this.


class CustomEnvironment(ExampleEnvironmentBase):

    name: str = "custom_env"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.assets.asset import Asset
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena.tasks.dummy_task import DummyTask
        from isaaclab.assets.asset_base_cfg import AssetBaseCfg
        from isaaclab import sim as sim_utils

        def _disable_all_lights_except(keep_prim_paths: set[str]):
            """Disable (set intensity=0) all USD Lux lights except those in keep_prim_paths.

            This is useful to approximate Isaac Sim viewport's 'Grey Studio' soft lighting for
            sensor renders, by removing hard key lights that produce strong shadows.
            """
            try:
                import omni.usd
                from pxr import UsdLux
            except Exception:
                return
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return

            for prim in stage.Traverse():
                if not prim.IsValid():
                    continue
                path = str(prim.GetPath())
                if path in keep_prim_paths:
                    continue
                # Any USD Lux light (DomeLight, DistantLight, RectLight, etc.)
                light = UsdLux.LightAPI(prim)
                if not light:
                    continue
                # Set intensity to zero to effectively disable contribution
                try:
                    light.CreateIntensityAttr().Set(0.0)
                except Exception:
                    pass

        class DomeLightAsset(Asset):
            """A minimal Arena Asset wrapper around an IsaacLab DomeLightCfg."""

            def __init__(self, name: str, prim_path: str, dome_light_cfg: sim_utils.DomeLightCfg):
                super().__init__(name=name, tags=["light"])
                self._prim_path = prim_path
                self._dome_light_cfg = dome_light_cfg

            def get_object_cfg(self) -> dict[str, AssetBaseCfg]:
                return {
                    self.name: AssetBaseCfg(
                        prim_path=self._prim_path,
                        spawn=self._dome_light_cfg,
                    )
                }

        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        # franka_libero 定义了两个相机（腕部 + 场景），需启用相机才会加入 scene
        enable_cameras = getattr(args_cli, "enable_cameras", False)
        if getattr(args_cli, "embodiment", None) == "franka_libero":
            enable_cameras = True  # 使用 franka_libero 时默认开启双相机
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=enable_cameras)
        # 场景固定相机属于环境/背景，不属于 embodiment：在腕部相机 cfg 上合并 scene_cam
        # 必须合并场景相机
        if enable_cameras:
            from isaaclab_arena.utils.configclass import combine_configclass_instances

            profile = _scene_cam_profile_for_background(args_cli.background)
            scene_cam_cfg = build_custom_env_scene_camera_cfg(profile)
            embodiment.camera_config = combine_configclass_instances(
                "CameraCfg",
                embodiment.camera_config,
                scene_cam_cfg,
            )

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=object_init_position[background.name][pick_up_object.name],
                rotation_wxyz=object_rotation_dict[pick_up_object.name],
            )
        )
        embodiment.set_initial_pose(
            Pose(
                position_xyz=embodiment_init_position[background.name][args_cli.embodiment],
                rotation_wxyz=embodiment_init_rotation[background.name][args_cli.embodiment],
            )
        )

        # # NOTE(alexmillane, 2025.09.08): This is a sub-optimal destination location
        # # in the room. I'd like to use the bottom shelf, however, the whole shelf is
        # # a single prim and therefore I cannot pick out the bottom shelf specifically.
        # # NOTE(alexmillane, 2025.09.08): I've also had to apply the rigid body API to
        # # the lid via the UI.
        # # TODO(alexmillane, 2025.09.08): Separate the self into prims so we can reference
        # # the bottom shelf specifically.
        # prim_path 必须与 parent_asset.name 一致：Isaac Lab 下为 {ENV_REGEX_NS}/<background.name>/<USD 内相对路径>
        # packing_table 默认子 prim 为 container_h20；其它背景需在 USD 中确认篮子 prim，并用 --destination_prim 指定
        _dest_prim = getattr(args_cli, "destination_prim", None) or "container_h20"
        #print(f"{{ENV_REGEX_NS}}/{background.name}/{_dest_prim}")
        destination_location = ObjectReference(
            name="destination_location",
            prim_path=f"{{ENV_REGEX_NS}}/{background.name}/{_dest_prim}",
            parent_asset=background,
        )

        # Lighting: for sensor renders, Isaac Sim viewport presets (e.g. "Grey Studio") don't apply.
        # To get a similar soft, diffuse look (minimal hard shadows), we:
        # 1) add a neutral dome light, 2) optionally disable other lights in the stage.
        assets = [background, pick_up_object]
        if True:
        #if getattr(args_cli, "lighting", "stage_lights") == "grey_studio":
            grey_studio_light = DomeLightAsset(
                name="light_rig_grey_studio",
                prim_path="/World/LightRigGreyStudio",
                dome_light_cfg=sim_utils.DomeLightCfg(
                    intensity=2000.0,
                    exposure=0.0,
                    color=(0.75, 0.75, 0.75),
                    visible_in_primary_ray=False,
                ),
            )
            assets.append(grey_studio_light)
            # Disable other authored stage lights after the stage is loaded (startup event).
            from isaaclab.managers import EventTermCfg
            from isaaclab.utils import configclass

            
            def _disable_other_lights_startup(env, env_ids=None):
                _disable_all_lights_except({"/World/LightRigGreyStudio"})

            # 与 USD 下物体 prim 名一致（通常等于 --object / 资产 name）
            _object_name_for_gauss = pick_up_object.name

            def _hide_and_show_gauss(env, env_ids=None):
                """刷新抓取物在视口中的显示（克隆/首帧后偶发不画，手动点小眼睛会好）。

                仅用 ``visibility`` 在同一帧内 hide+show 往往不足以触发 Hydra/RTX 重绘；
                这里改为 **post_update 分两帧**：先 ``invisible`` 再 ``inherited``，并对 **刚体根**
                与常见子路径（含 ``.../gauss``）分别尝试，与 Stage 眼睛改的是同一 ``visibility`` 属性。

                若仍无效：检查该 USD 是否用自定义渲染（非 Imageable）、或需在 DCC 里打开 **defaultPrim**、
                **extent**、材质路径等。
                """
                try:
                    import omni.kit.app
                    import omni.usd
                    from pxr import UsdGeom
                except Exception:
                    return

                obj = _object_name_for_gauss
                _post_handle = [None]
                state = {"frame": 0}

                def _candidate_paths_for_env(env_index: int) -> list[str]:
                    """Isaac Lab 刚体常见为 /World/envs/env_i/<name>；子级随 USD 结构追加。"""
                    base = f"/World/envs/env_{env_index}/{obj}"
                    return [
                        #base,
                        f"{base}/{obj}/gauss",
                        #f"{base}/{obj}/{obj}/gauss",
                    ]

                def _set_visibility_on_candidates(vis_token):
                    stage = omni.usd.get_context().get_stage()
                    if stage is None:
                        return
                    n = int(getattr(env, "num_envs", 1))
                    for i in range(n):
                        for path in _candidate_paths_for_env(i):
                            prim = stage.GetPrimAtPath(path)
                            if not prim.IsValid():
                                continue
                            try:
                                UsdGeom.Imageable(prim).GetVisibilityAttr().Set(vis_token)
                            except Exception:
                                pass

                def _on_post_update(_event):
                    state["frame"] += 1
                    if state["frame"] == 1:
                        _set_visibility_on_candidates(UsdGeom.Tokens.invisible)
                        return
                    if state["frame"] == 2:
                        _set_visibility_on_candidates(UsdGeom.Tokens.inherited)
                        if _post_handle[0] is not None:
                            _post_handle[0].unsubscribe()
                            _post_handle[0] = None

                app = omni.kit.app.get_app_interface()
                _post_handle[0] = app.get_post_update_event_stream().create_subscription_to_pop(_on_post_update)

            @configclass
            class EventsCfg:
                disable_other_lights: EventTermCfg = EventTermCfg(func=_disable_other_lights_startup, mode="startup")
                hide_and_show_gauss: EventTermCfg = EventTermCfg(func=_hide_and_show_gauss, mode="reset")

        scene = Scene(assets=assets)
        if True:
        #if getattr(args_cli, "lighting", "stage_lights") == "grey_studio":
            scene.events_cfg = EventsCfg()
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=PickAndPlaceTask(
                pick_up_object,
                destination_location,
                background,
                viewer_lookat_offset=_resolve_viewer_lookat_offset(args_cli),
            ),
            #task=DummyTask(),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--background", type=str, default="packing_table")
        parser.add_argument(
            "--destination_prim",
            type=str,
            default="container_h20",
            help=(
                "篮子/放置区在背景 USD 里相对 defaultPrim 的 prim 路径（单段或多段，如 container_h20 或 Meshes/basket）。"
                "须能在该 USD 中解析；packing_table 场景默认为 container_h20。"
            ),
        )
        parser.add_argument("--object", type=str, default="sugar_box")
        parser.add_argument("--embodiment", type=str, default="custom")
        parser.add_argument(
            "--lighting",
            type=str,
            choices=["stage_lights", "grey_studio"],
            default="stage_lights",
            help="Lighting mode for sensor renders. 'stage_lights' uses lights authored in the stage. "
            "'grey_studio' adds a soft neutral dome light and disables other stage lights (minimize shadows).",
        )
        # NOTE(alexmillane, 2025.09.04): We need a teleop device argument in order
        # to be used in the record_demos.py script.
        parser.add_argument("--teleop_device", type=str, default=None)
