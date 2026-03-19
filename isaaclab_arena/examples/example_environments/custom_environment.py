# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase

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

        background = self.asset_registry.get_asset_by_name("packing_table")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        # franka_libero 定义了两个相机（腕部 + 场景），需启用相机才会加入 scene
        enable_cameras = getattr(args_cli, "enable_cameras", False)
        if getattr(args_cli, "embodiment", None) == "franka_libero":
            enable_cameras = True  # 使用 franka_libero 时默认开启双相机
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=enable_cameras)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(0.7, 0.20, 0.15),
                rotation_wxyz=(0.5, 0.5, -0.5, 0.5),
            )
        )
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(-0.25, 0.0, 0.0),
                rotation_wxyz=(1, 0, 0, 0),
            )
        )

        # # NOTE(alexmillane, 2025.09.08): This is a sub-optimal destination location
        # # in the room. I'd like to use the bottom shelf, however, the whole shelf is
        # # a single prim and therefore I cannot pick out the bottom shelf specifically.
        # # NOTE(alexmillane, 2025.09.08): I've also had to apply the rigid body API to
        # # the lid via the UI.
        # # TODO(alexmillane, 2025.09.08): Separate the self into prims so we can reference
        # # the bottom shelf specifically.
        destination_location = ObjectReference(
            name="destination_location",
            prim_path="{ENV_REGEX_NS}/packing_table/container_h20",
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
                    intensity=4000.0,
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

            @configclass
            class LightingEventsCfg:
                disable_other_lights: EventTermCfg = EventTermCfg(func=_disable_other_lights_startup, mode="startup")

        scene = Scene(assets=assets)
        if True:
        #if getattr(args_cli, "lighting", "stage_lights") == "grey_studio":
            scene.events_cfg = LightingEventsCfg()
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=PickAndPlaceTask(pick_up_object, destination_location, background),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="sugar_box")
        parser.add_argument("--embodiment", type=str, default="custom")
        parser.add_argument(
            "--lighting",
            type=str,
            choices=["stage_lights", "grey_studio"],
            default="grey_studio",
            help="Lighting mode for sensor renders. 'stage_lights' uses lights authored in the stage. "
            "'grey_studio' adds a soft neutral dome light and disables other stage lights (minimize shadows).",
        )
        # NOTE(alexmillane, 2025.09.04): We need a teleop device argument in order
        # to be used in the record_demos.py script.
        parser.add_argument("--teleop_device", type=str, default=None)
