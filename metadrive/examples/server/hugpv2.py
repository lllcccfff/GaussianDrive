#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""

import argparse
import json
from _codecs import encode
from pathlib import Path

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup
from numpy import dtype
from numpy.core.multiarray import scalar
from numpy.dtypes import Float64DType

from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.manager.scenario_data_manager import InvalidMetadataException
from metadrive.viewer.viewer import Viewer


class SimulatorInterface:
    def __init__(self):
        self.current_timestamp = None

        torch.serialization.add_safe_globals(
            [
                (scalar, "numpy.core.multiarray.scalar"),
                (scalar, "numpy._core.multiarray.scalar"),
                dtype,
                Float64DType,
                encode,
            ]
        )

    def load_metadata(self, cfg_path):
        cfg_path = Path(cfg_path)
        if not cfg_path.name == "config.yml":
            raise InvalidMetadataException

        self.config, self.pipeline, _, _ = eval_setup(cfg_path)

        # Decomposite cameras into egos and cameras
        dataset_config_path = None
        if self.config.data is not None:
            dataset_config_path = self.config.data
        elif self.config.pipeline.datamanager.data is not None:
            dataset_config_path = self.config.pipeline.datamanager.data
        assert dataset_config_path is not None
        if dataset_config_path.is_dir():
            dataset_config_path = dataset_config_path / "transforms.json"
        with open(dataset_config_path) as f:
            dataset_config = json.load(f)

        physics_world_step_size = int(2e4)
        ego_poses = {
            i * physics_world_step_size: v
            for i, (k, v) in enumerate(
                sorted(list(dataset_config["sim_data"]["egos_data"].items()), key=lambda x: x[0])
            )
        }
        self.num_frames = len(ego_poses)

        return (
            {},
            self.config.experiment_name,
            [0, len(ego_poses) * physics_world_step_size],
            dataset_config["sim_data"]["cameras_data"],
            ego_poses,
            {},
            None,
            # str(dataset_config_path.parent / "sparse_pc.obj"),
        )

    def load_model(self, cfg):
        return  # Do nothing

    def update_scene(self, timestamp, object_poses):
        self.current_timestamp = timestamp

    def render(self, K, H, W, extrinsics):
        c2w = extrinsics.inverse()
        # OpenCV -> OpenGL (Used by nerfstudio)
        c2w[0:3, 1:3] *= -1

        # Transform to dataset coordinate
        scale = self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
        w2s = self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform

        c2w[:3, :3] = w2s[:3, :3] @ c2w[:3, :3]
        c2w[:3, 3] = (w2s[:3, :3] @ c2w[:3, 3] + w2s[:3, 3]) * scale

        camera = Cameras(
            camera_to_worlds=c2w[None, :3, :4],
            fx=K[0, 0].item(),
            fy=K[1, 1].item(),
            cx=K[0, 2].item(),
            cy=K[1, 2].item(),
            width=W,
            height=H,
            camera_type=CameraType.PERSPECTIVE,
            times=torch.tensor(self.current_timestamp / self.num_frames),  # Normalized to [0, 1]
        )
        with torch.no_grad():
            outputs = self.pipeline.model.get_outputs_for_camera(camera)

        rgb = (outputs["rgb"] * 255).cpu().numpy().astype(np.uint8)
        # print(rgb.mean())
        return rgb  # [H, W, 3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--scene_config_directory", type=str)
    parser.add_argument("--host", type=str, default="localhost", help="Server IP")
    parser.add_argument("--port", type=int, default=56789, help="Server port")
    args = parser.parse_args()
    asset_path = AssetLoader.asset_path

    cfg = {
        "scene_config_directory": args.scene_config_directory,
    }

    model = SimulatorInterface()

    env = ScenarioEnv(model, cfg)
    obs, _ = env.reset()

    # Start visualizer server when 'o' key is pressed
    viser = Viewer(0, 0, mode="server", host=args.host, port=args.port)
    action = [0, 0]

    for i in range(1, 100000):
        o, r, tm, tc, info = env.step(action)

        if tm or tc:
            env.reset()
            action = [0, 0]
            continue

        if viser.is_running():
            o_for_vis = o["gaussian"]["CAM_FRONT"][-1]
            turn_signal = o["navigation"]["turn_signal"]
            # print("[Turn Signal] ", turn_signal)
            action = viser.run(o_for_vis)

    env.close()
    viser.shutdown()
