#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import argparse

from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.visualizer.visualizer import Visualizer


RENDER_MESSAGE = {
    "Quit": "ESC",
    "Switch perspective": "Q or B",
    "Reset Episode": "R",
    "Keyboard Control": "W,A,S,D",
    "Start Visualizer Server": "O",
}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reactive_traffic", action="store_true")
    parser.add_argument("--waymo", action="store_true")
    parser.add_argument("--add_sensor", action="store_true")
    parser.add_argument('--host', type=str, default='localhost', help='Server IP')
    parser.add_argument('--port', type=int, default=56789, help='Server port')
    args = parser.parse_args()
    extra_args = dict(film_size=(2000, 2000)) if args.top_down else {}
    asset_path = AssetLoader.asset_path
    use_waymo = args.waymo
    print(HELP_MESSAGE)

    cfg = {
        "manual_control": True,
        "sequential_seed": True,
        "reactive_traffic": True if args.reactive_traffic else False,
    }
    if args.add_sensor:
        additional_cfg = {
            "interface_panel": ["rgb_camera", "depth_camera", "semantic"],
            "sensors": {
                "rgb_camera": (RGBCamera, 256, 256),
                "depth_camera": (DepthCamera, 256, 256),
                "semantic": (SemanticCamera, 256, 256)
            },
            'image_observation': True,
        }
        cfg.update(additional_cfg)

    try:
        env = ScenarioEnv(cfg)
        obs, _ = env.reset()

        # Start visualizer server when 'o' key is pressed
        viser = Visualizer(0, 0, mode='server', host=args.host, port=args.port)

        for i in range(1, 100000):
            o, r, tm, tc, info = env.step([1.0, 0.])
            
            if viser.is_running():
                viser.run(o)
            
            if tm or tc:
                env.reset()
    finally:
        env.close()
        viser.shutdown()