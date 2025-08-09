#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import argparse
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
    parser.add_argument("-c", "--scene_config_directory", type=str)
    parser.add_argument('--host', type=str, default='localhost', help='Server IP')
    parser.add_argument('--port', type=int, default=56789, help='Server port')
    args = parser.parse_args()
    asset_path = AssetLoader.asset_path
    use_waymo = args.waymo
    print(HELP_MESSAGE)

    cfg = {
        "manual_control": True,
        "sequential_seed": True,
        "reactive_traffic": True if args.reactive_traffic else False,
        "scene_config_directory": args.scene_config_directory,
    }
    if args.add_sensor:
        additional_cfg = {
            'image_observation': True,
        }
        cfg.update(additional_cfg)

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
    env.close()
    viser.shutdown()