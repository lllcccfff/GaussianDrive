#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""

import argparse

from easydrive.models.scenes.simulator_interface import SimulatorInterface

from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.viewer.viewer import Viewer

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
    parser.add_argument("--host", type=str, default="localhost", help="Server IP")
    parser.add_argument("--port", type=int, default=56789, help="Server port")
    args = parser.parse_args()
    asset_path = AssetLoader.asset_path
    use_waymo = args.waymo
    print(HELP_MESSAGE)

    cfg = {
        "scene_config_directory": args.scene_config_directory,
    }
    if args.add_sensor:
        additional_cfg = {
            "image_observation": True,
        }
        cfg.update(additional_cfg)

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
            o_for_vis = o["gaussian"]["FRONT"][-1]
            turn_signal = o["navigation"]["turn_signal"]
            print("[Turn Signal] ", turn_signal)
            action = viser.run(o_for_vis)

    env.close()
    viser.shutdown()
