#!/usr/bin/env python
"""
This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
"""
import argparse
from metadrive.constants import HELP_MESSAGE
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.misc.nurec_interface.simulator_interface import SimulatorInterface
from metadrive.viewer.viewer import Viewer
import imageio
import os
import time
import numpy as np
import time
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
    parser.add_argument('--grpc-host', type=str, default='localhost', help='gRPC server host')
    parser.add_argument('--grpc-port', type=int, default=50051, help='gRPC server port')
    parser.add_argument('--grpc-timeout', type=float, default=60.0, help='gRPC timeout (seconds)')
    parser.add_argument("-s", "--store", action="store_true", help="Store each episode video to simulation_record")
    args = parser.parse_args()
    asset_path = AssetLoader.asset_path
    use_waymo = args.waymo
    print(HELP_MESSAGE)

    cfg = {
        "scene_config_directory": args.scene_config_directory,
    }
    if args.add_sensor:
        additional_cfg = {
            'image_observation': True,
        }
        cfg.update(additional_cfg)
    
    model = SimulatorInterface(
        grpc_host=args.grpc_host,
        grpc_port=args.grpc_port,
        grpc_timeout_s=args.grpc_timeout,
        resolution_scale=0.5
    )

    env = ScenarioEnv(model, cfg)
    obs, _ = env.reset()
    physics_dt = env.config["physics_world_step_size"] * env.config["decision_repeat"] * 1e-6
    record_frames = []
    record_dir = None
    default_camera_name = "camera_front_tele_30fov"

    def _to_uint8(frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            scale = 255.0 if frame.max() <= 1.0 else 1.0
            frame = (frame * scale).astype(np.uint8)
        return frame

    def extract_frame(observation):
        cameras = observation.get("gaussian", {})
        if not cameras:
            return None
        if default_camera_name not in cameras:
            return None
        return cameras[default_camera_name][-1]

    def save_record(frames):
        if not frames:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(record_dir, f"{timestamp}.mp4")
        with imageio.get_writer(video_path, fps=20) as writer:
            for frame in frames:
                if frame is None:
                    continue
                writer.append_data(_to_uint8(frame))

    if args.store:
        record_dir = os.path.join(os.getcwd(), "simulation_record")
        os.makedirs(record_dir, exist_ok=True)
        record_frames.append(extract_frame(obs))

    # Start visualizer server when 'o' key is pressed
    viser = Viewer(0, 0, mode='server', host=args.host, port=args.port, physics_dt=physics_dt)
    action = [0, 0]

    for i in range(1, 100000):
        o, r, tm, tc, info = env.step(action)

        if args.store:
            record_frames.append(extract_frame(o))
        # print(i)
        if tm or tc :
            if args.store:
                save_record(record_frames)
                record_frames = []
                breakpoint()
            obs, _ = env.reset()
            if args.store:
                record_frames.append(extract_frame(obs))
            action = [0, 0]
            continue
        if viser.is_running():
            o_for_vis = o["gaussian"]["camera_front_tele_30fov"][-1]
            turn_signal = o["navigation"]["turn_signal"]
            action = viser.run(o_for_vis)

    env.close()
    viser.shutdown()
