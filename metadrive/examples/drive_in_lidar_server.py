# #!/usr/bin/env python
# """
# This script demonstrates how to use the environment where traffic and road map are loaded from Waymo dataset.
# """
# import argparse
# from metadrive.constants import HELP_MESSAGE
# from metadrive.engine.asset_loader import AssetLoader
# from metadrive.envs.scenario_env import ScenarioEnv
# from easydrive.models.scenes.simulator_interface_lidar import SimulatorInterface
# from metadrive.viewer.viewer import Viewer
# import imageio
# import os
# import shutil
# import torch
# RENDER_MESSAGE = {
#     "Quit": "ESC",
#     "Switch perspective": "Q or B",
#     "Reset Episode": "R",
#     "Keyboard Control": "W,A,S,D",
#     "Start Visualizer Server": "O",
# }



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--reactive_traffic", action="store_true")
#     parser.add_argument("--waymo", action="store_true")
#     parser.add_argument("--add_sensor", action="store_true")
#     parser.add_argument("-c", "--scene_config_directory", type=str)
#     parser.add_argument('--host', type=str, default='localhost', help='Server IP')
#     parser.add_argument('--port', type=int, default=56789, help='Server port')
#     args = parser.parse_args()
#     asset_path = AssetLoader.asset_path
#     use_waymo = args.waymo
#     print(HELP_MESSAGE)

#     cfg = {
#         "scene_config_directory": args.scene_config_directory,
#     }
#     if args.add_sensor:
#         additional_cfg = {
#             'image_observation': True,
#         }
#         cfg.update(additional_cfg)
    
#     model = SimulatorInterface()

#     env = ScenarioEnv(model, cfg)
#     obs, _ = env.reset()

#     # Start visualizer server when 'o' key is pressed
#     viser = Viewer(0, 0, mode='server', host=args.host, port=args.port)
#     action = [0, 0]

#     img_stack = []
#     imgnum = 0
#     shutil.rmtree('recorded_images', ignore_errors=True)
#     os.makedirs('recorded_images', exist_ok=True)
#     shutil.rmtree('recorded_depth', ignore_errors=True)
#     os.makedirs('recorded_depth', exist_ok=True)
#     shutil.rmtree('recorded_semantic', ignore_errors=True)
#     os.makedirs('recorded_semantic', exist_ok=True)
#     shutil.rmtree('recorded_lidar', ignore_errors=True)
#     os.makedirs('recorded_lidar', exist_ok=True)
#     for i in range(1, 100000):
#         o, r, tm, tc, info = env.step(action)
        
#         if tm or tc:
#             for i, img_mm in enumerate(img_stack):
#                 img, depth, semantic, lidar = img_mm
#                 imageio.imwrite(f'recorded_images/{imgnum}.png', img)
#                 imageio.imwrite(f'recorded_depth/{imgnum}.png', depth)
#                 imageio.imwrite(f'recorded_semantic/{imgnum}.png', semantic)
#                 torch.save(lidar, f'recorded_lidar/{imgnum}.pt')
#                 imgnum += 1
#             img_stack = []
#             env.reset()
#             action = [0, 0]
#             continue

#         if viser.is_running():
#             o_for_vis = o['gaussian']['rgb']['FRONT'][-1]
#             img_stack.append((
#                 o_for_vis,
#                 o['gaussian']['depth'],
#                 o['gaussian']['semantic'],
#                 o['gaussian']['lidar']
#             ))
#             turn_signal = o['navigation']['turn_signal']
#             print('[Turn Signal] ', turn_signal)
#             action = viser.run(o_for_vis)

#     env.close()
#     viser.shutdown()