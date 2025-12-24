# StreetWorld

StreetWorld is built on MetaDrive, integrating Gaussian splatting rendering and simulation. It discovers scenes automatically from a scene-config directory and supports remote visualization in a client/server mode for interactive driving or RL training.

## Installation
- Install dependencies:
  ```bash
  pip install -e .[gym]
  ```
- If you use the default Gaussian renderer, ensure `easydrive` is installed.

## Quickstart Example
1) Start the server (handles simulation + rendering and waits for clients):
```bash
python -m metadrive.examples.drive_in_real_env_server \
  --scene_config_directory /path/to/scene_configs \
  --host <server-ip> \
  --port <server-port>
```

2) Start the client (on the same or a remote machine, keyboard control):
```bash
python -m metadrive.examples.remote_visualizer --host <server-ip> --port <server-port> --width <window_size_width> --height <window_size_height>
```

Controls: `W/A/S/D` drive.

## Integrate Your Gaussian Renderer
Read `GS_INTERGRATION.md` and implement the `SimulatorInterface` methods `load_metadata / load_model / render` to plug in custom rendering.

## Environment Config
We provide defualt environment config in `metadrive\default_config.py`
User are able to edit the component of actor and participant through `actor_config` and `participant_config`.
For each component (observer, policy, controller), user are required to provide component type and configuration.
And currently user can not config participant's controller, as it is automatically decided by the object's type.

## Scene Config
Place each scene config file under `SCENE_CONFIG_DIRECTORY` so `ScenarioEnv` can scan and recognize the reconstructed scene assets.
