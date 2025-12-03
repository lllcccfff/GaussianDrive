# GaussianDrive

GaussianDrive is built on MetaDrive, integrating Gaussian splatting rendering and real-scene replay. It discovers scenes automatically from a scene-config directory and supports remote visualization in a client/server mode for interactive driving or RL training.

## Installation
- Install dependencies:
  ```bash
  pip install -e .[gym]
  ```
- If you use the default Gaussian renderer, ensure `easydrive` is installed.

## Quickstart
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

## Scene Config
Place each scene config file under `SCENE_CONFIG_DIRECTORY` so `ScenarioEnv` can scan and recognize the reconstructed scene assets.

## FAQ
- `SimulatorInterface` import fails: confirm your renderer package is installed or added to `PYTHONPATH`.
- Scene not discovered: check that `--scene_config_directory` points to a directory containing per-scene config files.
- Rendering issues: ensure GPU/driver are normal; `render()` should return an RGB array of shape `(H, W, 3)`.
