# Gaussian Splatting Integration Guide

This guide explains how to integrate your Gaussian splatting algorithm into the GaussianDrive simulator.

## Overview

To integrate your Gaussian splatting renderer, you need to implement a **SimulatorInterface** class that acts as a bridge between the MetaDrive simulator and your 3D Gaussian rendering pipeline.

## Scene Configuration Structure

Design Idea:

- **All scene configs are located in a single folder** specified by `scene_config_directory` (see examples/drive_in_real_env_server.py)
- **Each config file identifies one unique scene** with a unique scene name. The data type and internal is totally defined by your GS algorithm. It 
- The simulator **automatically discovers scenes** by iterating through all config files in this folder
- Your interface parses metadata and loads gaussian models by passing the config file of the corresponding scene

**Directory Structure Example:**
```
scene_config_directory/
├── scene_001.yaml          # Config for scene "scene_001"
├── scene_002.json          # Config for scene "scene_002"
└── downtown_drive.py     # Config for scene "downtown_drive"
```

When the environment initializes, it:
1. Scans `scene_config_directory` for all config files
2. For each config, calls `interface.load_metadata(cfg)` to parse scene data
3. Randomly selects a scene during episode resets
4. Calls `interface.load_model(cfg)` to load the gaussian model of the scene

## Required API Interface

Your `SimulatorInterface` class must implement the following methods:

### 1. `__init__(self, zNear=0.0001, zFar=1000)`

**Parameters:**
- `zNear` (float): Near clipping plane distance
- `zFar` (float): Far clipping plane distance

**Responsibilities:**
- Initialize required attributes

### 2. `load_metadata(self, cfg) -> tuple`

**Parameters:**
- `cfg`: Configuration object loaded from a single scene config file. Use this to identify which scene to load.

**Returns:**
A tuple of `(scene_name, frame_range, camera_params, ego_poses, tracking_data)`:
- `scene_name` (str): **Unique identifier for this scene** (extract from cfg)
- `frame_range` (list[int, int]): [start_frame, end_frame]
- `camera_params` (dict): Camera parameters for each camera view
  ```python
  {
      str(camera_name): {
          "K": list[3][3],        # 3x3 intrinsic matrix
          "H": int,               # Image height
          "W": int,               # Image width
          "ego2camera": list[4][4] # 4x4 transformation from ego to camera
      }
  }
  ```
- `ego_poses` (dict): Ego vehicle poses per frame
  ```python
  {
      int(frame): list[4][4]  # 4x4 ego-to-world transform matrix
  }
  ```
- `tracking_data` (dict): Tracked object trajectories
  ```python
  {
      str(object_id): {
          "transforms": {
              int(frame): list[4][4]  # 4x4 object-to-world transform
          },
          "size": list[3],          # [length, width, height]
          "type": str               # "vehicle", "pedestrian", "cyclist"
      }
  }
  ```

**Responsibilities:**
- **Identify the scene from cfg** (e.g., cfg.scene_name)
- Parse the scene data into the required structure
- Cache necessary data internally for later use

### 3. `load_model(self, cfg)`

**Parameters:**
- `cfg`: Configuration object for a specific scene. **Use this to identify which scene's model to load.**

**Responsibilities:**
- Initialize your rendering pipeline
- **Identify the scene from cfg** (same way as in `load_metadata`)
- Load the pre-trained Gaussian model weights for this specific scene
- prepare GPU/CUDA resources

### 4. `update_scene(self, frame, object_poses)`

**Parameters:**
- `frame` (int): Current frame number
- `object_poses` (dict): Current poses of dynamic objects
  ```python
  {
      "object_id": torch.Tensor[4, 4]  # 4x4 object-to-world transform
  }
  ```

**Responsibilities:**
- Update dynamic object poses in your scene
- Prepare scene for rendering at the given frame

### 5. `render(self, K, H, W, extrinsics) -> np.ndarray`

**Parameters:**
- `K` (torch.Tensor or list): 3x3 camera intrinsic matrix
- `H` (int): Image height in pixels
- `W` (int): Image width in pixels
- `extrinsics` (torch.Tensor or list): 4x4 camera-to-world transformation matrix

**Returns:**
- `np.ndarray`: Rendered RGB image of shape `(H, W, 3)` with values in range [0, 255] (uint8) or [0.0, 1.0] (float32)

**Responsibilities:**
- Render the scene from a given camera viewpoint.


## Coordinate Systems

