import gymnasium as gym
import numpy as np

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.obs.observation_base import BaseObservation
import torch
from scipy.spatial.transform import Rotation as R

class GaussianObservation(BaseObservation):
    """
    Use only image info as input
    """
    STACK_SIZE = 3  # use continuous 3 image as the input

    def __init__(self, config):
        super().__init__(config)
        self.STACK_SIZE = config["stack_size"]
        self.clip_rgb = config['clip_rgb']
        self.camera_configs = config.get('cameras', {})

    def reset(self, controller, render_fn, camera_params, **kwargs):
        """
        Clear stack
        :param env: MetaDrive
        :param vehicle: BaseVehicle
        :return: None
        """
        
        self.controller = controller
        self.render_fn = render_fn
        self.build_camera_params(camera_params)

        if self.clip_rgb:
            self.state = {cam_name: np.zeros(self.an_observation_shape(cam['H'], cam['W']), dtype=np.float32) for cam_name, cam in self.params.items()}
        else:
            self.state = {cam_name: np.zeros(self.an_observation_shape(cam['H'], cam['W']), dtype=np.uint8) for cam_name, cam in self.params.items()}

    def build_camera_params(self, _camera_params):
        if not self.camera_configs:
            self.params = _camera_params
            return

        parsed_camera_params = {}
        R_ego2cam_base = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ], dtype=np.float32)

        for cam_name, cam_cfg in self.camera_configs.items():
            H = int(cam_cfg["H"])
            W = int(cam_cfg["W"])
            focal = float(cam_cfg["focal"])
            K = torch.tensor([
                [focal, 0, W / 2.0],
                [0, focal, H / 2.0],
                [0, 0, 1]
            ], dtype=torch.float32)

            hpr = np.asarray(cam_cfg["hpr"], dtype=np.float32)
            hpr_rad = np.deg2rad(hpr)
            R_additional = R.from_euler('ZYX', hpr_rad, degrees=False).as_matrix()
            R_final = np.asarray(R_ego2cam_base @ R_additional, dtype=np.float32)

            offset = np.asarray(cam_cfg["offset"], dtype=np.float32)
            translation = -np.asarray(R_final @ offset, dtype=np.float32)

            ego2camera = np.eye(4, dtype=np.float32)
            ego2camera[:3, :3] = R_final
            ego2camera[:3, 3] = translation

            parsed_camera_params[cam_name] = {
                "K": K,
                "H": H,
                "W": W,
                "ego2camera": torch.from_numpy(ego2camera)
            }

        self.params = parsed_camera_params
        


    @property
    def observation_space(self):
        # sensor_cls = self.config["sensors"][self.image_source][0]
        # assert sensor_cls == "MainCamera" or issubclass(sensor_cls, BaseCamera), "Sensor should be BaseCamera"
        
        space = {}
        for name, sensor in self.params.items():
            shape = self.an_observation_shape(sensor['H'], sensor['W'])
            if self.clip_rgb:
                space[name] = gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32)
            else:
                space[name] = gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
        return space

    def an_observation_shape(self, h, w):
        return (self.STACK_SIZE, h, w, 3)
 
    def observe(self):
        """
        Get the image Observation. By setting new_parent_node and the reset parameters, it can capture a new image from
        a different position and pose
        """
        ego_pose = torch.tensor(self.controller.transform).inverse()
        for cam_name, params in self.params.items():
            extrinsics = params['ego2camera'] @ ego_pose
            ret = self.render_fn(
                K=params['K'],
                H=params['H'],
                W=params['W'],
                extrinsics=extrinsics,
            )
            self.state[cam_name] = np.roll(self.state[cam_name], -1, axis=0)
            self.state[cam_name][-1] = ret


        return self.state


    def destroy(self):
        """
        Clear memory
        """
        super(GaussianObservation, self).destroy()
        self.state = None
