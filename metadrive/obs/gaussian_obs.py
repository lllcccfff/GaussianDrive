import gymnasium as gym
import numpy as np

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.obs.observation_base import BaseObservation

import torch


class GaussianStateObservation(BaseObservation):
    """
    Use ego state info, navigation info and front cam image/top down image as input
    The shape needs special handling
    """
    IMAGE = "image"
    STATE = "state"

    def __init__(self, config):
        super().__init__(config)
        self.img_obs = GaussianObservation(config)

    def reset(self, **kwargs):
        self.img_obs.reset(**kwargs)

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                self.IMAGE: self.img_obs.observation_space,
                self.STATE: None
            }
        )

    def observe(self):
        return {self.IMAGE: self.img_obs.observe(), self.STATE: None}

    def destroy(self):
        super().destroy()
        self.img_obs.destroy()
        # self.state_obs.destroy()


class GaussianObservation(BaseObservation):
    """
    Use only image info as input
    """
    STACK_SIZE = 3  # use continuous 3 image as the input

    def __init__(self, config):
        super().__init__(config)
        self.STACK_SIZE = config["stack_size"]
        self.clip_rgb = config['clip_rgb']

    def reset(self, controller, render_fn, camera_params, **kwargs):
        """
        Clear stack
        :param env: MetaDrive
        :param vehicle: BaseVehicle
        :return: None
        """
        
        self.controller = controller
        self.render_fn = render_fn
        self.params = camera_params

        if self.clip_rgb:
            self.state = {cam_name: np.zeros(self.an_observation_shape(cam['H'], cam['W']), dtype=np.float32) for cam_name, cam in self.params.items()}
        else:
            self.state = {cam_name: np.zeros(self.an_observation_shape(cam['H'], cam['W']), dtype=np.uint8) for cam_name, cam in self.params.items()}


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
        ego_pose = torch.tensor(self.controller.transform, device='cuda').inverse()
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