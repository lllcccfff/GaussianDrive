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
        self.img_obs = GaussianObservation(config, config["vehicle_config"]["image_source"], config["norm_pixel"])

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                self.IMAGE: self.img_obs.observation_space,
                self.STATE: None
            }
        )

    def observe(self, frame, camera_poses, vehicle: BaseVehicle):
        return {self.IMAGE: self.img_obs.observe(frame, camera_poses), self.STATE: None}

    def destroy(self):
        super().destroy()
        self.img_obs.destroy()
        self.state_obs.destroy()


class GaussianObservation(BaseObservation):
    """
    Use only image info as input
    """
    STACK_SIZE = 3  # use continuous 3 image as the input

    def __init__(self, config, clip_rgb: bool):
        self.STACK_SIZE = config["stack_size"]
        super().__init__(config)
        self.norm_pixel = clip_rgb
        self.state = {cam_name: np.zeros(self.observation_space.shape, dtype=np.float32) for cam_name in self.sensors.keys()}

    @property
    def observation_space(self):
        # sensor_cls = self.config["sensors"][self.image_source][0]
        # assert sensor_cls == "MainCamera" or issubclass(sensor_cls, BaseCamera), "Sensor should be BaseCamera"
        channel = 3
        shape = (self.config["sensors"][self.image_source][2],
                 self.config["sensors"][self.image_source][1]) + (channel, self.STACK_SIZE)
        shape = shape * len(self.sensors.keys())
        if self.norm_pixel:
            return gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32)
        else:
            return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)

    def observe(self, frame, camera_poses, position=None, hpr=None):
        """
        Get the image Observation. By setting new_parent_node and the reset parameters, it can capture a new image from
        a different position and pose
        """
        for cam_name, camera_pose in camera_poses:
            w2c = torch.tensor(camera_pose).cuda().contiguous()
            camera_center = w2c.inverse()[:3, 3]

            raw_camera = self.sensors[cam_name]
            full_proj = (w2c @ raw_camera.projection_matrix).contiguous()
            
            camera_params = {
                'world_view_transform': w2c.T,
                'full_proj_transform': full_proj.T,
                'camera_center': camera_center,
                'image_height': raw_camera.image_height,
                'image_width': raw_camera.image_width,
                'FoVx': raw_camera.FoVx,
                'FoVy': raw_camera.FoVy
            }    

            ray_batch = raw_camera.gen_rays(
                R=w2c[:3, :3].T, T=w2c[:3, 3], K=raw_camera.K,
                image_height=raw_camera.image_height, image_width=raw_camera.image_width
            )
            ret = self.engine.render(
                frame=frame,
                camera_params=camera_params,
                ray_batch=ray_batch
            )['rgb']
    
            self.state[cam_name] = np.roll(self.state[cam_name], -1, axis=-1)
            self.state[cam_name][..., -1] = ret
        return self.state

    def get_image(self):
        return self.state.copy()[:, :, -1]

    def reset(self, env, vehicle=None):
        """
        Clear stack
        :param env: MetaDrive
        :param vehicle: BaseVehicle
        :return: None
        """
        self.sensors = self.engine.data_manager.get_current_scenario_data()['camera_objects']

        self.state = {cam_name: np.zeros(self.observation_space.shape, dtype=np.float32) for cam_name in self.sensors.keys()}

    def destroy(self):
        """
        Clear memory
        """
        super(GaussianObservation, self).destroy()
        self.state = None