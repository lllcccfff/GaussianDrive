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
        self.img_obs = GaussianObservation(config, config["clip_rgb"])

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                self.IMAGE: self.img_obs.observation_space,
                self.STATE: None
            }
        )

    def observe(self, frame, camera_poses, extra_boxes=None):
        return {self.IMAGE: self.img_obs.observe(frame, camera_poses, extra_boxes), self.STATE: None}

    def reset(self):
        self.img_obs.reset()

    def destroy(self):
        super().destroy()
        self.img_obs.destroy()
        # self.state_obs.destroy()


class GaussianObservation(BaseObservation):
    """
    Use only image info as input
    """
    STACK_SIZE = 3  # use continuous 3 image as the input

    def __init__(self, config, clip_rgb: bool):
        self.STACK_SIZE = config["stack_size"]
        super().__init__(config)
        self.clip_rgb = clip_rgb

    @property
    def observation_space(self):
        # sensor_cls = self.config["sensors"][self.image_source][0]
        # assert sensor_cls == "MainCamera" or issubclass(sensor_cls, BaseCamera), "Sensor should be BaseCamera"
        
        space = {}
        for sname, sensor in self.sensors.items():
            shape = self.an_observation_shape(sensor.image_height, sensor.image_width)
            if self.clip_rgb:
                space[sname] = gym.spaces.Box(-0.0, 1.0, shape=shape, dtype=np.float32)
            else:
                space[sname] = gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
        return space

    def an_observation_shape(self, h, w):
        return (self.STACK_SIZE, h, w, 3)
 
    def observe(self, frame, camera_poses, extra_boxes=None):
        """
        Get the image Observation. By setting new_parent_node and the reset parameters, it can capture a new image from
        a different position and pose
        """
        for cam_name, camera_pose in camera_poses.items():
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
                ray_batch=ray_batch,
                extra_boxes=extra_boxes
            )['vis_rgb']
    
            self.state[cam_name] = np.roll(self.state[cam_name], -1, axis=0)
            self.state[cam_name][-1] = ret
        return self.state

    def reset(self, vehicle=None):
        """
        Clear stack
        :param env: MetaDrive
        :param vehicle: BaseVehicle
        :return: None
        """
        self.sensors = self.engine.data_manager.get_current_scenario_data()['camera_objects']
        self.state = {cam_name: np.zeros(self.an_observation_shape(cam.image_height, cam.image_width), dtype=np.float32) for cam_name, cam in self.sensors.items()}


    def destroy(self):
        """
        Clear memory
        """
        super(GaussianObservation, self).destroy()
        self.state = None