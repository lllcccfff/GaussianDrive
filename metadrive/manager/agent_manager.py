import copy
import numpy as np
import torch
from gymnasium.spaces import Space
from metadrive.constants import DEFAULT_AGENT
from metadrive.engine.logger import get_logger
from metadrive.manager.base_manager import BaseManager
from metadrive.policy.env_input_policy import EnvInputPolicy

logger = get_logger()
class VehicleAgentManager(BaseManager):
    """
    This class maintains a single vehicle agent in the environment.
    Simplified to handle only one vehicle object instead of multiple agents.

    Note:
    agent name: Single agent name (typically default_agent)
    object name: The unique name for the single vehicle object
    """
    INITIALIZED = False  # when vehicle instance is created, it will be set to True

    def __init__(self, init_observations):
        """
        The real init is happened in self.init(), in which super().__init__() will be called
        """
        """
        The real init is happened in self.init(), in which super().__init__() will be called
        """
        # for getting {agent_id: BaseObject}, use agent_manager.active_agents
        # fake init. before creating engine, it is necessary when all objects re-created in runtime
        self.observations = init_observations  # its value is map<agent_id, obs> before init() is called
        self.episode_created_agents = None
        

    def _create_agent(self, config_dict: dict):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type, vehicle_type
        # Only create one agent - use the first config or default agent
        agent_id = "default_agent"

        v_type = random_vehicle_type(self.np_random) if self.engine.global_config["random_agent_model"] else \
            vehicle_type[config_dict["vehicle_model"] if config_dict.get("vehicle_model", False) else "default"]

        obj_name = agent_id if self.engine.global_config["force_reuse_object_name"] else None
        
        current_metadata = self.engine.data_manager.get_current_scenario_data()
        cameras, agent_state = current_metadata['camera_objects'], current_metadata['agent_state']
        ground_height = current_metadata['ground_height']
                
        ego_poses = current_metadata['ego_poses']
        p = [agent_state['spawn_position'][0], agent_state['spawn_position'][1], ground_height + v_type.DEFAULT_HEIGHT / 2]
        obj = self.spawn_object(
            v_type, 
            vehicle_config=config_dict, 
            name=obj_name,
            position=p,
            random_seed=self.generate_seed(),
            heading=agent_state['spawn_heading']
        )
        self.init_pos = agent_state['spawn_position']
        self.dest_pos = agent_state['destination']
        self.ego2cameras = self._calc_ego2camera(obj, cameras, ego_poses)

        args = [obj, self.generate_seed(), ]
        self.add_policy(obj.id, self.agent_policy, *args)

        # Return single agent instead of dictionary
        return obj

    def _calc_ego2camera(self, vehicle_object, cameras, ego_poses):
        ego2cameras = {}
        for cam_name, camera in cameras.items():
            w2c = camera.world_view_transform[0].T
            ego2world = torch.from_numpy(vehicle_object.transform).cuda()

            ego2cameras[cam_name] = w2c @ ego2world
        return ego2cameras

    @property
    def agent_policy(self):
        """Get the agent policy class

        Make sure you access the global config via get_global_config() instead of self.engine.global_config

        Returns:
            Agent Policy class
        """
        from metadrive.engine.engine_utils import get_global_config
        # Takeover policy shares the control between RL agent (whose action is input via env.step)
        # and external control device (whose action is input via controller).
        if not get_global_config()["agent_policy"]:
            policy = EnvInputPolicy
        else:
            policy = get_global_config()["agent_policy"]
        return policy

    def before_reset(self):
        if not self.INITIALIZED:
            super().__init__()
            self.INITIALIZED = True
        self.episode_created_agents = None

        super().before_reset()

    def reset(self):
        """
        Agent manager is really initialized after the BaseObject Instances are created
        """
        self.episode_created_agent = self._create_agent(config_dict=self.engine.global_config["vehicle_config"])
        self.observations['default_agent'].reset()


    def after_reset(self):
        # it is used when reset() is called to reset its original agent_id
        self._agent_object = self.episode_created_agent

        assert isinstance(self.get_action_spaces(), Space)

    def try_actuate_agent(self, step_infos, stage="before_step"):
        """
        Some policies should make decision before physics world actuation, in particular, those need decision-making
        But other policies like ReplayPolicy should be called in after_step, as they already know the final state and
        exempt the requirement for rolling out the dynamic system to get it.
        """
        assert stage == "before_step" or stage == "after_step"
        # Handle single agent
        if self._agent_object:
            agent_id = self._agent_object.id  # Get the single agent
            policy = self.get_policy(agent_id)
            assert policy is not None, "No policy is set for agent {}".format(agent_id)

            if stage == "before_step":
                action = policy.act(self.engine.cur_action)
                step_infos[agent_id] = policy.get_action_info()
                step_infos[agent_id].update(self._agent_object.before_step(action))

        return step_infos

    def before_step(self):
        # not in replay mode
        step_infos = self.try_actuate_agent(dict(), stage="before_step")
        return step_infos

    def after_step(self, *args, **kwargs):
        step_infos = self.try_actuate_agent({}, stage="after_step")
        step_infos.update({'after_step': self._agent_object.after_step()})
        return step_infos

    def get_sensor_pose(self):
        ego_pose = self._agent_object.transform
        camera_pose_in_ego = self.ego2cameras
        sensor_pose = {}
        for sensor_name, ego2camera in camera_pose_in_ego.items():
            sensor_pose[sensor_name] = ego2camera @ torch.tensor(ego_pose, device='cuda').inverse()
        return sensor_pose

    def get_observations(self):
        step = self.engine.traffic_manager.current_frame
        traffics = self.engine.traffic_manager.traffic_poses
        return self.observations['default_agent'].observe(step, self.get_sensor_pose(), traffics)

    def get_observation_spaces(self):
        return self.observation.observation_space

    def get_action_spaces(self):
        return self.get_policy(self._agent_object.id).get_input_space()

    def get_state(self):
        ret = super().get_state()
        ret["created_agents"] = self.episode_created_agent.name
        return ret

    def is_arrive(self):
        p = self._agent_object.position
        dest = self.dest_pos
        return (p[0] - dest[0])**2 + (p[1] - dest[1])**2 < 4

    @property
    def just_terminated_agents(self):
        assert not self.engine.replay_episode
        ret = {}
        for agent_name, v_name in self._agents_finished_this_frame.items():
            v = self.get_agent(v_name, raise_error=False)
            ret[agent_name] = v
        return ret

    def destroy(self):
        # when new agent joins in the game, we only change this two maps.
        if self.INITIALIZED:
            super().destroy()
        for obs in self.observations.values():
            obs.destroy()
        self.observations = {}

        self.INITIALIZED = False
