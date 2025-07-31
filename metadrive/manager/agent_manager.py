import copy
import numpy as np
from gymnasium.spaces import Space
from metadrive.constants import DEFAULT_AGENT
from metadrive.engine.logger import get_logger
from metadrive.manager.base_manager import BaseManager
from metadrive.policy.AI_protect_policy import AIProtectPolicy
from metadrive.policy.idm_policy import TrajectoryIDMPolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy, TakeoverPolicy, TakeoverPolicyWithoutBrake
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy

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
        super().__init__(self)
        # for getting {agent_id: BaseObject}, use agent_manager.active_agents

        # fake init. before creating engine, it is necessary when all objects re-created in runtime
        self.observations = copy.copy(init_observations)  # its value is map<agent_id, obs> before init() is called
        self._init_observations = init_observations  # map <agent_id, observation>

        # init spaces before initializing env.engine
        observation_space = {
            agent_id: single_obs.observation_space
            for agent_id, single_obs in init_observations.items()
        }
        init_action_space = self._get_action_space()
        assert isinstance(init_action_space, dict)
        assert isinstance(observation_space, dict)
        self._init_observation_spaces = observation_space
        self._init_action_spaces = init_action_space
        self.observation_spaces = copy.copy(observation_space)
        self.action_spaces = copy.copy(init_action_space)
        self.episode_created_agents = None
        
        # For single-agent env

        self._agent_finished_this_frame = None  # for observation space
        self._dying_countdown = 0

    def _create_agent(self, config_dict: dict):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type, vehicle_type
        # Only create one agent - use the first config or default agent
        agent_id = list(config_dict.keys())[0] if config_dict else "default_agent"
        v_config = list(config_dict.values())[0] if config_dict else {}

        v_type = random_vehicle_type(self.np_random) if self.engine.global_config["random_agent_model"] else \
            vehicle_type[v_config["vehicle_model"] if v_config.get("vehicle_model", False) else "default"]

        obj_name = agent_id if self.engine.global_config["force_reuse_object_name"] else None
        
        current_metadata = self.engine.data_manager.get_current_scenario_data()
        cameras, agent_state = current_metadata['cameras_objects'], current_metadata['agent_state']
        ground_height = current_metadata['ground_height']
        
        ego_poses = current_metadata['ego_poses']
        self.ego2cameras = self._calc_ego2camera(obj, cameras, ego_poses, ground_height)

        p = [agent_state['spawn_position'][0], agent_state['spawn_position'][1], ground_height + v_type.HEIGHT / 2]
        obj = self.spawn_object(
            v_type, 
            vehicle_config=v_config, 
            name=obj_name,
            position=p,
            heading=agent_state['spawn_heading']
        )
        self.init_pos = agent_state['spawn_position']
        self.dest_pos = agent_state['destination']

        policy_cls = self.agent_policy
        args = [obj, self.generate_seed(), ]
        self.add_policy(obj.id, policy_cls, *args)

        # Return single agent instead of dictionary
        return obj

    def _calc_ego2camera(self, vehicle_object, cameras, ego_poses, ground_height):
        ego2cameras = {}
        for cam_name, camera in cameras:
            w2c = camera.world_view_transform[0]
            ego2world = ego_poses[0]
            ego2world[2, 3] = ground_height + vehicle_object.HEIGHT / 2
            ego2cameras[cam_name] = np.linalg.inv(ego2world) @ w2c
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
        if get_global_config()["agent_policy"] in [TakeoverPolicy, TakeoverPolicyWithoutBrake]:
            return get_global_config()["agent_policy"]
        if get_global_config()["manual_control"]:
            policy = ManualControlPolicy
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
        self.episode_created_agent = self._create_agent(config_dict=self.engine.global_config["agent_configs"])
        self.observations.reset()


    def after_reset(self):
        # it is used when reset() is called to reset its original agent_id
        self._agent_object = self.episode_created_agent

        assert isinstance(self.action_spaces, Space)

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
            policy = self.get_policy(self._agent_to_object[agent_id])
            assert policy is not None, "No policy is set for agent {}".format(agent_id)
            if isinstance(policy, ReplayTrafficParticipantPolicy):
                if stage == "after_step":
                    policy.act(agent_id)
                    step_infos[agent_id] = policy.get_action_info()
                else:
                    step_infos[agent_id] = self._agent_object.before_step([0, 0])
            else:
                if stage == "before_step":
                    action = policy.act(agent_id)
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
        ego_pose = self._agent_object.get_transform()
        camera_pose_in_ego = self.ego2cameras
        sensor_pose = {}
        for sensor_name, camera_pose in camera_pose_in_ego.items():
            sensor_pose[sensor_name] = camera_pose @ np.linalg.inv(ego_pose)
        return sensor_pose

    def get_observations(self, step):
        return self.observations.observe(step, self.get_sensor_pose(), self._agent_object, self.engine.traffic_manager.traffic_poses)

    def get_observation_spaces(self):
        return self.observation_space

    def get_action_spaces(self):
        return self.action_spaces

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
        self._agent_to_object = {}
        self._object_to_agent = {}
        self._active_objects = {}
        for obs in self.observations.values():
            obs.destroy()
        self.observations = {}
        self.observation_spaces = {}
        self.action_spaces = {}

        self.INITIALIZED = False

        self._agents_finished_this_frame = {}