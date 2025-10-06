import copy
import numpy as np
import torch
from gymnasium.spaces import Space
from metadrive.constants import DEFAULT_AGENT
from metadrive.utils.logger import get_logger
from metadrive.manager.base_manager import BaseManager
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.policy.replay_policy import ReplayPolicy
from metadrive.base_class.base_object import BaseObject
logger = get_logger()
class AgentManager(BaseManager):
    """
    This class maintains a single vehicle agent in the environment.
    Simplified to handle only one vehicle object instead of multiple agents.

    Note:
    agent name: Single agent name (typically default_agent)
    object name: The unique name for the single vehicle object
    """
    INITIALIZED = False  # when vehicle instance is created, it will be set to True

    def __init__(self, config, step_manager):
        """
        The real init is happened in self.init(), in which super().__init__() will be called
        """
        """
        The real init is happened in self.init(), in which super().__init__() will be called
        """
        super().__init__()
        self.INITIALIZED = False

        # for getting {agent_id: BaseObject}, use agent_manager.active_agents
        self.config = config
        self.step_manager = step_manager
        self.observer = None
        self.policy = None
        
    def lazy_init(self):
        self.observer = self.config['observer'](self.config['observer_config'])
        self.policy = self.config['policy'](step_manager=self.step_manager, config=self.config['policy_config'])
        self.INITIALIZED = True
        
    def reset(self, config=None, **kwargs):

        """
        Agent manager is really initialized after the BaseObject Instances are created
        """
        self.last_observation = None
        if config is not None:
            self.config = config

        if not self.INITIALIZED:
            self.lazy_init()
        
        self.controller = self._create_agent(**kwargs)

        self.observer.reset(controller=self.controller, seed=self.generate_seed(), **kwargs)
        self.policy.reset(controller=self.controller, seed=self.generate_seed(), **kwargs)

        if self.policy.spawn_frame == self.step_manager.current_frame:
            self.controller.attachDyWld()
        
        assert isinstance(self.get_action_spaces(), Space)
        
    def _create_agent(self, physics_world, init_state, **kwargs):
        # Only create one agent - use the first config or default agent
        obj_name = "default_agent"

        obj = self.spawn_object(
            self.config['controller'], 
            name=obj_name,
            config=self.config['controller_config'], 
            physics_world=physics_world,
            random_seed=self.generate_seed(),
            size=self.config['controller_config'].get('size', None),
            position=init_state['spawn_position'],
            heading=init_state['spawn_heading']
        )
        self.init_pos = init_state['spawn_position']
        self.dest_pos = init_state['destination']
        return obj

    # def _calc_ego2camera(self, vehicle_object, cameras, ego_poses, start_frame):
    #     ego2cameras = {}
    #     ego2world = ego_poses[start_frame].cuda()
    #     for cam_name, camera in cameras.items():
    #         w2c = camera.world_view_transform[start_frame].T
    #         ego2cameras[cam_name] = w2c @ ego2world
    #     return ego2cameras

    def step(self, action=None):

        """
        Some policies should make decision before physics world actuation, in particular, those need decision-making
        But other policies like ReplayPolicy should be called in after_step, as they already know the final state and
        exempt the requirement for rolling out the dynamic system to get it.
        """
        if not self.is_spawned:
            return
        elif self.policy.spawn_frame == self.step_manager.current_frame:
            self.controller.attachDyWld()
        elif self.is_arrive:
            if self.controller is not None:
                self.clear_object(self.controller.id)
                self.controller = None
        else:
            if isinstance(self.policy, EnvInputPolicy):
                action = self.policy.act(action)
            else:
                action = self.policy.act(self.last_observation)
            
            if isinstance(self.policy, ReplayPolicy):
                self.controller.move(state_info=action)
            else:
                self.controller.move(action=action)

        return

    def observe(self):
        if self.is_arrive or not self.is_spawned:
            return {}
        
        self.last_observation = self.observer.observe()
        return {'observation': self.last_observation}

    @property
    def is_arrive(self):
        return self.policy.is_arrive
    
    @property
    def is_spawned(self):
        return self.policy.is_spawned
    
    def get_pose(self):
        return self.controller.transform
    
    def get_observation_spaces(self):
        return self.observer.observation_space

    def get_action_spaces(self):
        return self.policy.get_input_space()

    def get_state(self):
        ret = super().get_state()
        ret["created_agents"] = self._agent_object.name
        return ret
    
    def destroy(self):
        # when new agent joins in the game, we only change this two maps.
        if self.INITIALIZED:
            super().destroy()
        self.clear_all_objects()
        self.observer.destroy()
        self.policy.destroy()

        self.controller = None
        self.observer = None
        self.policy = None

        self.INITIALIZED = False
