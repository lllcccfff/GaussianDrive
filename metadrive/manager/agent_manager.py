import copy
import math
import numpy as np
import torch
from gymnasium.spaces import Space
from metadrive.constants import DEFAULT_AGENT
from metadrive.utils.logger import get_logger
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.manager.base_manager import BaseManager
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.policy.replay_policy import ReplayPolicy
from metadrive.obs.gaussian_obs import GaussianObservation
from metadrive.obs.navigation_obs import NavigationObservation
from metadrive.base_class.base_object import BaseObject

logger = get_logger()


class AgentState:
    IDLE = "idle"
    NOT_SPAWN = "not_spawn"
    ALIVE = "alive"
    SUCCESS = "arrive_dest"
    OUT_OF_ROAD = "out_of_road"
    OUT_OF_STEP = "out_of_step"
    CRASH_VEHICLE = "crash_vehicle"
    CRASH_HUMAN = "crash_human"
    CRASH_OBJECT = "crash_object"
    CRASH_WORLD = "crash_world"


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
        self.max_step = config.get("max_step", 10_000)

        # for getting {agent_id: BaseObject}, use agent_manager.active_agents
        self.config = config
        self.step_manager = step_manager
        self.observer = None
        self.policy = None

    def lazy_init(self):
        self.observer = self.config["observer"](self.config["observer_config"])
        self.policy = self.config["policy"](step_manager=self.step_manager, config=self.config["policy_config"])
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
        self.state = AgentState.NOT_SPAWN

        self.observer.reset(controller=self.controller, seed=self.generate_seed(), **kwargs)
        self.policy.reset(controller=self.controller, seed=self.generate_seed(), **kwargs)

        if isinstance(self.observer, NavigationObservation):
            self.policy.destination = self.observer.destination

        if math.isclose(self.policy.spawn_timestamp, self.step_manager.current_timestamp):
            self.controller.attachDyWld()

        assert isinstance(self.get_action_spaces(), Space)

    def _create_agent(self, physics_world, init_state, **kwargs):
        # Only create one agent - use the first config or default agent
        obj_name = "default_agent"

        obj = self.spawn_object(
            self.config["controller"],
            name=obj_name,
            config=self.config["controller_config"],
            physics_world=physics_world,
            random_seed=self.generate_seed(),
            size=self.config["controller_config"].get("size", None),
            position=init_state["spawn_position"],
            heading_theta=init_state["spawn_heading"],
        )
        # self.init_pos = init_state['spawn_position']
        # self.dest_pos = init_state['destination']
        return obj

    def step(self, action=None):
        """
        Some policies should make decision before physics world actuation, in particular, those need decision-making
        But other policies like ReplayPolicy should be called in after_step, as they already know the final state and
        exempt the requirement for rolling out the dynamic system to get it.
        """
        if self.state == AgentState.ALIVE:
            if isinstance(self.policy, EnvInputPolicy):
                action = self.policy.act(action)
            else:
                action = self.policy.act(self.last_observation)

            if isinstance(self.policy, ReplayPolicy):
                self.controller.move(state_info=action)
            else:
                self.controller.move(action=action)

        return

    def update_state(self):
        """
        Derive and cache the agent's discrete state using policy signals.
        """
        # Not spawned yet
        if self.state == AgentState.NOT_SPAWN and self.policy.is_spawned:
            self.controller.attachDyWld()
            self.state = AgentState.ALIVE
            return

        if self.state == AgentState.ALIVE:
            # crash checks from controller
            if isinstance(self.controller, BaseVehicle):
                self.controller.crash_check()

                if self.controller.crash_human:
                    self.clear_all_objects()
                    self.state = AgentState.CRASH_HUMAN
                    return
                if self.controller.crash_vehicle:
                    self.clear_all_objects()
                    self.state = AgentState.CRASH_VEHICLE
                    return
                if self.controller.crash_object:
                    self.clear_all_objects()
                    self.state = AgentState.CRASH_OBJECT
                    return
                if self.controller.crash_world:
                    self.clear_all_objects()
                    self.state = AgentState.CRASH_WORLD
                    return

            if self.step_manager.eposide_step >= self.max_step:
                self.clear_all_objects()
                self.state = AgentState.OUT_OF_STEP
                return

            if not self.policy.is_in_trajectory:
                self.clear_all_objects()
                self.state = AgentState.OUT_OF_ROAD
                return

            if self.policy.is_arrive:
                self.clear_all_objects()
                self.state = AgentState.SUCCESS
                return

    def observe(self):
        if self.state == AgentState.ALIVE:
            self.last_observation = self.observer.observe()
        return {"observation": self.last_observation}

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
