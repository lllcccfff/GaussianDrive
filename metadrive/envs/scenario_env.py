"""
This environment can load all scenarios exported from other environments via env.export_scenarios()
"""

import numpy as np

import torch
from metadrive.manager.agent_manager import AgentState
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.scenario_curriculum_manager import ScenarioCurriculumManager
from metadrive.manager.scenario_data_manager import ScenarioDataManager, ScenarioOnlineDataManager
from metadrive.manager.scenario_map_manager import ScenarioMapManager
from metadrive.manager.agent_manager import AgentManager
from metadrive.utils import get_np_random
from metadrive.utils.math import wrap_to_pi

SCENARIO_ENV_CONFIG = dict(
    # ===== Scenario Config =====
    data_directory=AssetLoader.file_path("nuscenes", unix_style=False),
    start_scenario_index=0,

    # Set num_scenarios=-1 to load all scenarios in the data directory.
    num_scenarios=3,
    sequential_seed=False,  # Whether to set seed (the index of map) sequentially across episodes
    worker_index=0,  # Allowing multi-worker sampling with Rllib
    num_workers=1,  # Allowing multi-worker sampling with Rllib

    # ===== Curriculum Config =====
    curriculum_level=1,  # i.e. set to 5 to split the data into 5 difficulty level
    episodes_to_evaluate_curriculum=None,
    target_success_rate=0.8,

    # ===== Map Config =====
    store_map=True,
    store_data=True,
    need_lane_localization=True,
    no_map=False,
    map_region_size=1024,
    cull_lanes_outside_map=True,

    # ===== Scenario =====
    no_traffic=False,  # nothing will be generated including objects/pedestrian/vehicles
    no_static_vehicles=False,  # static vehicle will be removed
    no_light=False,  # no traffic light
    reactive_traffic=False,  # turn on to enable idm traffic
    filter_overlapping_car=True,  # If in one frame a traffic vehicle collides with ego car, it won't be created.
    default_vehicle_in_traffic=False,
    skip_missing_light=True,
    static_traffic_object=True,
    show_sidewalk=False,
    even_sample_vehicle_class=None,  # Deprecated.

    # ===== Reward Scheme =====
    # See: https://github.com/metadriverse/metadrive/issues/283
    success_reward=5.0,
    out_of_road_penalty=5.0,
    on_lane_line_penalty=1.,
    crash_vehicle_penalty=1.,
    crash_object_penalty=1.0,
    crash_human_penalty=1.0,
    driving_reward=1.0,
    steering_range_penalty=0.5,
    heading_penalty=1.0,
    lateral_penalty=.5,
    max_lateral_dist=4,
    no_negative_reward=True,

    # ===== Cost Scheme =====
    crash_vehicle_cost=1.0,
    crash_object_cost=1.0,
    out_of_road_cost=1.0,
    crash_human_cost=1.0,

    # ===== Termination Scheme =====
    out_of_route_done=False,
    crash_vehicle_done=False,
    crash_object_done=False,
    crash_human_done=False,
    relax_out_of_road_done=True,
)



class ScenarioEnv(BaseEnv):
    @classmethod
    def default_config(cls):
        config = super(ScenarioEnv, cls).default_config()
        config.update(SCENARIO_ENV_CONFIG)
        return config

    def __init__(self, model, config=None):
        super(ScenarioEnv, self).__init__(model, config)
        if self.config["curriculum_level"] > 1:
            assert self.config["num_scenarios"] % self.config["curriculum_level"] == 0, \
                "Each level should have the same number of scenarios"
            if self.config["num_workers"] > 1:
                num = int(self.config["num_scenarios"] / self.config["curriculum_level"])
                assert num % self.config["num_workers"] == 0
        if self.config["num_workers"] > 1:
            assert self.config["sequential_seed"], \
                "If using > 1 workers, you have to allow sequential_seed for consistency!"

    def _post_process_config(self, config):
        config = super(ScenarioEnv, self)._post_process_config(config)
        return config

    def _init_agent_manager(self):
        return AgentManager(self.config['actor_config'], self.step_manager)

    def done_function(self):
        state_info = self.agent_managers['actor'].state
        is_max_step = self.config["max_step"] is not None and self.episode_lengths >= self.config["max_step"]


        def msg(reason):
            return "Episode ended! Scenario Index: {} Scenario id: {} Reason: {}.".format(
                self.current_seed, self.data_manager.current_scenario_id, reason
            )
        
        done = False
        if state_info == AgentState.SUCCESS:
            done = True
            self.logger.debug(msg("arrive_dest"), extra={"log_once": True})
        elif state_info == AgentState.OUT_OF_ROAD:
            done = True
            self.logger.debug(msg("out_of_road"), extra={"log_once": True})
        elif state_info == AgentState.OUT_OF_STEP:
            done = True
            self.logger.debug(msg("out_of_step of object"), extra={"log_once": True})
        elif state_info == AgentState.CRASH_HUMAN:
            done = True
            self.logger.debug(msg("crash human"), extra={"log_once": True})
        elif state_info == AgentState.CRASH_VEHICLE:
            done = True
            self.logger.debug(msg("crash vehicle"), extra={"log_once": True})
        elif state_info == AgentState.CRASH_OBJECT:
            done = True
            self.logger.debug(msg("crash object"), extra={"log_once": True})
        elif state_info == AgentState.CRASH_WORLD:
            done = True
            self.logger.debug(msg("crash background"), extra={"log_once": True})
        elif is_max_step:
            state_info = AgentState.OUT_OF_STEP
            done = True
            self.logger.debug(msg("max step"), extra={"log_once": True})

        # # log data to curriculum manager
        # self.engine.curriculum_manager.log_episode(
        #     done_info[TerminationState.SUCCESS], vehicle.navigation.route_completion
        # )

        return done, {'reason': state_info}

    def cost_function(self):
        actor_mgr = self.agent_managers['actor']
        state = actor_mgr.state

        step_info = dict(num_crash_object=0, num_crash_human=0, num_crash_vehicle=0, num_on_line=0)
        cost = 0

        if state == AgentState.OUT_OF_ROAD:
            cost += self.config["out_of_road_cost"]
        if state == AgentState.CRASH_VEHICLE:
            cost += self.config["crash_vehicle_cost"]
            step_info["crash_vehicle_cost"] = self.config["crash_vehicle_cost"]
            step_info["num_crash_vehicle"] = 1
        if state == AgentState.CRASH_HUMAN:
            cost += self.config["crash_human_cost"]
            step_info["num_crash_human"] = 1
        if state == AgentState.CRASH_OBJECT:
            step_info["num_crash_object"] = 1

        step_info["cost"] = cost
        return cost, step_info

    def reward_function(self):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        state = self.agent_managers['actor'].state
        step_info = dict()

        # crash penalty
        reward = 0
        if state == AgentState.CRASH_VEHICLE:
            reward = -self.config["crash_vehicle_penalty"]
        if state == AgentState.CRASH_HUMAN:
            reward = -self.config["crash_human_penalty"]

        step_info["step_reward"] = reward

        # termination reward
        if state == AgentState.SUCCESS:
            reward = self.config["success_reward"]
        elif state == AgentState.OUT_OF_ROAD:
            reward = -self.config["out_of_road_penalty"]

        return reward, step_info

class ScenarioOnlineEnv(ScenarioEnv):
    """
    This environment allow the user to pass in scenario data directly.
    """
    def default_config(cls):
        config = super(ScenarioOnlineEnv, cls).default_config()
        config.update({
            "store_map": False,
        })
        return config

    def __init__(self, config=None):
        super(ScenarioOnlineEnv, self).__init__(config)
        self.lazy_init()

        assert self.config["store_map"] is False, \
            "ScenarioOnlineEnv should not store map. Please set store_map=False in config"

    def _setup(self):
        """Overwrite the data_manager by ScenarioOnlineDataManager"""
        super()._setup()
        self.engine.update_manager("data_manager", ScenarioOnlineDataManager())

    def set_scenario(self, scenario_data):
        """Please call this function before env.reset()"""
        self.engine.data_manager.set_scenario(scenario_data)


class ScenarioWaypointEnv(ScenarioEnv):
    """
    This environment use WaypointPolicy. Even though the environment still runs in 10 HZ, we allow the external
    waypoint generator generates up to 5 waypoints at each step (controlled by config "waypoint_horizon").
    Say at step t, we receive 5 waypoints. Then we will set the agent states for t+1, t+2, t+3, t+4, t+5 if at
    t+1 ~ t+4 no additional waypoints are received. Here is the full timeline:

    step t=0: env.reset(), initial positions/obs are sent out. This corresponds to the t=0 or t=10 in WOMD dataset
    (TODO: we should allow control on the meaning of the t=0)
    step t=1: env.step(), agent receives 5 waypoints, we will record the waypoint sequences. Set agent state for t=1,
        and send out the obs for t=1.
    step t=2: env.step(), it's possible to get action=None, which means the agent will use the cached waypoint t=2,
        and set the agent state for t=2. The obs for t=2 will be sent out. If new waypoints are received, we will \
        instead set agent state to the first new waypoint.
    step t=3: ... continues the loop and receives action=None or new waypoints.
    step t=4: ...
    step t=5: ...
    step t=6: if we only receive action at t=1, and t=2~t=5 are all None, then this step will force to receive
        new waypoints. We will set the agent state to the first new waypoint.

    Most of the functions are implemented in WaypointPolicy.
    """
    @classmethod
    def default_config(cls):
        config = super(ScenarioWaypointEnv, cls).default_config()
        return config

    def _post_process_config(self, config):
        ret = super(ScenarioWaypointEnv, self)._post_process_config(config)
        assert config["set_static"], "Waypoint policy requires set_static=True"
        return ret

