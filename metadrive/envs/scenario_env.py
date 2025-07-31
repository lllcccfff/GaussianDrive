"""
This environment can load all scenarios exported from other environments via env.export_scenarios()
"""

import numpy as np

import torch
from metadrive.component.navigation_module.trajectory_navigation import TrajectoryNavigation
from metadrive.constants import TerminationState
from metadrive.engine.asset_loader import AssetLoader
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.scenario_agent_manager import ScenarioAgentManager
from metadrive.manager.scenario_curriculum_manager import ScenarioCurriculumManager
from metadrive.manager.scenario_data_manager import ScenarioDataManager, ScenarioOnlineDataManager
from metadrive.manager.scenario_light_manager import ScenarioLightManager
from metadrive.manager.scenario_map_manager import ScenarioMapManager
from metadrive.manager.scenario_traffic_manager import ScenarioTrafficManager
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.policy.waypoint_policy import WaypointPolicy
from metadrive.utils import get_np_random
from metadrive.utils.math import wrap_to_pi
from metadrive.engine.engine_utils import set_global_random_seed

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

    # ===== Agent config =====
    vehicle_config=dict(
        navigation_module=TrajectoryNavigation,
        lidar=dict(num_lasers=120, distance=50),
        lane_line_detector=dict(num_lasers=0, distance=50),
        side_detector=dict(num_lasers=12, distance=50),
    ),
    # If set_static=True, then the agent will not "fall from the sky". This will be helpful if you want to
    # capture per-frame data for the agent (for example for collecting static sensor data).
    # However, the physics simulation of the agent will be disable too. So in the visualization, the image will be
    # very chunky as the agent will suddenly move to the next position for each step.
    # Set to False for better visualization.
    set_static=False,

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

    # ===== others =====
    allowed_more_steps=None,  # horizon, None=infinite
    top_down_show_real_size=False,
    use_bounding_box=False,  # Set True to use a cube in visualization to represent every dynamic objects.
)

SCENARIO_WAYPOINT_ENV_CONFIG = dict(
    # How many waypoints will be used at each environmental step. Checkout ScenarioWaypointEnv for details.
    waypoint_horizon=5,
    agent_policy=WaypointPolicy,

    # Must set this to True, otherwise the agent will drift away from the waypoint when doing
    # "self.engine.step(self.config["decision_repeat"])" in "_step_simulator".
    set_static=True,
)


class ScenarioEnv(BaseEnv):
    @classmethod
    def default_config(cls):
        config = super(ScenarioEnv, cls).default_config()
        config.update(SCENARIO_ENV_CONFIG)
        return config

    def __init__(self, config=None):
        super(ScenarioEnv, self).__init__(config)
        if self.config["curriculum_level"] > 1:
            assert self.config["num_scenarios"] % self.config["curriculum_level"] == 0, \
                "Each level should have the same number of scenarios"
            if self.config["num_workers"] > 1:
                num = int(self.config["num_scenarios"] / self.config["curriculum_level"])
                assert num % self.config["num_workers"] == 0
        if self.config["num_workers"] > 1:
            assert self.config["sequential_seed"], \
                "If using > 1 workers, you have to allow sequential_seed for consistency!"
        self.start_index = self.config["start_scenario_index"]

    def _post_process_config(self, config):
        config = super(ScenarioEnv, self)._post_process_config(config)
        return config

    def _init_agent_manager(self):
        return ScenarioAgentManager(init_observations=self._get_observations())

    def setup_engine(self):
        super(ScenarioEnv, self).setup_engine()
        self.engine.register_manager("data_manager", ScenarioDataManager())
        self.engine.register_manager("map_manager", ScenarioMapManager())
        if not self.config["no_traffic"]:
            self.engine.register_manager("traffic_manager", ScenarioTrafficManager())
        # self.engine.register_manager("curriculum_manager", ScenarioCurriculumManager()) 

    def done_function(self):
        vehicle = self.agent
        done = False
        is_max_step = self.config["max_step"] is not None and self.episode_lengths >= self.config["max_step"]
        done_info = {
            TerminationState.CRASH_VEHICLE: vehicle.crash_vehicle,
            TerminationState.CRASH_OBJECT: vehicle.crash_object,
            TerminationState.CRASH_BUILDING: vehicle.crash_building,
            TerminationState.CRASH_HUMAN: vehicle.crash_human,
            TerminationState.CRASH_SIDEWALK: vehicle.crash_sidewalk,
            TerminationState.OUT_OF_ROAD: self._is_out_of_road(vehicle),
            TerminationState.SUCCESS: self._is_arrive_destination(vehicle),
            TerminationState.MAX_STEP: is_max_step,
            TerminationState.ENV_SEED: self.current_seed,
            # TerminationState.CURRENT_BLOCK: self.agent.navigation.current_road.block_ID(),
            # crash_vehicle=False, crash_object=False, crash_building=False, out_of_road=False, arrive_dest=False,
        }

        def msg(reason):
            return "Episode ended! Scenario Index: {} Scenario id: {} Reason: {}.".format(
                self.current_seed, self.engine.data_manager.current_scenario_id, reason
            )

        if done_info[TerminationState.SUCCESS]:
            done = True
            self.logger.debug(msg("arrive_dest"), extra={"log_once": True})
        elif done_info[TerminationState.OUT_OF_ROAD]:
            done = True
            self.logger.debug(msg("out_of_road"), extra={"log_once": True})
        elif done_info[TerminationState.CRASH_HUMAN] and self.config["crash_human_done"]:
            done = True
            self.logger.debug(msg("crash human"), extra={"log_once": True})
        elif done_info[TerminationState.CRASH_VEHICLE] and self.config["crash_vehicle_done"]:
            done = True
            self.logger.debug(msg("crash vehicle"), extra={"log_once": True})
        elif done_info[TerminationState.CRASH_OBJECT] and self.config["crash_object_done"]:
            done = True
            self.logger.debug(msg("crash object"), extra={"log_once": True})
        elif done_info[TerminationState.CRASH_BUILDING] and self.config["crash_object_done"]:
            done = True
            self.logger.debug(msg("crash building"), extra={"log_once": True})
        elif done_info[TerminationState.MAX_STEP]:
            if self.config["truncate_as_terminate"]:
                done = True
            self.logger.debug(msg("max step"), extra={"log_once": True})

        # # log data to curriculum manager
        # self.engine.curriculum_manager.log_episode(
        #     done_info[TerminationState.SUCCESS], vehicle.navigation.route_completion
        # )

        return done, done_info

    def cost_function(self):
        vehicle = self.agent
        step_info = dict(num_crash_object=0, num_crash_human=0, num_crash_vehicle=0, num_on_line=0)
        step_info["cost"] = 0
        if self._is_out_of_road(vehicle):
            step_info["cost"] += self.config["out_of_road_cost"]
        if vehicle.crash_vehicle:
            step_info["cost"] += self.config["crash_vehicle_cost"]
            step_info["crash_vehicle_cost"] = self.config["crash_vehicle_cost"]
            step_info["num_crash_vehicle"] = 1
        if vehicle.crash_human:
            step_info["cost"] += self.config["crash_human_cost"]
            step_info["num_crash_human"] = 1
        return step_info['cost'], step_info

    def reward_function(self):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.agent
        step_info = dict()

        # crash penalty
        if vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        if vehicle.crash_human:
            reward = -self.config["crash_human_penalty"]

        step_info["step_reward"] = reward

        # termination reward
        if self._is_arrive_destination():
            reward = self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]

        return reward, step_info

    def _is_arrive_destination(self):
        return self.engine.agent_manager.is_arrive()

    def _is_out_of_road(self, vehicle):
        ego_position = vehicle.position
        ego_position = torch.tensor([ego_position[0], ego_position[1]], dtype=torch.float32)

        ego_poses = self.engine.data_manager.get_current_scenario_data()['ego_poses']
        if isinstance(ego_poses, torch.Tensor):
            expert_positions = ego_poses[:, :2, 3]
        else:
            expert_positions = torch.stack([pose[:2, 3] for pose in ego_poses])

        distances = torch.norm(expert_positions - ego_position.unsqueeze(0), dim=1)
        min_distance = torch.min(distances).item()
        out_of_road_threshold = 5.0
        return min_distance > out_of_road_threshold

    def _reset_global_seed(self, force_seed=None):
        if force_seed is not None:
            current_seed = force_seed
        else:
            current_seed = get_np_random(None).randint(
                self.config["start_scenario_index"],
                self.config["start_scenario_index"] + int(self.config["num_scenarios"])
            )

        set_global_random_seed(current_seed)


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

    def setup_engine(self):
        """Overwrite the data_manager by ScenarioOnlineDataManager"""
        super().setup_engine()
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
        config.update(SCENARIO_WAYPOINT_ENV_CONFIG)
        return config

    def _post_process_config(self, config):
        ret = super(ScenarioWaypointEnv, self)._post_process_config(config)
        assert config["set_static"], "Waypoint policy requires set_static=True"
        return ret


if __name__ == "__main__":
    env = ScenarioEnv(
        {
            "use_render": True,
            "agent_policy": ReplayEgoCarPolicy,
            "manual_control": False,
            "show_interface": True,
            "show_logo": False,
            "show_fps": False,
            # "debug": True,
            # "debug_static_world": True,
            # "no_traffic": True,
            # "no_light": True,
            # "debug":True,
            # "no_traffic":True,
            # "start_scenario_index": 192,
            # "start_scenario_index": 1000,
            "num_scenarios": 3,
            "set_static": True,
            # "force_reuse_object_name": True,
            # "data_directory": "/home/shady/Downloads/test_processed",
            "horizon": 1000,
            "no_static_vehicles": True,
            # "show_policy_mark": True,
            # "show_coordinates": True,
            "vehicle_config": dict(
                show_navi_mark=False,
                no_wheel_friction=True,
                lidar=dict(num_lasers=120, distance=50, num_others=4),
                lane_line_detector=dict(num_lasers=12, distance=50),
                side_detector=dict(num_lasers=160, distance=50)
            ),
            "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),
        }
    )
    success = []
    env.reset(seed=0)
    while True:
        env.reset(seed=env.current_seed + 1)
        for t in range(10000):
            o, r, tm, tc, info = env.step([0, 0])
            assert env.observation_space.contains(o)
            c_lane = env.agent.lane
            long, lat, = c_lane.local_coordinates(env.agent.position)
            # if env.config["use_render"]:
            env.render(
                text={
                    # "obs_shape": len(o),
                    "seed": env.engine.global_seed + env.config["start_scenario_index"],
                    # "reward": r,
                }
                # mode="topdown"
            )

            if (tm or tc) and info["arrive_dest"]:
                print("seed:{}, success".format(env.engine.global_random_seed))
                break
