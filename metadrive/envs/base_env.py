import logging
import time
from collections import defaultdict
from typing import Union, Dict, AnyStr, Optional, Tuple, Callable

import gymnasium as gym
import numpy as np
from panda3d.core import PNMImage

from metadrive import constants
from metadrive.constants import DEFAULT_SENSOR_HPR, DEFAULT_SENSOR_OFFSET
from metadrive.constants import RENDER_MODE_NONE, DEFAULT_AGENT
from metadrive.constants import TerminationState, TerrainProperty
from metadrive.engine.engine_utils import initialize_engine, close_engine, \
    engine_initialized, set_global_random_seed, initialize_global_config, get_global_config
from metadrive.engine.logger import get_logger, set_log_level
from metadrive.manager.agent_manager import VehicleAgentManager
# from metadrive.manager.record_manager import RecordManager
# from metadrive.manager.replay_manager import ReplayManager
# from metadrive.obs.image_obs import ImageStateObservation
from metadrive.obs.observation_base import BaseObservation
from metadrive.obs.gaussian_obs import GaussianStateObservation
from metadrive.obs.observation_base import DummyObservation
# from metadrive.obs.state_obs import LidarStateObservation
from metadrive.scenario.utils import convert_recorded_scenario_exported
from metadrive.utils import merge_dicts, get_np_random, concat_step_infos
from metadrive.version import VERSION

from easydrive.engine.config import Config
from metadrive.default_config import BASE_DEFAULT_CONFIG

class BaseEnv(gym.Env):
    # Force to use this seed if necessary. Note that the recipient of the forced seed should be explicitly implemented.
    _DEBUG_RANDOM_SEED: Union[int, None] = None

    @classmethod
    def default_config(cls) -> Config:
        return Config(BASE_DEFAULT_CONFIG)

    # ===== Intialization =====
    def __init__(self, config: Config = None):
        if config is None:
            config = Config()
        self.logger = get_logger()
        set_log_level(config.get("log_level", logging.DEBUG if config.get("debug", False) else logging.INFO))
        default_config = self.default_config()
        default_config.merge_from(config, replace_keys=["agent_configs"])
        global_config = self._post_process_config(default_config)
        self.config = global_config
        initialize_global_config(self.config)

        # observation and action space
        self.agent_manager = self._init_agent_manager()

        # lazy initialization, create the main simulation in the lazy_init() func
        # self.engine: Optional[BaseEngine] = None

        # In MARL envs with respawn mechanism, varying episode lengths might happen.
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

        # press p to stop
        self.in_stop = False

        # scenarios
        self.start_index = 0

    def _post_process_config(self, config):
        """Add more special process to merged config"""
        # Cancel interface panel
        self.logger.info("Environment: {}".format(self.__class__.__name__))
        self.logger.info("MetaDrive version: {}".format(VERSION))

        # Adjust terrain
        # n = config["map_region_size"]
        # assert (n & (n - 1)) == 0 and 512 <= n <= 4096, "map_region_size should be pow of 2 and < 2048."
        # TerrainProperty.map_region_size = config["map_region_size"]

        # Multi-Thread
        # if config["image_on_cuda"]:
        #     self.logger.info("Turn Off Multi-thread rendering due to image_on_cuda=True")
        #     config["multi_thread_render"] = False

        # # Optimize sensor creation in none-screen mode
        # if not config["use_render"] and not config["image_observation"]:
        #     filtered = {}
        #     for id, cfg in config["sensors"].items():
        #         if len(cfg) > 0 and not issubclass(cfg[0], BaseCamera) and id != "main_camera":
        #             filtered[id] = cfg
        #     config["sensors"] = filtered
        #     config["interface_panel"] = []

        # # Check sensor existence
        # if config["use_render"] or "main_camera" in config["sensors"]:
        #     config["sensors"]["main_camera"] = ("MainCamera", *config["window_size"])

        # # Merge dashboard config with sensors
        # to_use = []
        # if not config["render_pipeline"] and config["show_interface"] and "main_camera" in config["sensors"]:
        #     for panel in config["interface_panel"]:
        #         if panel == "dashboard":
        #             config["sensors"]["dashboard"] = (DashBoard, )
        #         if panel not in config["sensors"]:
        #             self.logger.warning(
        #                 "Fail to add sensor: {} to the interface. Remove it from panel list!".format(panel)
        #             )
        #         elif panel == "main_camera":
        #             self.logger.warning("main_camera can not be added to interface_panel, remove")
        #         else:
        #             to_use.append(panel)
        # config["interface_panel"] = to_use


        # show sensor lists
        if config["truncate_as_terminate"]:
            self.logger.warning(
                "When reaching max steps, both 'terminate' and 'truncate will be True."
                "Generally, only the `truncate` should be `True`."
            )
        return config

    def _get_observations(self) -> Dict[str, "BaseObservation"]:
        return {DEFAULT_AGENT: self.get_single_observation()}

    def _get_agent_manager(self):
        return VehicleAgentManager(init_observations=self._get_observations())

    def lazy_init(self):
        """
        Only init once in runtime, variable here exists till the close_env is called
        :return: None
        """
        # It is the true init() func to create the main vehicle and its module, to avoid incompatible with ray
        initialize_engine(self.config)
        # engine setup
        self.setup_engine()
        # other optional initialization
        self._after_lazy_init()
        # self.logger.info(
        #     "Start Scenario Index: {}, Num Scenarios : {}".format(
        #         self.engine.data_manager.current_scenario_id, self.config["num_scenarios"]
        #     )
        # )

    @property
    def engine(self):
        from metadrive.engine.engine_utils import get_engine
        return get_engine()

    def _after_lazy_init(self):
        pass

    # ===== Run-time =====
    def step(self, actions: Union[Union[np.ndarray, list], Dict[AnyStr, Union[list, np.ndarray]], int]):
        engine_info = self._step_simulator(actions)  # step the simulation
        while self.in_stop:
            self.engine.taskMgr.step()  # pause simulation
        return self._get_step_return(actions, engine_info=engine_info)  # collect observation, reward, termination


    def _step_simulator(self, actions):
        # prepare for stepping the simulation
        scene_manager_before_step_infos = self.engine.before_step(actions)
        # step all entities and the simulator
        self.engine.step(self.config["decision_repeat"])
        # update states, if restore from episode data, position and heading will be force set in update_state() function
        scene_manager_after_step_infos = self.engine.after_step()

        # Note that we use shallow update for info dict in this function! This will accelerate system.
        return merge_dicts(
            scene_manager_after_step_infos, scene_manager_before_step_infos, allow_new_keys=True, without_copy=True
        )

    def reward_function(self, object_id: str) -> Tuple[float, Dict]:
        """
        Override this func to get a new reward function
        :param object_id: name of this object
        :return: reward, reward info
        """
        self.logger.warning("Reward function is not implemented. Return reward = 0", extra={"log_once": True})
        return 0, {}

    def cost_function(self, object_id: str) -> Tuple[float, Dict]:
        self.logger.warning("Cost function is not implemented. Return cost = 0", extra={"log_once": True})
        return 0, {}

    def done_function(self, object_id: str) -> Tuple[bool, Dict]:
        self.logger.warning("Done function is not implemented. Return Done = False", extra={"log_once": True})
        return False, {}


    def reset(self, seed: Union[None, int] = None):
        """
        Reset the env, scene can be restored and replayed by giving episode_data
        Reset the environment or load an episode from episode data to recover is
        :param seed: The seed to set the env. It is actually the scenario index you intend to choose
        :return: None
        """
        if self.logger is None:
            self.logger = get_logger()
            log_level = self.config.get("log_level", logging.DEBUG if self.config.get("debug", False) else logging.INFO)
            set_log_level(log_level)

        if not engine_initialized():
            self.lazy_init()  # it only works the first time when reset() is called to avoid the error when render

        self._reset_global_seed(seed)
        if self.engine is None:
            raise ValueError(
                "Current MetaDrive instance is broken. Please make sure there is only one active MetaDrive "
                "environment exists in one process. You can try to call env.close() and then call "
                "env.reset() to rescue this environment. However, a better and safer solution is to check the "
                "singleton of MetaDrive and restart your program."
            )
        reset_info = self.engine.reset()
        # render the scene
        # self.engine.taskMgr.step()
        # if self.top_down_renderer is not None:
        #     self.top_down_renderer.clear()
        #     self.engine.top_down_renderer = None

        self.dones = False
        self.episode_rewards = 0
        self.episode_lengths = 0

        assert self.config is self.engine.global_config is get_global_config(), "Inconsistent config may bring errors!"
        return self._get_reset_return(reset_info)

    def _get_reset_return(self, reset_info):
        # TODO: figure out how to get the information of the before step

        obses = {}
        done_infos = {}
        cost_infos = {}
        reward_infos = {}
        engine_info = reset_info
        obses = self.agent_manager.get_observations()
        _, reward_infos = self.reward_function()
        _, done_infos = self.done_function()
        _, cost_infos = self.cost_function()

        step_infos = concat_step_infos([engine_info, done_infos, reward_infos, cost_infos])

        return obses, step_infos

    def _wrap_info_as_single_agent(self, data):
        """
        Wrap to single agent info
        """
        agent_info = data.pop(next(iter(self.agents.keys())))
        data.update(agent_info)
        return data

    def _get_step_return(self, actions, engine_info):
        # update obs, dones, rewards, costs, calculate done at first !
        obses = {}
        done_infos = {}
        cost_infos = {}
        reward_infos = {}
        rewards = {}

        self.episode_lengths += 1
        rewards, reward_infos = self.reward_function()
        self.episode_rewards += rewards
        done_function_result, done_infos = self.done_function()
        _, cost_infos = self.cost_function()
        self.dones = done_function_result or self.dones
        obses = self.agent_manager.get_observations()

        step_infos = concat_step_infos([engine_info, done_infos, reward_infos, cost_infos])
        truncateds = step_infos.get(TerminationState.MAX_STEP, False)
        terminateds = self.dones

        # For extreme scenario only. Force to terminate all agents if the environmental step exceeds 5 times horizon.
        if self.config["horizon"] and self.episode_step > 5 * self.config["horizon"]:
            for k in truncateds:
                truncateds[k] = True
                if self.config["truncate_as_terminate"]:
                    self.dones[k] = terminateds[k] = True

        step_infos["episode_reward"] = self.episode_rewards
        step_infos["episode_length"] = self.episode_lengths

        return obses, rewards, terminateds, truncateds, step_infos

    def close(self):
        if self.engine is not None:
            close_engine()

    def force_close(self):
        print("Closing environment ... Please wait")
        self.close()
        time.sleep(2)  # Sleep two seconds
        raise KeyboardInterrupt("'Esc' is pressed. MetaDrive exits now.")

    def capture(self, file_name=None):
        if not hasattr(self, "_capture_img"):
            self._capture_img = PNMImage()
        self.engine.win.getScreenshot(self._capture_img)
        if file_name is None:
            file_name = "main_index_{}_step_{}_{}.png".format(self.current_seed, self.engine.episode_step, time.time())
        self._capture_img.write(file_name)
        self.logger.info("Image is saved at: {}".format(file_name))

    def get_single_observation(self):
        """
        Get the observation for one object
        """
        if self.__class__ is BaseEnv:
            o = DummyObservation()
        else:
            if self.config["agent_observation"]:
                o = self.config["agent_observation"](self.config)
            else:
                o = GaussianStateObservation(self.config)
        return o

    def _wrap_as_single_agent(self, data):
        return data[next(iter(self.agents.keys()))]

    @property
    def current_seed(self):
        return self.engine.global_random_seed

    @property
    def num_scenarios(self):
        return self.config["num_scenarios"]

    @property
    def observations(self):
        """
        Return observations of active and controllable agents
        :return: Dict
        """
        return self

    @property
    def observation_space(self) -> gym.Space:
        """
        Return observation spaces of active and controllable agents
        :return: Dict
        """
        ret = self.agent_manager.get_observation_spaces()
        if not self.is_multi_agent:
            return next(iter(ret.values()))
        else:
            return gym.spaces.Dict(ret)

    @property
    def action_space(self) -> gym.Space:
        """
        Return action spaces of active and controllable agents. Generally, it is defined in AgentManager. But you can
        still overwrite this function to define the action space for the environment.
        :return: Dict
        """
        ret = self.agent_manager.get_action_spaces()
        if not self.is_multi_agent:
            return next(iter(ret.values()))
        else:
            return gym.spaces.Dict(ret)

    # @property
    # def vehicles(self):
    #     """
    #     Return all active vehicles
    #     :return: Dict[agent_id:vehicle]
    #     """
    #     self.logger.warning("env.vehicles will be deprecated soon. Use env.agents instead", extra={"log_once": True})
    #     return self.agents

    # @property
    # def vehicle(self):
    #     self.logger.warning("env.vehicle will be deprecated soon. Use env.agent instead", extra={"log_once": True})
    #     return self.agent

    @property
    def agent_object(self):
        """
        Return all active agents
        :return: Dict[agent_id:agent]
        """
        return self.agent_manager._agent_object


    @property
    def agents_including_just_terminated(self):
        """
        Return all agents that occupy some space in current environments
        :return: Dict[agent_id:vehicle]
        """
        ret = self.agent_manager.active_agents
        ret.update(self.agent_manager.just_terminated_agents)
        return ret

    def setup_engine(self):
        """
        Engine setting after launching
        """
        # self.engine.accept("r", self.reset)
        # self.engine.accept("c", self.capture)
        # self.engine.accept("p", self.stop)
        # self.engine.accept("b", self.switch_to_top_down_view)
        # self.engine.accept("q", self.switch_to_third_person_view)
        # self.engine.accept("]", self.next_seed_reset)
        # self.engine.accept("[", self.last_seed_reset)
        self.engine.register_manager("agent_manager", self.agent_manager)
        # self.engine.register_manager("record_manager", RecordManager())
        # self.engine.register_manager("replay_manager", ReplayManager())

    @property
    def current_map(self):
        return self.engine.current_map

    @property
    def maps(self):
        return self.engine.map_manager.maps

    def _render_topdown(self, text, *args, **kwargs):
        return self.engine.render_topdown(text, *args, **kwargs)

    @property
    def main_camera(self):
        return self.engine.main_camera

    @property
    def current_track_agent(self):
        return self.engine.current_track_agent

    @property
    def top_down_renderer(self):
        return self.engine.top_down_renderer

    @property
    def episode_step(self):
        return self.engine.episode_step if self.engine is not None else 0

    def export_scenarios(
        self,
        policies: Union[dict, Callable],
        scenario_index: Union[list, int],
        max_episode_length=None,
        verbose=False,
        suppress_warning=False,
        render_topdown=False,
        return_done_info=True,
        to_dict=True
    ):
        """
        We export scenarios into a unified format with 10hz sample rate
        """
        def _act(observation):
            if isinstance(policies, dict):
                ret = {}
                for id, o in observation.items():
                    ret[id] = policies[id](o)
            else:
                ret = policies(observation)
            return ret

        if self.is_multi_agent:
            assert isinstance(policies, dict), "In MARL setting, policies should be mapped to agents according to id"
        else:
            assert isinstance(policies, Callable), "In single agent case, policy should be a callable object, taking" \
                                                   "observation as input."
        scenarios_to_export = dict()
        if isinstance(scenario_index, int):
            scenario_index = [scenario_index]
        self.config["record_episode"] = True
        done_info = {}
        for index in scenario_index:
            obs = self.reset(seed=index)
            done = False
            count = 0
            info = None
            while not done:
                obs, reward, terminated, truncated, info = self.step(_act(obs))
                done = terminated or truncated
                count += 1
                if max_episode_length is not None and count > max_episode_length:
                    done = True
                    info[TerminationState.MAX_STEP] = True
                if count > 10000 and not suppress_warning:
                    self.logger.warning(
                        "Episode length is too long! If this behavior is intended, "
                        "set suppress_warning=True to disable this message"
                    )
                if render_topdown:
                    self.render("topdown")
            episode = self.engine.dump_episode()
            if verbose:
                self.logger.info("Finish scenario {} with {} steps.".format(index, count))
            scenarios_to_export[index] = convert_recorded_scenario_exported(episode, to_dict=to_dict)
            done_info[index] = info
        self.config["record_episode"] = False
        if return_done_info:
            return scenarios_to_export, done_info
        else:
            return scenarios_to_export

    def stop(self):
        self.in_stop = not self.in_stop

    def switch_to_top_down_view(self):
        self.main_camera.stop_track()

    def switch_to_third_person_view(self):
        if self.main_camera is None:
            return
        self.main_camera.reset()
        if self.config["prefer_track_agent"] is not None and self.config["prefer_track_agent"] in self.agents.keys():
            new_v = self.agents[self.config["prefer_track_agent"]]
            current_track_agent = new_v
        else:
            if self.main_camera.is_bird_view_camera():
                current_track_agent = self.current_track_agent
            else:
                agents = list(self.engine.agents.values())
                if len(agents) <= 1:
                    return
                if self.current_track_agent in agents:
                    agents.remove(self.current_track_agent)
                new_v = get_np_random().choice(agents)
                current_track_agent = new_v
        self.main_camera.track(current_track_agent)
        for name, sensor in self.engine.sensors.items():
            if hasattr(sensor, "track") and name != "main_camera":
                sensor.track(current_track_agent.origin, constants.DEFAULT_SENSOR_OFFSET, DEFAULT_SENSOR_HPR)
        return

    def next_seed_reset(self):
        if self.current_seed + 1 < self.start_index + self.num_scenarios:
            self.reset(self.current_seed + 1)
        else:
            self.logger.warning(
                "Can't load next scenario! Current seed is already the max scenario index."
                "Allowed index: {}-{}".format(self.start_index, self.start_index + self.num_scenarios - 1)
            )

    def last_seed_reset(self):
        if self.current_seed - 1 >= self.start_index:
            self.reset(self.current_seed - 1)
        else:
            self.logger.warning(
                "Can't load last scenario! Current seed is already the min scenario index"
                "Allowed index: {}-{}".format(self.start_index, self.start_index + self.num_scenarios - 1)
            )

    def _reset_global_seed(self, force_seed=None):
        current_seed = force_seed if force_seed is not None else \
            get_np_random(self._DEBUG_RANDOM_SEED).randint(self.start_index, self.start_index + self.num_scenarios)
        assert self.start_index <= current_seed < self.start_index + self.num_scenarios, \
            "scenario_index (seed) should be in [{}:{})".format(self.start_index, self.start_index + self.num_scenarios)
        self.seed(current_seed)


if __name__ == '__main__':
    cfg = {"use_render": True}
    env = BaseEnv(cfg)
    env.reset()
    while True:
        env.step(env.action_space.sample())
