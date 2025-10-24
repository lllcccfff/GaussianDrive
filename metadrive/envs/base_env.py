import logging
import time
from collections import defaultdict
from typing import Union, Dict, AnyStr, Optional, Tuple, Callable
from collections import OrderedDict
import torch
import gymnasium as gym
import numpy as np
from panda3d.core import PNMImage

from metadrive import constants
from metadrive.constants import DEFAULT_SENSOR_HPR, DEFAULT_SENSOR_OFFSET
from metadrive.constants import RENDER_MODE_NONE, DEFAULT_AGENT
from metadrive.constants import TerminationState, TerrainProperty
from metadrive.utils.logger import get_logger, set_log_level
from metadrive.manager.agent_manager import AgentManager
# from metadrive.manager.record_manager import RecordManager
# from metadrive.manager.replay_manager import ReplayManager
# from metadrive.obs.image_obs import ImageStateObservation
from metadrive.obs.observation_base import BaseObservation
from metadrive.obs.gaussian_obs import GaussianStateObservation
from metadrive.obs.observation_base import DummyObservation
# from metadrive.obs.state_obs import LidarStateObservation
from metadrive.scenario.utils import convert_recorded_scenario_exported
from metadrive.utils import merge_dicts, get_np_random, concat_step_infos
from metadrive.utils.logger import get_logger, reset_logger
from metadrive.engine.core.physics_world import PhysicsWorld
from metadrive.engine.step_counter import StepCounter
from metadrive.engine.core.collision_callback import collision_callback
from panda3d.core import AntialiasAttrib, loadPrcFileData, LineSegs, PythonCallbackObject, Vec3, NodePath
from metadrive.version import VERSION
from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.vehicle.vehicle_type import get_vehicle_type
from metadrive.manager.scenario_data_manager import ScenarioDataManager, ScenarioOnlineDataManager
from metadrive.manager.scenario_map_manager import ScenarioMapManager

from easydrive.engine.config import Config
from metadrive.default_config import BASE_DEFAULT_CONFIG

class BaseEnv(gym.Env):
    # Force to use this seed if necessary. Note that the recipient of the forced seed should be explicitly implemented.

    @classmethod
    def default_config(cls) -> Config:
        return Config(BASE_DEFAULT_CONFIG)

    def __init__(self, model, config: Config = None):
        
        if config is None:
            config = Config()
        default_config = self.default_config()
        default_config.merge_from(config, replace_keys=["agent_configs"])

        self.logger = get_logger()
        # set_log_level(config.get("log_level", logging.DEBUG if config.get("debug", False) else logging.INFO))

        # In MARL envs with respawn mechanism, varying episode lengths might happen.
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)

        # press p to stop
        self.in_stop = False

        # scenarios
        self.start_index = 0

        self.model = model

        self.setup(default_config)

    # def _post_process_config(self, config):
    #     """Add more special process to merged config"""
    #     # Cancel interface panel
    #     self.logger.info("Environment: {}".format(self.__class__.__name__))
    #     self.logger.info("MetaDrive version: {}".format(VERSION))

    #     # show sensor lists
    #     if config["truncate_as_terminate"]:
    #         self.logger.warning(
    #             "When reaching max steps, both 'terminate' and 'truncate will be True."
    #             "Generally, only the `truncate` should be `True`."
    #         )
    #     return config


    def setup(self, config):
        """
        Engine setting after launching
        """
        self._register_manager("data_manager", ScenarioDataManager(config, self.model.load_metadata))
        self._register_manager("map_manager", ScenarioMapManager(self.config['map_config'], self.model.load_model))
        self._register_manager("step_manager", StepCounter(self.config['physics_world_step_size'],))

        # self._register_manager("record_manager", RecordManager())
        # self._register_manager("replay_manager", ReplayManager())

        # physics world
        self.physics_world = PhysicsWorld(disable_collision=self.config["disable_collision"])

        # collision callback
        self.physics_world.dynamic_world.setContactAddedCallback(PythonCallbackObject(collision_callback))

        self.agent_managers = {}
        self.agent_managers['actor'] = self._init_agent_manager()


    @property
    def config(self):
        if hasattr(self.data_manager, "current_config"):
            return self.data_manager.current_config
        else:
            return self.data_manager.base_config

    def _init_agent_manager(self):
        raise NotImplementedError

    def _register_manager(self, manager_name: str, manager):
        """
        Add a manager to BaseEnv, then all objects can communicate with this class
        :param manager_name: name shouldn't exist in self.managers and not be same as any class attribute
        :param manager: subclass of BaseManager
        """
        # assert manager_name not in self.managers, "Manager {} already exists in BaseEnv, Use update_manager() to " \
        #                                            "overwrite".format(manager_name)
        # assert not hasattr(self, manager_name), "Manager name can not be same as the attribute in BaseEnv"
        # self.managers[manager_name] = manager
        setattr(self, manager_name, manager)

    def reset(self, seed: Union[None, int] = None):
        # Update record replay
        self.replay_episode = True if self.config["replay_episode"] is not None else False
        self.record_episode = self.config["record_episode"]
        self.only_reset_when_replay = self.config["only_reset_when_replay"]
        """
        Reset the env, scene can be restored and replayed by giving episode_data
        Reset the environment or load an episode from episode data to recover is
        :param seed: The seed to set the env. It is actually the scenario index you intend to choose
        :return: None
        """
        # reset_logger()
        # if self.logger is None:
        #     self.logger = get_logger()
        #     log_level = self.config.get("log_level", logging.DEBUG if self.config.get("debug", False) else logging.INFO)
        #     set_log_level(log_level)

        self.dones = False
        self.episode_rewards = 0
        self.episode_lengths = 0

        self._reset_global_seed(seed)

        # reset manager
        for manager in [self.map_manager] + list(self.agent_managers.values()):
            manager.clear_all_objects()
        self._object_clean_check()
        
        all_agent = list(self.agent_managers.keys())
        for n in all_agent:
            if n != 'actor':
                self.agent_managers[n].destroy()
                self.agent_managers.pop(n)

        self.data_manager.reset()

        scenario_data = self.data_manager.get_current_scenario_data()
        self.step_manager.reset(**scenario_data)
        self.map_manager.reset(config=self.config['map_config'], physics_world=self.physics_world, **scenario_data)
        self._update_participants(scenario_data)

        self._update_scene()

        step_infos = {}
        for mgr_n, manager in self.agent_managers.items() :
            new_step_infos = manager.observe()
            step_infos[mgr_n] = new_step_infos

        return self._get_reset_return(step_infos)

    def _reset_global_seed(self, force_seed=None):
        if force_seed is not None:
            current_seed = force_seed
        else:
            current_seed = get_np_random(None).randint(0, 0xffffffff)
        self.current_seed = current_seed
        for mgr in [self.data_manager, self.map_manager] + list(self.agent_managers.values()):
            mgr.seed(current_seed)

    def _object_clean_check(self):
        # rigid body check
        bodies = []
        for world in [self.physics_world.dynamic_world, self.physics_world.static_world]:
            bodies += world.getRigidBodies()
            bodies += world.getSoftBodies()
            bodies += world.getGhosts()
            bodies += world.getVehicles()
            bodies += world.getCharacters()
            # bodies += world.getManifolds()

        filtered = []
        for body in bodies:
            # if body.getName() in ["detector_mask", "debug"]:
            #     continue
            filtered.append(body)
        assert len(filtered) == 0, "Physics Bodies should be cleaned before manager.reset() is called. " \
                                   "Uncleared bodies: {}".format(filtered)

    def _update_participants(self, scenario_data):
        camera_params = scenario_data['camera_params']
        for name, init_state in scenario_data['init_state'].items():
            if name != 'actor':
                tracking = scenario_data['participants'][name]
                cfg = self.config['participant_config'].copy()
                cfg['controller_config']['size'] = tracking['size']
                if tracking['type'] == 'vehicle':
                    cfg['controller'] = get_vehicle_type(tracking['size'][1], False)
                elif tracking['type'] == 'pedestrian':
                    cfg['controller'] = Pedestrian
                elif tracking['type'] == 'cyclist':
                    cfg['controller'] = Cyclist
                
                self.agent_managers[name] = AgentManager(cfg, self.step_manager)
            else:
                tracking = scenario_data['ego_poses']
            self.agent_managers[name].reset(
                config=self.config['actor_config'] if name == 'actor' else cfg,
                physics_world=self.physics_world,
                render_fn=self.model.render,
                camera_params=camera_params,
                init_state=init_state,
                state=scenario_data['agent_state'][name],
                timestamp_range=scenario_data['timestamp_range']
            )

    def _get_reset_return(self, reset_info):
        # TODO: figure out how to get the information of the before step
        obses = {}
        done_infos = {}
        cost_infos = {}
        reward_infos = {}
        obses = reset_info['actor']['observation']
        _, reward_infos = self.reward_function()
        _, done_infos = self.done_function()
        _, cost_infos = self.cost_function()

        step_infos = concat_step_infos([reset_info, done_infos, reward_infos, cost_infos])

        return obses, step_infos

    # ===== Run-time =====
    def step(self, actions: Union[Union[np.ndarray, list], Dict[AnyStr, Union[list, np.ndarray]], int]):
        self.step_manager.step()
        
        # prepare for stepping the simulation
        before_step_infos = {}

        for i in range(self.config["decision_repeat"]):
            # simulate or replay
            for manager in self.agent_managers.values():
                manager.step(actions)

            self.step_physics_world()
            # the recording should happen after step physics world
            # if "record_manager" in self.managers and i < self.config["decision_repeat"] - 1:
            #     self.record_manager.step()

        # to get new pose and update gaussian model
        self._update_scene()

        after_step_infos = {}
        for mgr_n, manager in self.agent_managers.items() :
            new_step_infos = manager.observe()
            after_step_infos[mgr_n] = new_step_infos

        # Note that we use shallow update for info dict in this function! This will accelerate system.
        engine_info = merge_dicts(
            after_step_infos, before_step_infos, allow_new_keys=True, without_copy=True
        )

        return self._get_step_return(actions, engine_info=engine_info)  # collect observation, reward, termination

    def _update_scene(self):
        new_object_poses = {}
        for name, mgr in self.agent_managers.items():
            if name == 'actor': continue
            if mgr.is_spawned and not mgr.is_arrive:
                new_object_poses[name] = torch.from_numpy(mgr.get_pose())
        self.model.update_scene(self.step_manager.current_timestamp, new_object_poses)


    def step_physics_world(self):
        dt = self.config["physics_world_step_size"] * 1e-6
        self.physics_world.dynamic_world.doPhysics(dt, 1, dt)

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
        obses = engine_info['actor']['observation']

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

    def reward_function(self, object_id: str) -> Tuple[float, Dict]:
        raise NotImplementedError

    def cost_function(self, object_id: str) -> Tuple[float, Dict]:
        raise NotImplementedError

    def done_function(self, object_id: str) -> Tuple[bool, Dict]:
        raise NotImplementedError
    
    def close(self):
        raise NotImplementedError

    def capture(self, file_name=None):
        if not hasattr(self, "_capture_img"):
            self._capture_img = PNMImage()
        self.engine.win.getScreenshot(self._capture_img)
        if file_name is None:
            file_name = "main_index_{}_step_{}_{}.png".format(self.current_seed, self.engine.episode_step, time.time())
        self._capture_img.write(file_name)
        self.logger.info("Image is saved at: {}".format(file_name))

    @property
    def actor_controller(self):
        return self.agent_managers['actor'].controller


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
        ret = self.actor_manager.get_observation_spaces()
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
        ret = self.actor_manager.get_action_spaces()
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

if __name__ == '__main__':
    cfg = {"use_render": True}
    env = BaseEnv(cfg)
    env.reset()
    while True:
        env.step(env.action_space.sample())
