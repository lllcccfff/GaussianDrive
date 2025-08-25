import os
import pickle
import time
from collections import OrderedDict
from typing import Callable, Optional, Union, List, Dict, AnyStr

import numpy as np

from metadrive.base_class.randomizable import Randomizable
from metadrive.constants import RENDER_MODE_NONE
from metadrive.engine.core.engine_core import EngineCore
from metadrive.engine.interface import Interface
from metadrive.engine.logger import get_logger, reset_logger

from metadrive.utils import concat_step_infos

logger = get_logger()


def generate_distinct_rgb_values():
    # Try to avoid (0,0,0) and (255,255,255) to avoid confusion with the background and other objects.
    r = np.linspace(16, 256 - 16, 16).astype(int)
    g = np.linspace(16, 256 - 16, 16).astype(int)
    b = np.linspace(16, 256 - 16, 16).astype(int)

    # Create a meshgrid and reshape to get all combinations of r, g, b
    rgbs = np.array(np.meshgrid(r, g, b)).T.reshape(-1, 3)

    # Normalize the values to be between 0 and 1
    rgbs = rgbs / 255.0

    return tuple(tuple(round(vv, 5) for vv in v) for v in rgbs)


COLOR_SPACE = generate_distinct_rgb_values()


class BaseEngine(EngineCore, Randomizable):
    """
    Due to the feature of Panda3D, BaseEngine should only be created once(Singleton Pattern)
    It is a pure game engine, which is not task-specific, while BaseEngine connects the
    driving task and the game engine modified from Panda3D Engine.
    """
    singleton = None
    global_random_seed = None

    MAX_COLOR = len(COLOR_SPACE)
    COLORS_OCCUPIED = set()
    COLORS_FREE = set(COLOR_SPACE)

    def __init__(self, global_config):
        self.c_id = dict()
        self.id_c = dict()

        EngineCore.__init__(self, global_config)
        Randomizable.__init__(self, self.global_random_seed)
        self.episode_step = 0
        BaseEngine.singleton = self

        # managers
        self._managers = OrderedDict()

        # for recovering, they can not exist together
        self.record_episode = False
        self.replay_episode = False
        self.only_reset_when_replay = False
        # self.accept("s", self._stop_replay)

        self._spawned_objects = dict()
        self._object_policies = dict()
        self._object_tasks = dict()

        # the clear function is a fake clear, objects cleared is stored for future use
        self._dying_objects = dict()

        # topdown renderer
        self.top_down_renderer = None

        # warm up
        self.warmup()

        # curriculum reset
        self._max_level = self.global_config.get("curriculum_level", 1)
        self._current_level = 0
        self._num_scenarios_per_level = int(self.global_config.get("num_scenarios", 1) / self._max_level)

    def render(self, **kwargs):
        rendered = self.managers['map_manager'].model.render(**kwargs)
        return rendered

    def reset(self):
        """
        Clear and generate the whole scene
        """
        # reset logger
        reset_logger()
        step_infos = {}

        # initialize
        self._episode_start_time = time.time()
        self.episode_step = 0
        # if self.global_config["debug_physics_world"]:
        #     self.addTask(self.report_body_nums, "report_num")

        # Update record replay
        self.replay_episode = True if self.global_config["replay_episode"] is not None else False
        self.record_episode = self.global_config["record_episode"]
        self.only_reset_when_replay = self.global_config["only_reset_when_replay"]

        # _debug_memory_usage = False

        # if _debug_memory_usage:

        #     def process_memory():
        #         import psutil
        #         import os
        #         process = psutil.Process(os.getpid())
        #         mem_info = process.memory_info()
        #         return mem_info.rss

        #     cm = process_memory()

        # reset manager
        for manager_name, manager in self._managers.items():
            # clean all manager
            new_step_infos = manager.before_reset()
            step_infos = concat_step_infos([step_infos, new_step_infos])
            # if _debug_memory_usage:
            #     lm = process_memory()
            #     if lm - cm != 0:
            #         print("{}: Before Reset! Mem Change {:.3f}MB".format(manager_name, (lm - cm) / 1e6))
            #     cm = lm
        self._object_clean_check()

        for manager_name, manager in self.managers.items():
            if self.replay_episode and self.only_reset_when_replay and manager is not self.replay_manager:
                # The scene will be generated from replay manager in only reset replay mode
                continue
            new_step_infos = manager.reset()
            step_infos = concat_step_infos([step_infos, new_step_infos])

            # if _debug_memory_usage:
            #     lm = process_memory()
            #     if lm - cm != 0:
            #         print("{}: Reset! Mem Change {:.3f}MB".format(manager_name, (lm - cm) / 1e6))
            #     cm = lm

        for manager_name, manager in self.managers.items():
            new_step_infos = manager.after_reset()
            step_infos = concat_step_infos([step_infos, new_step_infos])

            # if _debug_memory_usage:
            #     lm = process_memory()
            #     if lm - cm != 0:
            #         print("{}: After Reset! Mem Change {:.3f}MB".format(manager_name, (lm - cm) / 1e6))
            #     cm = lm

        # reset terrain
        # center_p = self.current_map.get_center_point() if isinstance(self.current_map, PGMap) else [0, 0]
        # center_p = [0, 0]
        # self.terrain.reset(center_p)


        # reset colors
        BaseEngine.COLORS_FREE = set(COLOR_SPACE)
        BaseEngine.COLORS_OCCUPIED = set() 
        # new_i2c = {}
        # new_c2i = {}
        # # print("rest objects", len(self.get_objects()))
        # for object in self.get_objects().values():
        #     if object.id in self.id_c.keys():
        #         id = object.id
        #         color = self.id_c[object.id]
        #         BaseEngine.COLORS_OCCUPIED.add(color)
        #         BaseEngine.COLORS_FREE.remove(color)
        #         new_i2c[id] = color
        #         new_c2i[color] = id
        # # print(len(BaseEngine.COLORS_FREE), len(BaseEngine.COLORS_OCCUPIED))
        # self.c_id = new_c2i
        # self.id_c = new_i2c
        return step_infos

    def before_step(self, action: np.array):
        """
        Entities make decision here, and prepare for step
        All entities can access this global manager to query or interact with others
        :param external_actions: Dict[agent_id:action]
        :return:
        """
        self.episode_step += 1
        step_infos = {}
        self.cur_action = action
        for manager in self.managers.values():
            new_step_infos = manager.before_step()
            step_infos = concat_step_infos([step_infos, new_step_infos])
        return step_infos

    def step(self, step_num: int = 1) -> None:
        """
        Step the dynamics of each entity on the road.
        :param step_num: Decision of all entities will repeat *step_num* times
        """
        for i in range(step_num):
            # simulate or replay
            for name, manager in self.managers.items():
                if name != "record_manager":
                    manager.step()
            self.step_physics_world()
            # the recording should happen after step physics world
            if "record_manager" in self.managers and i < step_num - 1:
                # last recording should be finished in after_step(), as some objects may be created in after_step.
                # We repeat run simulator ```step_num``` frames, and record after each frame.
                # The recording of last frame is actually finished when all managers finish the ```after_step()```
                # function. So the recording for the last time should be done after that.
                # An example is that in ```PGTrafficManager``` we may create new vehicles in
                # ```after_step()``` of the traffic manager. Therefore, we can't record the frame before that.
                # These new cars' states can be recorded only if we run ```record_managers.step()```
                # after the creation of new cars and then can be recorded in ```record_managers.after_step()```
                self.record_manager.step()

    def after_step(self, *args, **kwargs) -> Dict:
        """
        Update states after finishing movement
        :return: if this episode is done
        """

        step_infos = {}
        if self.record_episode:
            assert list(self.managers.keys())[-1] == "record_manager", "Record Manager should have lowest priority"
        for manager in self.managers.values():
            new_step_info = manager.after_step(*args, **kwargs)
            step_infos = concat_step_infos([step_infos, new_step_info])
        
        return step_infos

    def dump_episode(self, pkl_file_name=None) -> None:
        """Dump the data of an episode."""
        assert self.record_manager is not None
        episode_state = self.record_manager.get_episode_metadata()
        if pkl_file_name is not None:
            with open(pkl_file_name, "wb+") as file:
                pickle.dump(episode_state, file)
        return episode_state

    def close(self):
        """
        Note:
        Instead of calling this func directly, close Engine by using engine_utils.close_engine
        """
        if len(self._managers) > 0:
            for name, manager in self._managers.items():
                setattr(self, name, None)
                if manager is not None:
                    manager.destroy()
        # clear all objects in spawned_object
        # self.clear_objects([id for id in self._spawned_objects.keys()])
        for id, obj in self._spawned_objects.items():
            if id in self._object_policies:
                self._object_policies.pop(id).destroy()
            if id in self._object_tasks:
                self._object_tasks.pop(id).destroy()
            self._clean_color(obj.id)
            obj.destroy()
        for cls, pending_obj in self._dying_objects.items():
            for obj in pending_obj:
                self._clean_color(obj.id)
                obj.destroy()
        self._dying_objects = {}
        if self.main_camera is not None:
            self.main_camera.destroy()
        self.interface.destroy()
        self.close_engine()

        if self.top_down_renderer is not None:
            self.top_down_renderer.close()
            del self.top_down_renderer
            self.top_down_renderer = None

        Randomizable.destroy(self)

    def __del__(self):
        logger.debug("{} is destroyed".format(self.__class__.__name__))

    # def _stop_replay(self):
    #     raise DeprecationWarning
    #     if not self.IN_REPLAY:
    #         return
    #     self.STOP_REPLAY = not self.STOP_REPLAY

    # def register_manager(self, manager_name: str, manager):
    #     """
    #     Add a manager to BaseEngine, then all objects can communicate with this class
    #     :param manager_name: name shouldn't exist in self._managers and not be same as any class attribute
    #     :param manager: subclass of BaseManager
    #     """
    #     assert manager_name not in self._managers, "Manager {} already exists in BaseEngine, Use update_manager() to " \
    #                                                "overwrite".format(manager_name)
    #     assert not hasattr(self, manager_name), "Manager name can not be same as the attribute in BaseEngine"
    #     self._managers[manager_name] = manager
    #     setattr(self, manager_name, manager)
    #     self._managers = OrderedDict(sorted(self._managers.items(), key=lambda k_v: k_v[-1].PRIORITY))

    # def seed(self, random_seed):
    #     start_seed = self.gets_start_index(self.global_config)
    #     random_seed = ((random_seed - start_seed) % self._num_scenarios_per_level) + start_seed
    #     random_seed += self._current_level * self._num_scenarios_per_level
    #     self.global_random_seed = random_seed
    #     super(BaseEngine, self).seed(random_seed)
    #     for mgr in self._managers.values():
    #         mgr.seed(random_seed)

    # @staticmethod
    # def gets_start_index(config):
    #     start_seed = config.get("start_seed", None)
    #     start_scenario_index = config.get("start_scenario_index", None)
    #     assert start_seed is None or start_scenario_index is None, \
    #         "It is not allowed to define `start_seed` and `start_scenario_index`"
    #     if start_seed is not None:
    #         return start_seed
    #     elif start_scenario_index is not None:
    #         return start_scenario_index
    #     else:
    #         logger.warning("Can not find `start_seed` or `start_scenario_index`. Use 0 as `start_seed`")
    #         return 0

    # @property
    # def max_level(self):
    #     return self._max_level

    # @property
    # def current_level(self):
    #     return self._current_level

    # def level_up(self):
    #     old_level = self._current_level
    #     self._current_level = min(self._current_level + 1, self._max_level - 1)
    #     if old_level != self._current_level:
    #         self.seed(self.current_seed + self._num_scenarios_per_level)

    # @property
    # def num_scenarios_per_level(self):
    #     return self._num_scenarios_per_level

    @property
    def current_seed(self):
        return self.global_random_seed

    @property
    def global_seed(self):
        return self.global_random_seed

    def _object_clean_check(self):
        # objects check
        from metadrive.component.vehicle.base_vehicle import BaseVehicle
        from metadrive.component.static_object.traffic_object import TrafficObject
        for manager in self._managers.values():
            assert len(manager.spawned_objects) == 0, manager

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

    def update_manager(self, manager_name: str, manager, destroy_previous_manager=True):
        """
        Update an existing manager with a new one
        :param manager_name: existing manager name
        :param manager: new manager
        """
        assert manager_name in self._managers, "You may want to call register manager, since {} is not in engine".format(
            manager_name
        )
        existing_manager = self._managers.pop(manager_name)
        if destroy_previous_manager:
            existing_manager.destroy()
        self._managers[manager_name] = manager
        setattr(self, manager_name, manager)
        self._managers = OrderedDict(sorted(self._managers.items(), key=lambda k_v: k_v[-1].PRIORITY))

    @property
    def managers(self):
        # whether to froze other managers
        return {"replay_manager": self.replay_manager} if self.replay_episode and not \
            self.only_reset_when_replay else self._managers

    def _get_window_image(self, return_bytes=False):
        window_count = self.graphicsEngine.getNumWindows() - 1
        texture = self.graphicsEngine.getWindow(window_count).getDisplayRegion(0).getScreenshot()

        assert texture.getXSize() == self.global_config["window_size"][0], (
            texture.getXSize(), texture.getYSize(), self.global_config["window_size"]
        )
        assert texture.getYSize() == self.global_config["window_size"][1], (
            texture.getXSize(), texture.getYSize(), self.global_config["window_size"]
        )

        image_bytes = texture.getRamImage().getData()

        if return_bytes:
            return image_bytes, (texture.getXSize(), texture.getYSize())

        img = np.frombuffer(image_bytes, dtype=np.uint8)
        img = img.reshape((texture.getYSize(), texture.getXSize(), 4))
        img = img[::-1]  # Flip vertically
        img = img[..., :-1]  # Discard useless alpha channel
        img = img[..., ::-1]  # Correct the colors

        return img

    def register_manager(self, manager_name: str, manager):
        """
        Add a manager to BaseEngine, then all objects can communicate with this class
        :param manager_name: name shouldn't exist in self._managers and not be same as any class attribute
        :param manager: subclass of BaseManager
        """
        assert manager_name not in self._managers, "Manager {} already exists in BaseEngine, Use update_manager() to " \
                                                   "overwrite".format(manager_name)
        assert not hasattr(self, manager_name), "Manager name can not be same as the attribute in BaseEngine"
        self._managers[manager_name] = manager
        setattr(self, manager_name, manager)
        # self._managers = OrderedDict(sorted(self._managers.items(), key=lambda k_v: k_v[-1].PRIORITY))

    def seed(self, random_seed):
        self.global_random_seed = random_seed
        super(BaseEngine, self).seed(random_seed)
        for mgr in self._managers.values():
            mgr.seed(random_seed)

    def warmup(self):
        """
        This function automatically initialize models/objects. It can prevent the lagging when creating some objects
        for the first time.
        """
        if self.global_config["preload_models"] and self.mode != RENDER_MODE_NONE:
            from metadrive.component.traffic_participants.pedestrian import Pedestrian
            from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
            from metadrive.component.static_object.traffic_object import TrafficBarrier
            from metadrive.component.static_object.traffic_object import TrafficCone
            Pedestrian.init_pedestrian_model()
            warm_up_pedestrian = self.spawn_object(Pedestrian, position=[0, 0], heading_theta=0, record=False)
            warm_up_light = self.spawn_object(BaseTrafficLight, lane=None, position=[0, 0], record=False)
            barrier = self.spawn_object(TrafficBarrier, position=[0, 0], heading_theta=0, record=False)
            cone = self.spawn_object(TrafficCone, position=[0, 0], heading_theta=0, record=False)
            for vel in Pedestrian.SPEED_LIST:
                warm_up_pedestrian.set_velocity([1, 0], vel - 0.1)
            self.clear_objects([warm_up_pedestrian.id, warm_up_light.id, barrier.id, cone.id], record=False)
            warm_up_pedestrian = None
            warm_up_light = None
            barrier = None
            cone = None

    # @staticmethod
    # def try_pull_asset():
    #     from metadrive.engine.asset_loader import AssetLoader
    #     msg = "Assets folder doesn't exist. Begin to download assets..."
    #     if not os.path.exists(AssetLoader.asset_path):
    #         AssetLoader.logger.warning(msg)
    #         pull_asset(update=False)
    #     else:
    #         if AssetLoader.should_update_asset():
    #             AssetLoader.logger.warning(
    #                 "Assets outdated! Current: {}, Expected: {}. "
    #                 "Updating the assets ...".format(asset_version(), VERSION)
    #             )
    #             pull_asset(update=True)
    #         else:
    #             AssetLoader.logger.info("Assets version: {}".format(VERSION))

    # def change_object_name(self, obj, new_name):
    #     raise DeprecationWarning("This function is too dangerous to be used")
    #     """
    #     Change the name of one object, Note: it may bring some bugs if abusing
    #     """
    #     obj = self._spawned_objects.pop(obj.name)
    #     self._spawned_objects[new_name] = obj

    # def add_task(self, object_id, task):
    #     raise DeprecationWarning
    #     self._object_tasks[object_id] = task

    # def has_task(self, object_id):
    #     raise DeprecationWarning
    #     return True if object_id in self._object_tasks else False

    # def get_task(self, object_id):
    #     """
    #     Return task of specific object with id
    #     :param object_id: a filter function, only return objects satisfying this condition
    #     :return: task
    #     """
    #     raise DeprecationWarning
    #     assert object_id in self._object_tasks, "Can not find the task for object(id: {})".format(object_id)
    #     return self._object_tasks[object_id]


if __name__ == "__main__":
    from metadrive.envs.base_env import BASE_DEFAULT_CONFIG

    BASE_DEFAULT_CONFIG["use_render"] = True
    BASE_DEFAULT_CONFIG["show_interface"] = False
    BASE_DEFAULT_CONFIG["render_pipeline"] = True
    world = BaseEngine(BASE_DEFAULT_CONFIG)

    from metadrive.engine.asset_loader import AssetLoader

    car_model = world.loader.loadModel(AssetLoader.file_path("models", "vehicle", "lada", "vehicle.gltf"))
    car_model.reparentTo(world.render)
    car_model.set_pos(0, 0, 190)
    # world.render_pipeline.prepare_scene(env.engine.render)

    world.run()
