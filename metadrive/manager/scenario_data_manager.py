import copy
import os
import numpy as np
import torch
from metadrive.manager.base_manager import BaseManager
from metadrive.scenario.scenario_description import ScenarioDescription as SD, MetaDriveType
from metadrive.scenario.utils import read_scenario_data, read_dataset_summary
from metadrive.scenario.parse_object_state import parse_full_trajectory, parse_object_state, get_idm_route
from metadrive.component.vehicle.vehicle_type import random_vehicle_type, vehicle_type

import json


from easydrive.engine.config import Config
from metadrive.default_config import BASE_DEFAULT_CONFIG

class ScenarioDataManager(BaseManager):
    DEFAULT_DATA_BUFFER_SIZE = 100
    PRIORITY = -10

    @classmethod
    def default_config(cls) -> Config:
        return Config(BASE_DEFAULT_CONFIG)

    def __init__(self, config, loader):
        
        super(ScenarioDataManager, self).__init__()        
        if config is None:
            config = Config()
        default_config = self.default_config()
        default_config.merge_from(config, replace_keys=["agent_configs"])
        self.base_config = default_config

        # self.store_data = engine.global_config["store_data"]
        self.directory = self.base_config["scene_config_directory"]

        # for multi-worker
        # self._scenarios = {}

        # Read summary file first:
        self.read_metadata(loader)
        self.base_config["num_scenarios"] = self.num_scenarios

        # sort scenario for curriculum training
        self.scenario_difficulty = None
        # self.sort_scenarios()


        # stat
        # self.coverage = [0 for _ in range(self.num_scenarios)]
    def _post_process_config(self, config):
        pass

    def read_metadata(self, loader):
        self.metadata, self.idx2scene = {}, []
        self.num_scenarios = 0
        for config_file in os.listdir(self.directory):
            self.num_scenarios += 1
            cfg = Config.fromfile(filename=os.path.join(self.directory, config_file))

            scene_name, frame_range, camera_params, ego_poses, participants = loader(cfg)
            # frame_range : list|tuple [2]
            # camera params : 
            #     "camera_name" :
            #         "K" : list[3][3]
            #         "H" : int
            #         "W" : int
            #         "ego2camera" : list[4][4]
            # ego poses : 
            #     1 : list[4][4]
            #     ...
            #     n : list[4][4]            
            # participants :
            #     "unique_name" : 
            #         "size" :
            #         "type" :
            #         "transforms" :
            #             1 : list[4][4]
            #             ...
            #             n : list[4][4]

            self.metadata[scene_name] = ScenarioDataManager.restructure_metadata(
                config=cfg,
                frame_range=frame_range,
                camera_params=camera_params,
                ego_poses=ego_poses,
                participants=participants
            )
            self.idx2scene.append(scene_name)

    @staticmethod
    def restructure_metadata(config, frame_range, camera_params, ego_poses, participants):
        init_state, agent_state = {}, {}
        for name, tracking in (participants | {'actor': ego_poses}).items():
            if name == 'actor':
                frame_list = range(*frame_range)
                traj = ego_poses
            else:
                frame_list = sorted(tracking['transforms'].keys())
                traj = {frame : tracking['transforms'][frame] for frame in frame_list}
            
            first_frame, last_frame = frame_list[0], frame_list[-1]
            parsed_data = {}
            for frame in frame_list:
                parsed_data[frame] = parse_object_state(traj, frame, first_frame, include_z_position=True)
            
            first_state, last_state = parsed_data[first_frame], parsed_data[last_frame]
            init_state[name] = dict(
                spawn_position=list(first_state["position"]),
                spawn_heading=first_state["heading_theta"],
                spawn_velocity=first_state["velocity"],
                spawn_angular_velocity=first_state["angular_velocity"],
                destination=last_state["position"]
            )
            agent_state[name] = parsed_data

        return {
            'scene_config': config,
            'camera_params':camera_params,
            'ego_poses': ego_poses,
            'participants': participants,
            'init_state': init_state,
            "agent_state": agent_state,
            'frame_range': frame_range
        }

    def reset(self):
        # if not self.store_data:
        #     assert len(self._scenarios) <= 1, "It seems you access multiple scenarios in one episode"
        #     self._scenarios = {}
        self.current_scenario_id = self.np_random.randint(0, self.num_scenarios)
        self.current_config = self.base_config.copy()

        config_dict=self.current_config["actor_config"]
        config_dict["controller"] = config_dict.get("controller", random_vehicle_type(self.np_random)) 

        current_metadata = self.get_current_scenario_data()
        start_frame, end_frame = current_metadata['frame_range']
        ego_poses = current_metadata['ego_poses']
        ground_height = ego_poses[start_frame][2, 3] - config_dict["controller"].DEFAULT_HEIGHT / 2
        current_metadata['ground_height'] = ground_height

    def get_current_scenario_data(self):
        return self.get_scenario_data(self.current_scenario_id)

    def get_scenario_data(self, i, should_copy=False):
        assert 0 <= i < self.num_scenarios, \
            "scenario index exceeds range, scenario index: {}, worker_index: {}".format(i, self.worker_index)
        scenario_name = self.idx2scene[i]
        return self.metadata[scenario_name]

    @property
    def current_scenario_length(self):
        frame_range = self.get_current_scenario_data()['frame_range']
        return frame_range[1] - frame_range[0]

    def sort_scenarios(self):
        """
        TODO(LQY): consider exposing this API to config
        Sort scenarios to support curriculum training. You are encouraged to customize your own sort method
        :return: sorted scenario list
        """
        if self.engine.max_level == 0:
            raise ValueError("Curriculum Level should be greater than 1")
        elif self.engine.max_level == 1:
            return

        def _score(scenario_id):
            file_path = os.path.join(self.directory, self.mapping[scenario_id], scenario_id)
            scenario = read_scenario_data(file_path, centralize=True)
            obj_weight = 0

            # calculate curvature
            ego_car_id = scenario[SD.METADATA][SD.SDC_ID]
            state_dict = scenario["tracks"][ego_car_id]["state"]
            valid_track = state_dict["position"][np.where(state_dict["valid"].astype(int))][..., :2]

            dir = valid_track[1:] - valid_track[:-1]
            dir = np.arctan2(dir[..., 1], dir[..., 0])
            curvature = sum(abs(dir[1:] - dir[:-1]) / np.pi) + 1

            sdc_moving_dist = SD.sdc_moving_dist(scenario)
            num_moving_objs = SD.num_moving_object(scenario, object_type=MetaDriveType.VEHICLE)
            return sdc_moving_dist * curvature + num_moving_objs * obj_weight, scenario

        start = self.start_scenario_index
        end = self.start_scenario_index + self.num_scenarios
        id_score_scenarios = [(s_id, *_score(s_id)) for s_id in self.summary_lookup[start:end]]
        id_score_scenarios = sorted(id_score_scenarios, key=lambda scenario: scenario[-2])
        self.summary_lookup[start:end] = [id_score_scenario[0] for id_score_scenario in id_score_scenarios]
        self.scenario_difficulty = {
            id_score_scenario[0]: id_score_scenario[1]
            for id_score_scenario in id_score_scenarios
        }
        self._scenarios = {i + start: id_score_scenario[-1] for i, id_score_scenario in enumerate(id_score_scenarios)}

    @property
    def current_scenario_difficulty(self):
        return self.scenario_difficulty[self.summary_lookup[self.engine.global_random_seed]
                                        ] if self.scenario_difficulty is not None else 0

    # @property
    # def data_coverage(self):
    #     return sum(self.coverage) / len(self.coverage) * self.engine.global_config["num_workers"]

    def destroy(self):
        """
        Clear memory
        """
        super(ScenarioDataManager, self).destroy()
        self._scenarios = {}
        # Config.clear_nested_dict(self.summary_dict)
        self.summary_lookup.clear()
        self.mapping.clear()
        self.summary_dict, self.summary_lookup, self.mapping = None, None, None


class ScenarioOnlineDataManager(BaseManager):
    """
    Compared to ScenarioDataManager, this manager allow user to pass in Scenario Description online.
    It will not read data from disk, but receive data from user.
    """
    PRIORITY = -10
    _scenario = None

    @property
    def current_scenario_summary(self):
        return self.current_scenario[SD.METADATA]

    def set_scenario(self, scenario_description):
        SD.sanity_check(scenario_description)
        scenario_description = SD.centralize_to_ego_car_initial_position(scenario_description)
        self._scenario = scenario_description

    def get_scenario(self, seed=None, should_copy=False):
        assert self._scenario is not None, "Please set scenario first via env.set_scenario(scenario_description)!"
        if should_copy:
            return copy.deepcopy(self._scenario)
        return self._scenario

    def get_metadata(self):
        raise ValueError()
        state = super(ScenarioDataManager, self).get_metadata()
        raw_data = self.current_scenario
        state["raw_data"] = raw_data
        return state

    @property
    def current_scenario_length(self):
        return self.current_scenario[SD.LENGTH]

    @property
    def current_scenario(self):
        return self._scenario

    @property
    def current_scenario_difficulty(self):
        return 0

    @property
    def current_scenario_id(self):
        return self.current_scenario_summary["scenario_id"]

    @property
    def data_coverage(self):
        return None

    def destroy(self):
        """
        Clear memory
        """
        super(ScenarioOnlineDataManager, self).destroy()
        self._scenario = None
