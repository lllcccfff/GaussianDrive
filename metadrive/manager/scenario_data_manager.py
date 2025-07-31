import copy
import os
import numpy as np

from metadrive.manager.base_manager import BaseManager
from metadrive.scenario.scenario_description import ScenarioDescription as SD, MetaDriveType
from metadrive.scenario.utils import read_scenario_data, read_dataset_summary
from metadrive.scenario.parse_object_state import parse_full_trajectory, parse_object_state, get_idm_route

from easydrive.engine import config, DATALOADERS
from easydrive.dataloader.dataset.bounding_box_dataset import BoundingBoxDataset
from easydrive.dataloader.dataset.camera_based_dataset import CameraBasedDataset

class ScenarioDataManager(BaseManager):
    DEFAULT_DATA_BUFFER_SIZE = 100
    PRIORITY = -10
    def __init__(self):
        super(ScenarioDataManager, self).__init__()
        from metadrive.engine.engine_utils import get_engine
        engine = get_engine()

        self.store_data = engine.global_config["store_data"]
        self.directory = engine.global_config["scene_config_directory"]

        # for multi-worker
        self.worker_index = self.engine.global_config["worker_index"]
        self._scenarios = {}

        # Read summary file first:
        self.read_metadata()
        engine.global_config["num_scenarios"] = self.num_scenarios

        # sort scenario for curriculum training
        self.scenario_difficulty = None
        self.sort_scenarios()


        # stat
        self.coverage = [0 for _ in range(self.num_scenarios)]

    def read_metadata(self):
        self.metadata, self.idx2scene = {}, []
        self.num_scenarios = 0
        for config_file in enumerate(self.directory):
            self.num_scenarios += 1
            cfg = config.Config(filename=config_file)
            dataloader = DATALOADERS.build(
                cfg=cfg.dataloader_cfg,
                shuffle=False,
                data_involved="all",
                visualize=False,
            )
            scene_name = cfg.dataloader_cfg.dataset_cfg.scene_name
            camera_data = dataloader.dataset.sensors
            ego_poses = dataloader.load_dataset('EgoPoseDataset').ego_poses
            tracking_labels = dataloader.load_dataset('BoundingBoxDataset').bounding_boxes
            ground_height = dataloader.load_dataset('PointCloudDataset').ground_height
            self.metadata[scene_name] = ScenarioDataManager.restructure_metadata(
                config=cfg,
                cameras=camera_data,
                ego_poses=ego_poses,
                trackings=tracking_labels,
                ground_height=ground_height
            )
            self.idx2scene.append(scene_name)

    @staticmethod
    def restructure_metadata(config, cameras, ego_poses, trackings, ground_height):        
        init_state = parse_object_state(ego_poses, 0, check_last_state=False, include_z_position=True)
        last_state = parse_object_state(ego_poses, -1, check_last_state=True)
        agent_state = dict(
            spawn_position=list(init_state["position"]),
            spawn_heading=init_state["heading"],
            spawn_velocity=init_state["velocity"],
            destination=last_state["position"]
        )

        trajectory_policy_data = {}
        object_state = {}
        for object_id, tracking in trackings:
            traj = {}
            for frame in range(tracking.first_frame, tracking.last_frame):
                traj[frame] = tracking.get_transform(frame)
            
            parsed_data = {}
            for frame in range(tracking.first_frame, tracking.last_frame):
                parsed_data[frame] = parse_object_state(traj, frame)
            
            trajectory_policy_data[object_id] = parsed_data
            object_state[object_id] = parsed_data[tracking.first_frame]

        return {
            'config': config,
            'cameras_objects':cameras,
            'bounding_box_objects': trackings,
            'ground_height': ground_height,
            'ego_poses': ego_poses,
            'agent_state': agent_state,
            'trajectory_policy_data': trajectory_policy_data,
            'object_state': object_state
        }


    def before_reset(self):
        if not self.store_data:
            assert len(self._scenarios) <= 1, "It seems you access multiple scenarios in one episode"
            self._scenarios = {}
        self.current_scenario_id = self.np_random.randint(0, self.num_scenarios)

    def get_scenario_data(self, i, should_copy=False):
        assert 0 <= i < self.num_scenarios, \
            "scenario index exceeds range, scenario index: {}, worker_index: {}".format(i, self.worker_index)
        scenario_name = self.idx2scene[i]
        return self.metadata[scenario_name]

    def get_current_scenario_data(self):
        return self.get_scenario_data(self.current_scenario_id)

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

    @property
    def data_coverage(self):
        return sum(self.coverage) / len(self.coverage) * self.engine.global_config["num_workers"]

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
