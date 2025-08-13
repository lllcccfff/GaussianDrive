import copy

from metadrive.component.ground import GroundPlane
from metadrive.constants import DEFAULT_AGENT
from metadrive.manager.base_manager import BaseManager
from metadrive.engine.logger import get_logger, set_log_level

from easydrive.engine import MODELS
from easydrive.utils.base_utils import dotdict
logger = get_logger()


class ScenarioMapManager(BaseManager):
    PRIORITY = 0  # Map update has the most high priority
    DEFAULT_DATA_BUFFER_SIZE = 200

    def __init__(self):
        super(ScenarioMapManager, self).__init__()
        self.store_map = self.engine.global_config.get("store_map", False)
        self.current_map = None

        # we put the route searching function here
        self.sdc_start_point = None
        self.sdc_destinations = []
        self.sdc_dest_point = None
        self.current_sdc_route = None

        self.ground = None

    def reset(self):
        seed = self.engine.global_random_seed

        self.current_sdc_route = None
        self.sdc_dest_point = None

        scene_data = self.engine.data_manager.get_current_scenario_data()
        self.model = MODELS.build(
            cfg=scene_data['config'].model_cfg,
            render_mode='gui',
            renderer_cfg=scene_data['config'].renderer_cfg
        )
        self.model.model_setup(
            scene_data['CameraBasedDataset'],
            scene_data['BoundingBoxDataset']
        )
        self.model.load_model(**scene_data['config'].visualizer_cfg.model_path)

        self.spawn_object(
            GroundPlane, 
            direction=[0,0,1.], 
            constant=scene_data['ground_height'], 
            random_seed=self.random_seed
        )
        # self.update_route()

    def clear_object(self, object_id):
        obj = self.spawned_objects.pop(object_id)
        obj.destroy()  

    def update_route(self):
        data = self.engine.data_manager.current_scenario

    def filter_path(self, start_lanes, end_lanes):
        for start in start_lanes:
            for end in end_lanes:
                path = self.current_map.road_network.shortest_path(start[0].index, end[0].index)
                if len(path) > 0:
                    return (start[0].index, end[0].index)
        return None


    def load_map(self, map):
        map.attach_to_world()
        self.current_map = map

    def unload_map(self, map):
        map.detach_from_world()
        self.current_map = None
        if not self.engine.global_config["store_map"]:
            map.destroy()
            assert len(self.spawned_objects) == 0

    def destroy(self):
        self.clear_stored_maps()
        self._stored_maps = None
        self.current_map = None

        self.sdc_start_point = None
        self.sdc_destinations = []
        self.sdc_dest_point = None
        self.current_sdc_route = None

        super(ScenarioMapManager, self).destroy()
    
    def clear_stored_maps(self):
        for m in self._stored_maps.values():
            if m is not None:
                m.detach_from_world()
                m.destroy()
        self._stored_maps = {
            i: None
            for i in range(self.start_scenario_index, self.start_scenario_index + self.map_num)
        }

    @property
    def num_stored_maps(self):
        return sum([1 if m is not None else 0 for m in self.engine.map_manager._stored_maps.values()])
