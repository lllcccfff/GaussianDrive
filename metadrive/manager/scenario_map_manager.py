

from metadrive.component.terrain.ground import GroundPlane
from metadrive.component.terrain.mesh_terrain import MeshTerrain
from metadrive.manager.base_manager import BaseManager
from metadrive.utils.logger import get_logger

logger = get_logger()


class ScenarioMapManager(BaseManager):
    PRIORITY = 0  # Map update has the most high priority
    DEFAULT_DATA_BUFFER_SIZE = 200

    def __init__(self, config, loader):
        super(ScenarioMapManager, self).__init__()
        self.config = config
        self.store_map = self.config.get("store_map", False)
        self.current_map = None

        # we put the route searching function here
        self.sdc_start_point = None
        self.sdc_destinations = []
        self.sdc_dest_point = None
        self.current_sdc_route = None

        self.loader = loader
        self.ground = None

    def reset(self, config, scene_config, ground_height, physics_world, scene_mesh_path, **kwargs):
        self.config = config
        self.current_sdc_route = None
        self.sdc_dest_point = None

        vec_map = self.loader(scene_config)

        if not scene_mesh_path:
            self.spawn_object(
                GroundPlane,
                physics_world=physics_world,
                direction=[0, 0, 1.0],
                constant=ground_height,
                random_seed=self.random_seed,
            )
        else:
            self.spawn_object(
                MeshTerrain, model_path=scene_mesh_path, physics_world=physics_world, random_seed=self.random_seed
            )

    def clear_object(self, object_id):
        obj = self.spawned_objects.pop(object_id)
        obj.destroy()

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
            i: None for i in range(self.start_scenario_index, self.start_scenario_index + self.map_num)
        }
