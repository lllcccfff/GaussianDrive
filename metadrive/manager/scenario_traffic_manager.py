import copy

import numpy as np

from metadrive.component.static_object.traffic_object import TrafficCone, TrafficBarrier
from metadrive.component.traffic_participants.cyclist import Cyclist, CyclistBoundingBox
from metadrive.component.traffic_participants.pedestrian import Pedestrian, PedestrianBoundingBox
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.vehicle.vehicle_type import LVehicle, MVehicle, XLVehicle, \
    VaryingDynamicsBoundingBoxVehicle, SVehicle, DefaultVehicle
from metadrive.constants import DEFAULT_AGENT
from metadrive.engine.logger import get_logger
from metadrive.manager.base_manager import BaseManager
from metadrive.policy.idm_policy import TrajectoryIDMPolicy
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.policy.replay_policy import ReplayTrafficParticipantPolicy
from metadrive.scenario.parse_object_state import parse_object_state, get_idm_route, get_max_valid_indicis
from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.type import MetaDriveType
from metadrive.utils.math import norm
from metadrive.utils.math import wrap_to_pi

logger = get_logger()


class ScenarioTrafficManager(BaseManager):
    STATIC_THRESHOLD = 3  # m, static if moving distance < 5
    IDM_ACT_BATCH_SIZE = 5

    # project cars to ego vehicle coordinates, only vehicles behind ego car and in a certain region can get IDM policy
    IDM_CREATE_SIDE_CONSTRAINT = 15  # m
    IDM_CREATE_FORWARD_CONSTRAINT = -1  # m
    IDM_CREATE_MIN_LENGTH = 5  # m

    # project cars to ego vehicle coordinates, only vehicles outside the region can be created
    GENERATION_SIDE_CONSTRAINT = 2  # m
    GENERATION_FORWARD_CONSTRAINT = 8  # m

    # filter noise static object: barrier and cone
    MIN_VALID_FRAME_LEN = 20  # frames

    def __init__(self):
        super(ScenarioTrafficManager, self).__init__()
        # for filtering some static cars
        self._static_car_id = set()
        self._moving_car_id = set()


        # an async trick for accelerating IDM policy
        self.idm_policy_count = 0
        self._obj_to_clean_this_frame = []

        # some flags
        self.even_sample_v = self.engine.global_config.get("even_sample_vehicle_class", None)
        if self.even_sample_v is not None:
            raise DeprecationWarning("even_sample_vehicle_class is deprecated!")

        self.need_default_vehicle = self.engine.global_config["default_vehicle_in_traffic"]
        self.is_ego_vehicle_replay = self.engine.global_config["agent_policy"] == ReplayEgoCarPolicy
        self._filter_overlapping_car = self.engine.global_config["filter_overlapping_car"]

        # config
        self._traffic_v_config = self.get_traffic_v_config()

    def before_reset(self):
        super(ScenarioTrafficManager, self).before_reset()
        # reset_vehicle_type_count(self.np_random)

    def after_reset(self):
        self._static_car_id = set()
        self.spawned_object = {}
        self.idm_policy_count = 0
        self._try_spawning()
    
    def before_step(self, *args, **kwargs):
        self._obj_to_clean_this_frame = []
        for obj_id, obj in self.spawned_objects.values():
            if isinstance(self._object_policies[obj_id], TrajectoryIDMPolicy):
                p = self._object_policies[obj_id]
                if p.arrive_destination:
                    self._obj_to_clean_this_frame.append(obj_id)
                else:
                    do_speed_control = self.episode_step % self.IDM_ACT_BATCH_SIZE == p.policy_index
                    obj.before_step(p.act(do_speed_control))

    def after_step(self, *args, **kwargs):
        self.traffic_poses = {}

        if self.episode_step < self.current_scenario_length:
            replay_done = False
            self._try_spawning()

            for obj_id, obj in self.spawned_objects.items():
                if isinstance(self._object_policies[obj_id], ReplayTrafficParticipantPolicy):
                    # static object will not be cleaned!
                    policy = self._object_policies[obj_id]
                    if policy.is_current_step_valid(self.episode_step):
                        policy.act()
                    else:
                        self._obj_to_clean_this_frame.append(obj_id)

                if obj_id not in self._obj_to_clean_this_frame:
                    self.traffic_poses[obj_id] = obj.ego_pose

        for obj_id in self._obj_to_clean_this_frame:
            self.clear_object(obj_id)
        return dict(default_agent=dict(replay_done=replay_done))

    def _try_spawning(self):
        boundingboxes = self.scenario_data['bounding_box_objects']
        trajectories = self.scenario_data['trajectory_policy_data']
        object_states = self.scenario_data['object_state']
        for id, bbox in boundingboxes.items():
            if id in self.spawned_object:
                continue
            if self.episode_step < bbox.first_frame or self.episode_step > bbox.last_frame:
                return
            
            if bbox.object_type == 'rigid':
                self.spawn_vehicle(id, bbox, trajectories[id], object_states[id])
            elif bbox.object_type == 'nonrigid':
                self.spawn_pedestrian(id, bbox, trajectories[id], object_states[id])
            else:
                logger.warning("Do not support {}".format(bbox.object_type))
            # elif track["type"] == MetaDriveType.CYCLIST:
            #     self.spawn_cyclist(scenario_id, track)

    @property
    def scenario_data(self):
        return self.engine.data_manager.get_current_scenario_data()

    @property
    def current_scenario_length(self):
        return self.engine.data_manager.current_scenario_length

    @property
    def vehicles(self):
        return list(self.engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle)).values())

    def spawn_vehicle(self, oid, bbox, track, state):
        # create vehicle
        vehicle_class = get_vehicle_type(bbox.size[1], False)
        obj_name = oid
        v = self.spawn_object(
            vehicle_class, position=state['position'], heading=state["heading"], name=obj_name, size=bbox.size
        )

        # add policy
        need_reactive_traffic = self.engine.global_config["reactive_traffic"]
        if not need_reactive_traffic or oid in self._static_car_id:
            policy = self.add_policy(v.name, ReplayTrafficParticipantPolicy, v, track)
            policy.act()
        else:
            idm_route = get_idm_route(track["state"]["position"][..., :2])
            # only not static and behind ego car, it can get reactive policy
            self.add_policy(
                v.name, TrajectoryIDMPolicy, v, self.generate_seed(), idm_route,
                self.idm_policy_count % self.IDM_ACT_BATCH_SIZE
            )
            # no act() is required for IDMPolicy
            self.idm_policy_count += 1

    def spawn_pedestrian(self, oid, bbox, track, state):
        obj_name = oid
        obj = self.spawn_object(
            Pedestrian,
            name=obj_name,
            position=state["position"],
            heading_theta=state["heading"],
            size=bbox.size
        )
        policy = self.add_policy(obj.name, ReplayTrafficParticipantPolicy, obj, track)
        policy.act()

    def spawn_cyclist(self, oid, bbox, track, state):
        if self.episode_step < bbox.first_frame or self.episode_step > bbox.last_frame:
            return
        
        state = parse_object_state(track, self.episode_step, include_z_position=False)

        obj_name = scenario_id if self.engine.global_config["force_reuse_object_name"] else None
        cls = Cyclist

        position = list(state["position"])
        obj = self.spawn_object(
            cls,
            name=obj_name,
            position=position,
            heading_theta=state["heading"],
            width=state["width"],
            length=state["length"],
            height=state["height"],
        )
        self._scenario_id_to_obj_id[scenario_id] = obj.name
        self._obj_id_to_scenario_id[obj.name] = scenario_id
        policy = self.add_policy(obj.name, ReplayTrafficParticipantPolicy, obj, track)
        policy.act()

    def get_state(self):
        # Record mapping from original_id to new_id
        ret = {}
        ret[SD.ORIGINAL_ID_TO_OBJ_ID] = self.scenario_id_to_obj_id
        ret[SD.OBJ_ID_TO_ORIGINAL_ID] = self.obj_id_to_scenario_id
        return ret

    @property
    def ego_vehicle(self):
        return self.engine.agents[DEFAULT_AGENT]

    def is_static_object(self, obj_id):
        return isinstance(self.spawned_objects[obj_id], TrafficBarrier) \
            or isinstance(self.spawned_objects[obj_id], TrafficCone)

    @property
    def obj_id_to_scenario_id(self):
        # For outside access, we return traffic vehicles and ego car
        ret = copy.copy(self._obj_id_to_scenario_id)
        ret[self.sdc_object_id] = self.sdc_scenario_id
        return ret

    @property
    def scenario_id_to_obj_id(self):
        # For outside access, we return traffic vehicles and ego car
        ret = copy.copy(self._scenario_id_to_obj_id)
        ret[self.sdc_scenario_id] = self.sdc_object_id
        return ret

    @staticmethod
    def get_traffic_v_config():
        v_config = dict(
            navigation_module=None,
            show_navi_mark=False,
            show_dest_mark=False,
            enable_reverse=False,
            show_lidar=False,
            show_lane_line_detector=False,
            show_side_detector=False,
        )
        return v_config


# type_count = [0 for i in range(3)]


def get_vehicle_type(length, need_default_vehicle=False, use_bounding_box=False):
    if use_bounding_box:
        return VaryingDynamicsBoundingBoxVehicle
    if need_default_vehicle:
        return DefaultVehicle
    if length <= 4:
        return SVehicle
    elif length <= 5.2:
        return MVehicle
    elif length <= 6.2:
        return LVehicle
    else:
        return XLVehicle


# def reset_vehicle_type_count(np_random=None):
#     global type_count
#     if np_random is None:
#         type_count = [0 for i in range(3)]
#     else:
#         type_count = [np_random.randint(100) for i in range(3)]
