import math
import os
from collections import deque
from typing import Union, Optional, List

import numpy as np
from panda3d.bullet import BulletVehicle, BulletBoxShape, ZUp
from panda3d.core import Material, Vec3, TransformState

from metadrive.base_class.base_object import BaseObject
from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.lane.circular_lane import CircularLane
from metadrive.component.lane.point_lane import PointLane
from metadrive.component.lane.straight_lane import StraightLane
# from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.component.pg_space import VehicleParameterSpace, ParameterSpace
from metadrive.constants import CamMask, get_color_palette
from metadrive.constants import MetaDriveType, CollisionGroup
from metadrive.constants import Semantics
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.engine_utils import get_engine, engine_initialized
from metadrive.engine.logger import get_logger
from metadrive.engine.physics_node import BaseRigidBodyNode
from metadrive.utils import Config, safe_clip_for_small_array
from metadrive.utils.math import get_vertical_vector, norm, clip
from metadrive.utils.math import wrap_to_pi
from metadrive.utils.utils import get_object_from_node

logger = get_logger()


class BaseVehicleState:
    def __init__(self):
        self.init_state_info()

    def init_state_info(self):
        """
        Call this before reset()/step()
        """
        self.crash_vehicle = False
        self.crash_human = False
        self.crash_object = False
        self.crash_sidewalk = False
        self.crash_building = False

        # traffic light
        self.red_light = False
        self.yellow_light = False
        self.green_light = False  # should always be False, since we don't detect green light

        # lane line detection
        self.on_yellow_continuous_line = False
        self.on_white_continuous_line = False
        self.on_broken_line = False
        self.on_crosswalk = False

        # contact results, a set containing objects type name for rendering
        self.contact_results = set()


class BaseVehicle(BaseObject, BaseVehicleState):
    """
    Vehicle chassis and its wheels index
                    0       1
                    II-----II
                        |
                        |  <---chassis/wheelbase
                        |
                    II-----II
                    2       3
    """
    COLLISION_MASK = CollisionGroup.Vehicle
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.BASE_VEHICLE)
    MAX_LENGTH = 10
    MAX_WIDTH = 2.5
    MAX_STEERING = 60
    SEMANTIC_LABEL = Semantics.CAR.label

    # LENGTH = None
    # WIDTH = None
    # HEIGHT = None

    TIRE_RADIUS = None
    TIRE_WIDTH = 0.4
    LATERAL_TIRE_TO_CENTER = None
    FRONT_WHEELBASE = None
    REAR_WHEELBASE = None

    CHASSIS_TO_WHEEL_AXIS = 0.2
    SUSPENSION_LENGTH = 15
    SUSPENSION_STIFFNESS = 40

    # MASS = None

    # control
    STEERING_INCREMENT = 0.05

    # save memory, load model once
    model_collection = {}
    path = None

    def __init__(
        self,
        size=None,
        vehicle_config: Union[dict, Config] = None,
        name: str = None,
        random_seed=None,
        position=None,
        heading=None,
        _calling_reset=True,
    ):
        """
        This Vehicle Config is different from self.get_config(), and it is used to define which modules to use, and
        module parameters. And self.physics_config defines the physics feature of vehicles, such as length/width
        :param vehicle_config: mostly, vehicle module config
        :param random_seed: int
        """
        # check
        assert vehicle_config is not None, "Please specify the vehicle config."
        assert engine_initialized(), "Please make sure game engine is successfully initialized!"

        # NOTE: it is the game engine, not vehicle drivetrain
        # self.engine = get_engine()

        if size is None:
            size = (self.DEFAULT_WIDTH, self.DEFAULT_LENGTH, self.DEFAULT_HEIGHT)
        BaseObject.__init__(self, size, name, random_seed, self.engine.global_config["vehicle_config"])
        BaseVehicleState.__init__(self)
        self.update_config(vehicle_config)
        self.set_metadrive_type(MetaDriveType.VEHICLE)

        # build vehicle physics model
        self.vehicle, self.body = self._create_vehicle_chassis()
        self.wheels = self._create_wheel()
        self.attachDyWld(self.vehicle)

        # powertrain config
        self.max_steering = self.config["max_steering"]

        # state info
        self.throttle_brake = 0.0
        self.steering = 0
        self.last_current_action = deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)
        self.last_position = (0, 0)
        self.last_velocity = 0
        self.last_heading = 0
        self.dist_to_left_side = None
        self.dist_to_right_side = None

        # step info
        self.out_of_route = None
        self.on_lane = None
        self._init_step_info()

        #
        self.break_down = False
        # if self.engine.current_map is not None:
        if _calling_reset:
            self.reset(position=position, heading=heading, vehicle_config=vehicle_config)

    def _init_step_info(self):
        # done info will be initialized every frame
        self.init_state_info()
        self.out_of_route = False  # re-route is required if is false
        self.on_lane = True  # on lane surface or not

    @staticmethod
    def _preprocess_action(action):
        if action is None:
            return None, {"raw_action": None}
        action = safe_clip_for_small_array(action, -1, 1)
        return action, {'raw_action': (action[0], action[1])}

    def reset(
        self,
        name=None,
        random_seed=None,
        position: np.ndarray = None,
        heading: float = 0.0,
        velocity: np.ndarray = None,
        *args,
        **kwargs
    ):
        """
        pos is a 2-d array, and heading is a float (unit degree)
        if pos is not None, vehicle will be reset to the position
        else, vehicle will be reset to spawn place
        """
        if name is not None:
            self.rename(name)

        # reset fully
        if random_seed is not None:
            assert isinstance(random_seed, int)
            self.seed(random_seed)
            self.sample_parameters()

        from metadrive.component.vehicle.vehicle_type import vehicle_class_to_type
        self.config["vehicle_model"] = vehicle_class_to_type[self.__class__]

        self.set_heading_theta(heading)
        # self.set_wheel_friction(self.config["wheel_friction"])

        if len(position) == 2:
            self.set_position(position, height=self.HEIGHT / 2)
        elif len(position) == 3:
            self.set_position(position[:2], height=position[-1])
        else:
            raise ValueError()

        # done info
        self._init_step_info()

        self.update_dist_to_left_right()
        self.energy_consumption = 0

        if self.config["spawn_velocity"] is not None:
            self.set_velocity(self.config["spawn_velocity"], in_local_frame=self.config["spawn_velocity_car_frame"])

        # self.add_light()

    def before_step(self, action=None):
        """
        Save info and make decision before action
        """
        # init step info to store info before each step
        # if action is None:
        #     action = [0, 0]

        self._init_step_info()
        action, step_info = self._preprocess_action(action)

        self.last_position = self.position  # 2D vector
        self.last_velocity = self.velocity  # 2D vector
        self.last_heading_theta = self.heading_theta
        if action is not None:
            self.last_current_action.append(action)  # the real step of physics world is implemented in taskMgr.step()
        # if self.increment_steering:
        #     self._set_incremental_action(action)
        # else:
        self._set_action(action)
        return step_info

    def after_step(self):
        self._state_check()
        step_energy, episode_energy = self._update_energy_consumption()
        # self.out_of_route = self._out_of_route()
        step_info = {}
        step_info.update(
            {
                "speed": float(self.speed),
                "angular_speed": float(self.angular_velocity),
                "steering": float(self.steering),
                "acceleration": float(self.throttle_brake),
                "step_energy": step_energy,
                "episode_energy": episode_energy,
            }
        )

        return step_info

    def _out_of_route(self):
        left, right = self._dist_to_route_left_right()
        return True if right < 0 or left < 0 else False

    def _update_energy_consumption(self):
        """
        The calculation method is from
        https://www.researchgate.net/publication/262182035_Reduction_of_Fuel_Consumption_and_Exhaust_Pollutant_Using_Intelligent_Transport_chassis
        default: 3rd gear, try to use ae^bx to fit it, dp: (90, 8), (130, 12)
        :return: None
        """
        distance = norm(self.last_position[0] - self.position[0], self.last_position[1] - self.position[1]) / 1000  # km
        step_energy = 3.25 * math.pow(np.e, 0.01 * self.speed_km_h) * distance / 100
        # step_energy is in Liter, we return mL
        step_energy = step_energy * 1000
        self.energy_consumption += step_energy  # L/100 km
        return step_energy, self.energy_consumption
    
    def _state_check(self):
        """
        Check States and filter to update info
        """
        # result_1 = self.engine.physics_world.static_world.contactTest(self.body, True)
        result_2 = self.engine.physics_world.dynamic_world.contactTest(self.body, False)
        contacts = set()
        for contact in result_2.getContacts():
            node0 = contact.getNode0()
            node1 = contact.getNode1()
            node = node0 if node1.getName() == MetaDriveType.VEHICLE else node1
            name = node.getName()
            if name == MetaDriveType.VEHICLE:
                self.crash_vehicle = True
            elif name in [MetaDriveType.PEDESTRIAN, MetaDriveType.CYCLIST]:
                self.crash_human = True
            else:
                continue
            contacts.add(name)

        self.contact_results.update(contacts)

    """------------------------------------------- act -------------------------------------------------"""

    def set_steering(self, steering):
        steering = float(steering)
        self.chassis.setSteeringValue(steering, 0)
        self.chassis.setSteeringValue(steering, 1)
        self.steering = steering

    def set_throttle_brake(self, throttle_brake):
        throttle_brake = float(throttle_brake)
        self._apply_throttle_brake(throttle_brake)
        self.throttle_brake = throttle_brake

    def _set_action(self, action):
        if action is None:
            return
        steering = action[0]
        self.throttle_brake = action[1]
        self.steering = steering
        self.chassis.setSteeringValue(self.steering * self.max_steering, 0)
        self.chassis.setSteeringValue(self.steering * self.max_steering, 1)
        self._apply_throttle_brake(action[1])
        
    def _apply_throttle_brake(self, throttle_brake):
        max_engine_force = self.config["max_engine_force"]
        max_brake_force = self.config["max_brake_force"]
        for wheel_index in range(4):
            if throttle_brake >= 0:
                self.chassis.setBrake(2.0, wheel_index)
                if self.speed_km_h > self.max_speed_km_h:
                    self.chassis.applyEngineForce(0.0, wheel_index)
                else:
                    self.chassis.applyEngineForce(max_engine_force * throttle_brake, wheel_index)
            else:
                if self.enable_reverse:
                    self.chassis.applyEngineForce(max_engine_force * throttle_brake, wheel_index)
                    self.chassis.setBrake(0, wheel_index)
                else:
                    DEADZONE = 0.01

                    # Speed m/s in car's heading:
                    heading = self.heading
                    velocity = self.velocity
                    speed_in_heading = velocity[0] * heading[0] + velocity[1] * heading[1]

                    if speed_in_heading < DEADZONE:
                        self.chassis.applyEngineForce(0.0, wheel_index)
                        self.chassis.setBrake(2, wheel_index)
                    else:
                        self.chassis.applyEngineForce(0.0, wheel_index)
                        self.chassis.setBrake(abs(throttle_brake) * max_brake_force, wheel_index)

    """---------------------------------------- vehicle info ----------------------------------------------"""

    def update_dist_to_left_right(self):
        self.dist_to_left_side, self.dist_to_right_side = 0, 0

    """---------------------------------------- some math tool ----------------------------------------------"""

    def heading_diff(self, target_lane):
        lateral = None
        if isinstance(target_lane, StraightLane):
            lateral = np.asarray(get_vertical_vector(target_lane.end - target_lane.start)[1])
        elif isinstance(target_lane, CircularLane):
            if not target_lane.is_clockwise():
                lateral = self.position - target_lane.center
            else:
                lateral = target_lane.center - self.position
        elif isinstance(target_lane, PointLane):
            lateral = target_lane.lateral_direction(target_lane.local_coordinates(self.position)[0])

        lateral_norm = norm(lateral[0], lateral[1])
        forward_direction = self.heading
        # print(f"Old forward direction: {self.forward_direction}, new heading {self.heading}")
        forward_direction_norm = norm(forward_direction[0], forward_direction[1])
        if not lateral_norm * forward_direction_norm:
            return 0
        cos = (
            (forward_direction[0] * lateral[0] + forward_direction[1] * lateral[1]) /
            (lateral_norm * forward_direction_norm)
        )
        # return cos
        # Normalize to 0, 1
        return clip(cos, -1.0, 1.0) / 2 + 0.5

    def lane_distance_to(self, vehicle, lane: AbstractLane = None) -> float:
        assert self.navigation is not None, "a routing and localization module should be added " \
                                            "to interact with other vehicles"
        if not vehicle:
            return np.nan
        if not lane:
            lane = self.lane
        return lane.local_coordinates(vehicle.position)[0] - lane.local_coordinates(self.position)[0]

    """-------------------------------------- for vehicle making ------------------------------------------"""

    def _create_vehicle_chassis(self):
        # assert self.LENGTH < BaseVehicle.MAX_LENGTH, "Vehicle is too large!"
        # assert self.WIDTH < BaseVehicle.MAX_WIDTH, "Vehicle is too large!"

        chassis = BaseRigidBodyNode(self.name, MetaDriveType.VEHICLE, self.MASS)

        chassis_shape = BulletBoxShape(Vec3(self.WIDTH / 2, self.LENGTH / 2, self.HEIGHT / 2))
        ts = TransformState.makePos(Vec3(0, 0, self.HEIGHT / 2))
        chassis.addShape(chassis_shape, ts)
        chassis.setDeactivationEnabled(False)
        chassis.notifyCollisions(True)  # advance collision check, do callback in pg_collision_callback

        physics_world = get_engine().physics_world
        vehicle_chassis = BulletVehicle(physics_world.dynamic_world, chassis)
        vehicle_chassis.setCoordinateSystem(ZUp)
        return vehicle_chassis, chassis
    
    def _create_wheel(self):
        f_l = self.FRONT_WHEELBASE
        r_l = -self.REAR_WHEELBASE
        lateral = self.LATERAL_TIRE_TO_CENTER
        axis_height = self.TIRE_RADIUS - self.CHASSIS_TO_WHEEL_AXIS
        radius = self.TIRE_RADIUS
        wheels = []
        for id, pos in enumerate(
            [Vec3(lateral, f_l, axis_height), Vec3(-lateral, f_l, axis_height),
            Vec3(lateral, r_l, axis_height), Vec3(-lateral, r_l, axis_height)]
        ):
            wheel = self.vehicle.createWheel()
            wheel.setChassisConnectionPointCs(pos)
            wheel.setFrontWheel(True if id < 2 else False)
            wheel.setWheelDirectionCs(Vec3(0, 0, -1))
            wheel.setWheelAxleCs(Vec3(1, 0, 0))

            wheel.setWheelRadius(radius)
            wheel.setMaxSuspensionTravelCm(self.SUSPENSION_LENGTH)
            wheel.setSuspensionStiffness(self.SUSPENSION_STIFFNESS)
            wheel.setWheelsDampingRelaxation(4.8)
            wheel.setWheelsDampingCompression(1.2)
            wheel_friction = self.config["wheel_friction"] if not self.config["no_wheel_friction"] else 0
            wheel.setFrictionSlip(wheel_friction)
            wheel.setRollInfluence(0.5)
            wheels.append(wheel)
        return wheels

    def destroy(self):
        super(BaseVehicle, self).destroy()
        self.detachDyWld(self.vehicle)
        self.origin = None
        self.vehicle = None
        self.chassis = None
        self.wheels = None

    def set_velocity(self, velocity):
        super(BaseVehicle, self).set_velocity(velocity)
        self.last_velocity = self.velocity

    def set_position(self, position : List[float], height=None):
        if height is None:
            height = self.position[-1]
        if len(position) == 2:
            position.append(height)
        super(BaseVehicle, self).set_position(position)
        self.last_position = self.position

    def set_heading_theta(self, heading):
        super(BaseVehicle, self).set_heading_theta(heading)
        self.last_heading = self.heading_theta

    def get_state(self):
        """
        Fetch more information
        """
        state = super(BaseVehicle, self).get_state()
        state.update(
            {
                "steering": self.steering,
                "throttle_brake": self.throttle_brake,
                "crash_vehicle": self.crash_vehicle,
                "crash_object": self.crash_object,
                "crash_building": self.crash_building,
                "crash_sidewalk": self.crash_sidewalk,
                "size": (self.LENGTH, self.WIDTH, self.HEIGHT),
                "length": self.LENGTH,
                "width": self.WIDTH,
                "height": self.HEIGHT,
            }
        )
        if self.navigation is not None:
            state.update(self.navigation.get_state())
        return state

    # def get_raw_state(self):
    #     ret = dict(position=self.position, heading=self.heading, velocity=self.velocity)
    #     return ret

    def get_dynamics_parameters(self):
        # These two can be changed on the fly
        max_engine_force = self.config["max_engine_force"]
        max_brake_force = self.config["max_brake_force"]

        # These two can only be changed in init
        wheel_friction = self.config["wheel_friction"]
        assert self.max_steering == self.config["max_steering"]
        max_steering = self.max_steering

        mass = self.config["mass"] if self.config["mass"] else self.MASS

        ret = dict(
            max_engine_force=max_engine_force,
            max_brake_force=max_brake_force,
            wheel_friction=wheel_friction,
            max_steering=max_steering,
            mass=mass
        )
        return ret

    def _update_overtake_stat(self):
        return {"overtake_vehicle_num": 0}

    def __del__(self):
        super(BaseVehicle, self).__del__()
        # self.engine = None
        self.navigation = None
        self.wheels = None

    @property
    def reference_lanes(self):
        return self.navigation.current_ref_lanes

    @property
    def overspeed(self):
        return True if self.lane.speed_limit < self.speed_km_h else False

    @property
    def replay_done(self):
        return self._replay_done if hasattr(self, "_replay_done") else (
            self.crash_building or self.crash_vehicle or
            # self.on_white_continuous_line or
            self.on_yellow_continuous_line
        )

    @property
    def current_action(self):
        return self.last_current_action[-1]

    @property
    def last_action(self):
        return self.last_current_action[0]

    @property
    def max_speed_km_h(self):
        return self.config["max_speed_km_h"]

    @property
    def max_speed_m_s(self):
        return self.config["max_speed_km_h"] / 3.6

    @property
    def roll(self):
        """
        Return the roll of this object
        """
        return np.deg2rad(self.chassis.getR())

    def set_roll(self, roll):
        self.chassis.setR(roll)

    @property
    def pitch(self):
        """
        Return the pitch of this object
        """
        return np.deg2rad(self.chassis.getP())

    def set_pitch(self, pitch):
        self.chassis.setP(pitch)

