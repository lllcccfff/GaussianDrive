import math
import os
from collections import deque
from typing import Union, Optional, List

import numpy as np
from panda3d.bullet import BulletVehicle, BulletBoxShape, ZUp
from panda3d.core import Material, Vec3, TransformState

from metadrive.base_class.base_object import BaseObject
# from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.component.pg_space import VehicleParameterSpace, ParameterSpace
from metadrive.constants import CamMask, get_color_palette
from metadrive.constants import MetaDriveType, CollisionGroup
from metadrive.constants import Semantics
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.logger import get_logger
from metadrive.engine.physics_node import BaseRigidBodyNode
from metadrive.utils import Config, safe_clip_for_small_array
from metadrive.utils.math import get_vertical_vector, norm, clip
from metadrive.utils.math import wrap_to_pi
from metadrive.utils.utils import get_object_from_node
import torch
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
        self.crash_world = False

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
        config: Union[dict, Config],
        physics_world,
        size=None,
        name: str = None,
        random_seed=None,
        position=None,
        heading_theta=None,
        _calling_reset=True,
        **kwargs
    ):
        """
        This Vehicle Config is different from self.get_config(), and it is used to define which modules to use, and
        module parameters. And self.physics_config defines the physics feature of vehicles, such as length/width
        :param vehicle_config: mostly, vehicle module config
        :param random_seed: int
        """
        # check
        assert config is not None, "Please specify the vehicle config."

        # NOTE: it is the game engine, not vehicle drivetrain
        # self.engine = get_engine()

        if size is None:
            size = (self.DEFAULT_LENGTH, self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        BaseObject.__init__(self, physics_world, size, name, random_seed, config)
        BaseVehicleState.__init__(self)
        self.set_metadrive_type(MetaDriveType.VEHICLE)

        # build vehicle physics model
        self.vehicle, self.body = self._create_vehicle_chassis()
        self.wheels = self._create_wheel()
    
        # powertrain config
        self.enable_reverse = self.config["enable_reverse"]
        self.max_steering = self.config["max_steering"]
        self.max_engine_force = self.config["max_engine_force"]
        self.max_brake_force = self.config["max_brake_force"]

        # state info
        self.throttle_brake = 0.0
        self.steering = 0
        self.last_current_action = deque([(0.0, 0.0), (0.0, 0.0)], maxlen=2)
        self.last_position = (0, 0)
        self.last_velocity = 0
        self.last_heading = 0
        self.dist_to_left_side = None
        self.dist_to_right_side = None

        # VehicleFeedback state variables
        self._brake_pedal_position = 0.0  # 0-100
        self._accelerator_pedal_position = 0.0  # 0-100

        # step info
        self.out_of_route = None
        self.on_lane = None
        self._init_step_info()

        #
        self.break_down = False
        # if self.engine.current_map is not None:
        if _calling_reset:
            self.reset(position=position, heading_theta=heading_theta, vehicle_config=config, **kwargs)

    def _init_step_info(self):
        # done info will be initialized every frame
        self.init_state_info()
        self.out_of_route = False  # re-route is required if is false
        self.on_lane = True  # on lane surface or not

    @staticmethod
    def _preprocess_action(action):
        action = safe_clip_for_small_array(action, -1, 1)
        return action, {'raw_action': (action[0], action[1])}

    def attachDyWld(self):
        self.physics_world.dynamic_world.attach(self.body)
        self.physics_world.dynamic_world.attach(self.vehicle)

    def detachDyWld(self):
        self.physics_world.dynamic_world.remove(self.vehicle)
        self.physics_world.dynamic_world.remove(self.body)

    def reset(
        self,
        name=None,
        random_seed=None,
        position: np.ndarray = None,
        heading_theta: float = 0.0,
        velocity: np.ndarray = None,
        angluar_velocity: float = 0.0,
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


        self.set_heading_theta(heading_theta)
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

        if self.config["spawn_velocity"]:
            self.set_velocity(velocity)
            self.set_angular_velocity(angluar_velocity)

        # self.add_light()

    def move(self, action=None, state_info=None):
        """
        Save info and make decision before action
        """
        # init step info to store info before each step
        # if action is None:
        #     action = [0, 0]

        self._init_step_info()
        self.last_position = self.position  # 2D vector
        self.last_velocity = self.velocity  # 2D vector
        self.last_heading_theta = self.heading_theta

        if state_info:
            self.set_transform(state_info["transform"])

            self.set_velocity(state_info["velocity"])
            self.set_angular_velocity(state_info["angular_velocity"])
            step_info = None
        else:
            action, step_info = self._preprocess_action(action)
            self.last_current_action.append(action)  # the real step of physics world is implemented in taskMgr.step()
            # if self.increment_steering:
            #     self._set_incremental_action(action)
            # else:
            self._set_action(action)
        return step_info

    def after_step(self):
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
    
    def crash_check(self):
        """
        Check States and filter to update info
        """
        # result_1 = self.physics_world.static_world.contactTest(self.body, True)
        result_2 = self.physics_world.dynamic_world.contactTest(self.body, False)
        contact_infos = set()
        ground_contact = list()
        for contact in result_2.getContacts():
            node0 = contact.getNode0()
            node1 = contact.getNode1()
            name = node1.getName()
            if name == MetaDriveType.VEHICLE:
                self.crash_vehicle = True
            elif name in [MetaDriveType.PEDESTRIAN, MetaDriveType.CYCLIST]:
                self.crash_human = True
            elif name == MetaDriveType.GROUND:
                maniP = contact.getManifoldPoint()
                pos = maniP.getPositionWorldOnB()
                ground_contact.append(torch.tensor([pos.x, pos.y, pos.z]))
            else:
                continue
            contact_infos.add(name)
        
        if len(ground_contact) > 0:
            ground_contact = torch.stack(ground_contact).cuda().float()
            if self._is_crash_world(ground_contact):
                self.crash_world = True

        self.contact_results.update(contact_infos)

    def _is_crash_world(self, contact_points):
        wheel_centers = []
        for i in range(self.vehicle.getNumWheels()):
            wheel = self.vehicle.getWheel(i)
            wheel_center = wheel.getWorldTransform().getRow3(3)  
            wheel_center = torch.tensor([wheel_center.x, wheel_center.y, wheel_center.z])
            wheel_centers.append(wheel_center)
        wheel_centers = torch.stack(wheel_centers).cuda().float()

        diff = contact_points[:, :2].unsqueeze(1) - wheel_centers[:, :2].unsqueeze(0)  # [N,4,3]
        dist = diff.norm(dim=-1)                              # [N,4]
        nearest_idx = dist.argmin(dim=1)                                 # [N]
        nearest_z = wheel_centers[nearest_idx, 2]                        # [N]

        if (contact_points[:, 2] - nearest_z > 0).any().item():
            return True
        return False

    """------------------------------------------- act -------------------------------------------------"""

    def set_steering(self, steering):
        steering = float(steering)
        self.vehicle.setSteeringValue(steering, 0)
        self.vehicle.setSteeringValue(steering, 1)
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
        self.vehicle.setSteeringValue(self.steering * self.max_steering, 0)
        self.vehicle.setSteeringValue(self.steering * self.max_steering, 1)
        self._apply_throttle_brake(action[1])
        
    def _apply_throttle_brake(self, throttle_brake):
        for wheel_index in range(4):
            if throttle_brake >= 0:
                self.vehicle.setBrake(2.0, wheel_index)
                if self.speed_km_h > self.max_speed_km_h:
                    self.vehicle.applyEngineForce(0.0, wheel_index)
                    self._accelerator_pedal_position = 0.0
                else:
                    self.vehicle.applyEngineForce(self.max_engine_force * throttle_brake, wheel_index)
                    self._accelerator_pedal_position = throttle_brake * 100.0
                self._brake_pedal_position = 0.0
            else:
                if self.enable_reverse:
                    self.vehicle.applyEngineForce(self.max_engine_force * throttle_brake, wheel_index)
                    self.vehicle.setBrake(0, wheel_index)
                    self._accelerator_pedal_position = abs(throttle_brake) * 100.0
                    self._brake_pedal_position = 0.0
                else:
                    DEADZONE = 0.01

                    # Speed m/s in car's heading:
                    heading = self.heading
                    velocity = self.velocity
                    speed_in_heading = velocity[0] * heading[0] + velocity[1] * heading[1]

                    if speed_in_heading < DEADZONE:
                        self.vehicle.applyEngineForce(0.0, wheel_index)
                        self.vehicle.setBrake(2, wheel_index)
                        self._accelerator_pedal_position = 0.0
                        self._brake_pedal_position = 100.0
                    else:
                        self.vehicle.applyEngineForce(0.0, wheel_index)
                        self.vehicle.setBrake(abs(throttle_brake) * self.max_brake_force, wheel_index)
                        self._accelerator_pedal_position = 0.0
                        self._brake_pedal_position = abs(throttle_brake) * 100.0

    """---------------------------------------- vehicle info ----------------------------------------------"""

    def update_dist_to_left_right(self):
        self.dist_to_left_side, self.dist_to_right_side = 0, 0

    """---------------------------------------- some math tool ----------------------------------------------"""

    """-------------------------------------- for vehicle making ------------------------------------------"""

    def _create_vehicle_chassis(self):

        # assert self.LENGTH < BaseVehicle.MAX_LENGTH, "Vehicle is too large!"
        # assert self.WIDTH < BaseVehicle.MAX_WIDTH, "Vehicle is too large!"

        chassis = BaseRigidBodyNode(self.name, MetaDriveType.VEHICLE, self.MASS)

        chassis_shape = BulletBoxShape(Vec3(self.WIDTH / 2, self.LENGTH / 2, self.HEIGHT / 2))
        chassis_shape.setMargin(0.03)
        ts = TransformState.makePos(Vec3(0, 0, self.TIRE_RADIUS))
        chassis.addShape(chassis_shape, ts)
        chassis.setDeactivationEnabled(False)
        chassis.notifyCollisions(True)  # advance collision check, do callback in pg_collision_callback

        vehicle_chassis = BulletVehicle(self.physics_world.dynamic_world, chassis)
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
            wheel.setSuspensionStiffness(50)
            wheel.setWheelsDampingRelaxation(4.8)
            wheel.setWheelsDampingCompression(3.2)
            wheel_friction = self.config["wheel_friction"]
            wheel.setFrictionSlip(0.5)
            wheel.setRollInfluence(0.5)
            wheels.append(wheel)
        return wheels

    def destroy(self):
        super(BaseVehicle, self).destroy()
        self.detachDyWld()
        self.origin = None
        self.vehicle = None
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

    # ===== VehicleFeedback API =====
    # These methods provide vehicle state feedback matching the VehicleFeedback protocol

    def get_steering_wheel_angle(self):
        """
        Get current steering wheel angle in radians.
        Returns the normalized steering value multiplied by max_steering.
        """
        return self.steering * self.max_steering * (np.pi / 180.0)  # Convert to radians

    def get_vehicle_speed(self):
        """
        Get current vehicle speed in km/h.
        """
        return self.speed_km_h

    def get_front_left_wheel_speed(self):
        """
        Get front left wheel speed in m/s.
        Calculated from wheel rotation speed and wheel radius.
        """
        if len(self.wheels) < 1:
            return 0.0
        wheel = self.wheels[0]  # Front left wheel (index 0)
        rotation_speed = wheel.getDeltaRotation()  # rad/s
        wheel_radius = wheel.getWheelRadius()  # meters
        return rotation_speed * wheel_radius  # m/s

    def get_front_right_wheel_speed(self):
        """
        Get front right wheel speed in m/s.
        """
        if len(self.wheels) < 2:
            return 0.0
        wheel = self.wheels[1]  # Front right wheel (index 1)
        rotation_speed = wheel.getDeltaRotation()
        wheel_radius = wheel.getWheelRadius()
        return rotation_speed * wheel_radius

    def get_rear_left_wheel_speed(self):
        """
        Get rear left wheel speed in m/s.
        """
        if len(self.wheels) < 3:
            return 0.0
        wheel = self.wheels[2]  # Rear left wheel (index 2)
        rotation_speed = wheel.getDeltaRotation()
        wheel_radius = wheel.getWheelRadius()
        return rotation_speed * wheel_radius

    def get_rear_right_wheel_speed(self):
        """
        Get rear right wheel speed in m/s.
        """
        if len(self.wheels) < 4:
            return 0.0
        wheel = self.wheels[3]  # Rear right wheel (index 3)
        rotation_speed = wheel.getDeltaRotation()
        wheel_radius = wheel.getWheelRadius()
        return rotation_speed * wheel_radius

    def get_brake_pedal_position(self):
        """
        Get current brake pedal position (0-100).
        Returns the recorded brake pedal position.
        """
        return self._brake_pedal_position

    def get_accelerator_pedal_position(self):
        """
        Get current accelerator pedal position (0-100).
        Returns the recorded accelerator pedal position.
        """
        return self._accelerator_pedal_position

    def get_steering_wheel_speed(self):
        """
        Get steering wheel angular velocity in rad/s.
        Uses the chassis angular velocity around Z-axis.
        """
        angular_velocity = self.body.getAngularVelocity()
        return angular_velocity[2]  # Z-axis angular velocity in rad/s

    def get_longitudinal_acceleration(self):
        """
        Get longitudinal acceleration in m/s^2.
        Calculated from velocity change.
        """
        current_velocity = self.speed
        acceleration = (current_velocity - self.last_velocity) / (self.physics_world.physics_world_step_size / 1e6)
        return acceleration

    def get_left_directive_wheel_angle(self):
        """
        Get left front wheel steering angle in radians.
        """
        if len(self.wheels) < 1:
            return 0.0
        return self.wheels[0].getSteering() * (np.pi / 180.0)  # Convert to radians

    def get_right_directive_wheel_angle(self):
        """
        Get right front wheel steering angle in radians.
        """
        if len(self.wheels) < 2:
            return 0.0
        return self.wheels[1].getSteering() * (np.pi / 180.0)  # Convert to radians


