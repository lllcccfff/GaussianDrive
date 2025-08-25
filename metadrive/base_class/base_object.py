import copy
import math
from abc import ABC
from typing import Dict

import numpy as np
from panda3d.core import LVector3, TransformState, LMatrix4

from metadrive.base_class.base_runnable import BaseRunnable
from metadrive.constants import ObjectState
from metadrive.constants import Semantics
from metadrive.engine.logger import get_logger
from metadrive.engine.physics_node import BaseRigidBodyNode, BaseGhostBodyNode
from metadrive.type import MetaDriveType
from metadrive.utils import Vector
from metadrive.utils import random_string
from metadrive.utils.coordinates_shift import panda_vector, metadrive_vector
from metadrive.utils.math import clip
from metadrive.utils.math import norm
from metadrive.utils.math import wrap_to_pi

logger = get_logger()

class BaseObject(BaseRunnable, MetaDriveType, ABC):
    """
    BaseObject is something interacting with game engine. If something is expected to have an body in the world or have
    appearance in the world, it must be a subclass of BaseObject.

    It is created with name/config/randomEngine and can make decision in the world. Besides the random engine can help
    sample some special configs for it ,Properties and parameters in PARAMETER_SPACE of the object are fixed after
    calling __init__().
    """
    MASS = None  # if object has an body, the mass will be set automatically
    COLLISION_MASK = None
    SEMANTIC_LABEL = Semantics.UNLABELED.label

    YFront2X = np.array([
        [ 0., 1.,  0.],
        [ -1.,  0.,  0.],
        [ 0. , 0.,  1.],
    ])
    XFront2Y = np.array([
        [ 0., -1.,  0.],
        [ 1.,  0.,  0.],
        [ 0. , 0.,  1.],
    ]) 
    
    Panda2Real = None

    def __init__(self, size=None, name=None, random_seed=None, config=None, escape_random_seed_assertion=False):
        """
        Config is a static conception, which specified the parameters of one element.
        There parameters doesn't change, such as length of straight road, max speed of one vehicle, etc.
        """
        config = copy.deepcopy(config)
        BaseRunnable.__init__(self, name, random_seed, config)
        MetaDriveType.__init__(self)
        if not escape_random_seed_assertion:
            assert random_seed is not None, "Please assign a random seed for {} class.".format(self.class_name)

        # Following properties are available when this object needs visualization and physics property
        self.body: BaseRigidBodyNode = None

        if size:
            self.LENGTH, self.WIDTH, self.HEIGHT = size

    # @property
    # def z(self):
    #     return self.body.getPos()[-1]

    # def get_z(self):
    #     """Get the z coordinate (height) of the object"""
    #     return self.body.getPos()[-1]

    def destroy(self):
        """
        Fully delete this element and release the memory
        """
        super(BaseObject, self).destroy()

    def set_position(self, position):
        """
        Set this object to a place, the default value is the regular height for red car
        :param position: 2d array or list
        :param height: give a fixed height
        """
        assert len(position) == 3
        self.body.setTransform(self.body.getTransform().setPos(panda_vector(position)))

    @property
    def position(self):
        return self.body.getTransform().getPos()

    def set_heading_theta(self, heading_theta, to_deg=True) -> None:
        """
        Set heading theta for this object
        :param heading_theta: float
        :param in_rad: when set to True, heading theta should be in rad, otherwise, in degree
        """
        h = heading_theta
        if to_deg:
            h = heading_theta * 180 / np.pi
        # Apply panda2ego transform: -90 degrees to convert from ego (+X forward) to panda (+Y forward)
        h_panda = h - 90.0
        cur_hpr = self.body.getTransform().getHpr()
        new_hpr = LVector3(h_panda, cur_hpr[1], cur_hpr[2])
        self.body.setTransform(self.body.getTransform().setHpr(new_hpr))

    @property
    def heading_theta(self):
        """
        Get the heading theta of this object, unit [rad]
        :return:  heading in rad
        """
        h_panda = self.body.getTransform().getHpr()[0]
        # Apply inverse transform: +90 degrees to convert from panda (+Y forward) to ego (+X forward)
        h_ego = h_panda + 90.0
        return wrap_to_pi(h_ego / 180 * np.pi)

    def set_transform(self, m):
        M = m[:3, :3] @ BaseObject.YFront2X
        self.body.setTransform(TransformState.makeMat(LMatrix4(
            M[0, 0], M[1, 0], M[2, 0], m[3, 0],
            M[0, 1], M[1, 1], M[2, 1], m[3, 1],
            M[0, 2], M[1, 2], M[2, 2], m[3, 2],
            m[0, 3], m[1, 3], m[2, 3], m[3, 3]
        )))

    @property
    def transform(self):
        mat = self.body.getTransform().getMat()
        M = np.array([
            [mat[0][0], mat[1][0], mat[2][0], mat[3][0]],
            [mat[0][1], mat[1][1], mat[2][1], mat[3][1]],
            [mat[0][2], mat[1][2], mat[2][2], mat[3][2]],
            [mat[0][3], mat[1][3], mat[2][3], mat[3][3]]
        ], dtype=np.float32)
        M[:3, :3] = M[:3, :3]  @ BaseObject.XFront2Y
        return M

    def set_velocity(self, velocity):
        """
        Set velocity for object including the direction of velocity and the value (speed)
        The direction of velocity will be normalized automatically, value decided its scale
        :param direction: 2d array or list
        :param value: speed [m/s]
        :param in_local_frame: True, apply speed to local fram
        """
        self.body.setLinearVelocity(
            LVector3(velocity[0], velocity[1], self.body.getLinearVelocity()[-1])
        )

    @property
    def velocity(self):
        """
        Velocity, unit: m/s
        """
        velocity = self.body.getLinearVelocity()
        return np.asarray([velocity[0], velocity[1]])

    def set_angular_velocity(self, angular_velocity, in_rad=True):
        if not in_rad:
            angular_velocity = angular_velocity / 180 * np.pi
        self.body.setAngularVelocity(LVector3(0, 0, angular_velocity))
    
    @property
    def angular_velocity(self):
        return self.body.getAngularVelocity()[-1]

    def set_velocity_km_h(self, direction: list, value=None, in_local_frame=False):
        direction = np.array(direction)
        if value is None:
            direction /= 3.6
        else:
            value /= 3.6
        return self.set_velocity(direction, value, in_local_frame)

    @property
    def velocity_km_h(self):
        """
        Velocity, unit: km/h
        """
        return self.velocity * 3.6

    @property
    def speed(self):
        """
        return the speed in m/s
        """
        velocity = self.velocity
        speed = norm(velocity[0], velocity[1])
        return clip(speed, 0.0, 100000.0)

    @property
    def speed_km_h(self):
        """
        km/h
        """
        return self.speed * 3.6


    @property
    def heading(self):
        """
        Heading is a vector = [cos(heading_theta), sin(heading_theta)]
        """
        real_heading = self.heading_theta
        # heading = np.array([math.cos(real_heading), math.sin(real_heading)])
        heading = (math.cos(real_heading), math.sin(real_heading))
        return heading

    # @property
    # def roll(self):
    #     """
    #     Return the roll of this object. As it is facing to x, so roll is pitch
    #     """
    #     return np.deg2rad(self.origin.getP())

    # def set_roll(self, roll):
    #     """
    #     As it is facing to x, so roll is pitch
    #     """
    #     self.origin.setP(roll)

    # @property
    # def pitch(self):
    #     """
    #     Return the pitch of this object, as it is facing to x, so pitch is roll
    #     """
    #     return np.deg2rad(self.origin.getR())

    # def set_pitch(self, pitch):
    #     """As it is facing to x, so pitch is roll"""
    #     self.origin.setR(pitch)

    def get_state(self) -> Dict:
        pos = self.position
        state = {
            ObjectState.POSITION: [pos[0], pos[1], self.get_z()],
            ObjectState.HEADING_THETA: self.heading_theta,
            ObjectState.ROLL: self.roll,
            ObjectState.PITCH: self.pitch,
            ObjectState.VELOCITY: self.velocity,
            ObjectState.TYPE: type(self)
        }
        return state

    def set_state(self, state: Dict):
        self.set_position(state[ObjectState.POSITION])
        self.set_heading_theta(state[ObjectState.HEADING_THETA])
        self.set_pitch(state[ObjectState.PITCH])
        self.set_roll(state[ObjectState.ROLL])
        self.set_velocity(state[ObjectState.VELOCITY])

    def rename(self, new_name):
        super(BaseObject, self).rename(new_name)
        
        physics_node = self.body.getPythonTag(self.body.getName())
        if isinstance(physics_node, BaseGhostBodyNode) or isinstance(physics_node, BaseRigidBodyNode):
            physics_node.rename(new_name)

    def random_rename(self):
        self.rename(random_string())

    def attachDyWld(self, obj=None):
        if not obj:
            self.engine.physics_world.dynamic_world.attach(self.body)
        else:
            self.engine.physics_world.dynamic_world.attach(obj)
    
    def detachDyWld(self, obj=None):
        if not obj:
            self.engine.physics_world.dynamic_world.remove(self.body)
        else:
            self.engine.physics_world.dynamic_world.remove(obj)

    def set_kinematic(self, is_kinematic):
        self.body.setKinematic(is_kinematic)
        if is_kinematic:
            self.body.setActive(not is_kinematic)
            self.body.setStatic(not is_kinematic)