from typing import Tuple, Sequence
from metadrive.constants import CamMask

from panda3d.core import LVector3
from panda3d.bullet import BulletPlaneShape

from metadrive.base_class.base_object import BaseObject
from metadrive.constants import CollisionGroup
from metadrive.engine.physics_node import BaseRigidBodyNode

LaneIndex = Tuple[str, str, int]


class GroundPlane(BaseObject):
    TYPE_NAME = None
    COLLISION_MASK = CollisionGroup.TrafficParticipants
    HEIGHT = None

    def __init__(
            self,
            direction: Sequence[float], 
            constant: float = 0., 
            random_seed=None, 
            name=None, 
            config=None
        ):
        super(GroundPlane, self).__init__(random_seed=random_seed, name=name, config=config)


        self.body = BaseRigidBodyNode(self.name, self.TYPE_NAME, self.MASS)
        self.body.addShape(BulletPlaneShape(LVector3(*direction), constant))
        self.body.setStatic(True)
        self.body.setFriction(0.8)
        self.attachDyWld()


        self.set_metadrive_type(self.TYPE_NAME)

    def reset(self, random_seed=None, name=None, *args, **kwargs):
        pass