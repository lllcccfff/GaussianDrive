from typing import Tuple, Sequence
from metadrive.constants import CamMask

from panda3d.core import LVector3
from panda3d.bullet import BulletBoxShape

from metadrive.base_class.base_object import BaseObject
from metadrive.constants import CollisionGroup
from metadrive.engine.physics_node import BaseRigidBodyNode

LaneIndex = Tuple[str, str, int]


class BaseTrafficParticipant(BaseObject):
    TYPE_NAME = None
    COLLISION_MASK = CollisionGroup.TrafficParticipants
    HEIGHT = None

    def __init__(
            self,
            size,
            position: Sequence[float], 
            heading_theta: float = 0., 
            random_seed=None, 
            name=None, 
            config=None
        ):
        super(BaseTrafficParticipant, self).__init__(size=size, random_seed=random_seed, name=name, config=config)
        
        self.set_body()

        self.set_position(position)
        self.set_heading_theta(heading_theta)

        self.set_metadrive_type(self.TYPE_NAME)

        assert self.MASS is not None, "No mass for {}".format(self.class_name)
        assert self.TYPE_NAME is not None, "No name for {}".format(self.class_name)

    def reset(self, position: Sequence[float], heading_theta: float = 0., random_seed=None, name=None, *args, **kwargs):
        pass

    def set_body(self):        
        self.body = BaseRigidBodyNode(self.name, self.TYPE_NAME, self.MASS)
        self.body.addShape(BulletBoxShape((self.WIDTH / 2, self.LENGTH / 2, self.HEIGHT / 2)))
        
        self.body.setFriction(0.)
        self.body.setAnisotropicFriction(LVector3(0., 0., 0.))
        self.attachDyWld()

    def get_state(self):
        state = super(BaseTrafficParticipant, self).get_state()
        state.update({
            "length": self.LENGTH,
            "width": self.WIDTH,
            "height": self.HEIGHT,
        })
        return state
    
    def destroy(self):
        super(BaseTrafficParticipant, self).destroy()
        self.detachDyWld(self.body)
        self.body = None