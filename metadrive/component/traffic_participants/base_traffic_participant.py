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
            config,
            physics_world,
            size,
            position: Sequence[float], 
            heading_theta: float = 0., 
            random_seed=None, 
            name=None
        ):
        super(BaseTrafficParticipant, self).__init__(physics_world, size=size, random_seed=random_seed, name=name, config=config)
        
        self.set_body()

        self.set_position(position)
        self.set_heading_theta(heading_theta)

        self.set_metadrive_type(self.TYPE_NAME)

        assert self.MASS is not None, "No mass for {}".format(self.class_name)
        assert self.TYPE_NAME is not None, "No name for {}".format(self.class_name)

    def reset(self, position: Sequence[float], heading_theta: float = 0., random_seed=None, name=None, *args, **kwargs):
        pass
    
    def move(self, state_info):
        self.set_transform(state_info["transform"])

        self.set_velocity(state_info["velocity"])
        self.set_angular_velocity(state_info["angular_velocity"])

    def set_body(self):
        collision_geom = BulletBoxShape((self.WIDTH / 2, self.LENGTH / 2, self.HEIGHT / 2))
        self.body = BaseRigidBodyNode(self.name, self.TYPE_NAME, self.MASS)
        self.body.addShape(collision_geom)
        
        self.body.setFriction(0.)
        self.body.setAnisotropicFriction(LVector3(0., 0., 0.))

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
        self.body = None