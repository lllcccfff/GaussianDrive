from panda3d.core import Loader, NodePath
from panda3d.bullet import (
    BulletTriangleMesh,
    BulletTriangleMeshShape,
)
from metadrive.constants import MetaDriveType, CollisionGroup

from metadrive.base_class.base_object import BaseObject
from metadrive.constants import CollisionGroup
from metadrive.engine.physics_node import BaseRigidBodyNode


class MeshTerrain(BaseObject):
    COLLISION_MASK = CollisionGroup.TrafficParticipants
    MASS = 0.0  # 静态地面

    def __init__(
        self,
        physics_world,
        model_path: str,
        position=(0, 0, 0),
        scale=1.0,
        friction=0.8,
        restitution=0.0,
        random_seed=None,
        name="GroundMesh",
        config=None,
        **kwargs,
    ):
        super().__init__(physics_world=physics_world, random_seed=random_seed, name=name, config=config)

        self.set_metadrive_type(MetaDriveType.GROUND)

        # load model
        if not (model_path.endswith(".bam") or model_path.endswith(".egg") or model_path.endswith(".obj")):
            raise ValueError("Only .bam, .egg, .obj models are supported for MeshTerrain.")
        loader = Loader.getGlobalPtr()
        model = NodePath(loader.loadSync(model_path))
        model.setPos(position[0], position[1], position[2])
        model.setScale(scale)

        # build bullet shape
        bullet_mesh = BulletTriangleMesh()
        for nodePath in model.findAllMatches("**/+GeomNode"):
            geom_node = nodePath.node()
            for i in range(geom_node.getNumGeoms()):
                geom = geom_node.getGeom(i)
                transform = nodePath.getNetTransform().getMat()
                bullet_mesh.addGeom(geom, transform)

        shape = BulletTriangleMeshShape(bullet_mesh, dynamic=False)
        shape.setMargin(0.05)
        # attach
        self.body = BaseRigidBodyNode(self.name, MetaDriveType.GROUND, self.MASS)
        self.body.addShape(shape)
        self.body.setStatic(True)
        self.body.setFriction(friction)
        self.body.setRestitution(restitution)
        self.attachDyWld()

    def reset(self, random_seed=None, name=None, *args, **kwargs):
        """地面通常无需reset"""
        pass

    def destroy(self):
        self.detachDyWld(self.body)
        super().destroy()
