import platform

from panda3d.core import LineSegs, NodePath
from panda3d.core import Material, Vec3, LVecBase4

from metadrive.component.pg_space import VehicleParameterSpace, ParameterSpace
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import Semantics
from metadrive.engine.asset_loader import AssetLoader


class DefaultVehicle(BaseVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.DEFAULT_VEHICLE)
    # LENGTH = 4.51
    # WIDTH = 1.852
    # HEIGHT = 1.19
    TIRE_RADIUS = 0.313
    TIRE_WIDTH = 0.25
    LATERAL_TIRE_TO_CENTER = 0.815
    FRONT_WHEELBASE = 1.05234
    REAR_WHEELBASE = 1.4166
    MASS = 1000

    DEFAULT_LENGTH = 4.515  # meters
    DEFAULT_HEIGHT = 1.19  # meters
    DEFAULT_WIDTH = 1.852  # meters


# When using DefaultVehicle as traffic, please use this class.


class TrafficDefaultVehicle(DefaultVehicle):
    pass


class StaticDefaultVehicle(DefaultVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.STATIC_DEFAULT_VEHICLE)


class XLVehicle(BaseVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.XL_VEHICLE)
    # LENGTH = 5.8
    # WIDTH = 2.3
    # HEIGHT = 2.8
    TIRE_RADIUS = 0.37
    TIRE_WIDTH = 0.5
    LATERAL_TIRE_TO_CENTER = 0.931
    REAR_WHEELBASE = 1.075
    FRONT_WHEELBASE = 1.726
    CHASSIS_TO_WHEEL_AXIS = 0.3
    MASS = 1600

    SEMANTIC_LABEL = Semantics.TRUCK.label

    DEFAULT_LENGTH = 5.74  # meters
    DEFAULT_HEIGHT = 2.8  # meters
    DEFAULT_WIDTH = 2.3  # meters

class LVehicle(BaseVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.L_VEHICLE)
    # LENGTH = 4.5
    # WIDTH = 1.86
    # HEIGHT = 1.85
    TIRE_RADIUS = 0.429
    TIRE_WIDTH = 0.35
    LATERAL_TIRE_TO_CENTER = 0.75
    REAR_WHEELBASE = 1.218261
    FRONT_WHEELBASE = 1.5301
    MASS = 1300

    DEFAULT_LENGTH = 4.87  # meters
    DEFAULT_HEIGHT = 1.85  # meters
    DEFAULT_WIDTH = 2.046  # meters



class MVehicle(BaseVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.M_VEHICLE)
    # LENGTH = 4.4
    # WIDTH = 1.85
    # HEIGHT = 1.37
    TIRE_RADIUS = 0.39
    TIRE_WIDTH = 0.3
    LATERAL_TIRE_TO_CENTER = 0.803
    REAR_WHEELBASE = 1.203
    FRONT_WHEELBASE = 1.285
    MASS = 1200

    DEFAULT_LENGTH = 4.6  # meters
    DEFAULT_HEIGHT = 1.37  # meters
    DEFAULT_WIDTH = 1.85  # meters

class SVehicle(BaseVehicle):
    PARAMETER_SPACE = ParameterSpace(VehicleParameterSpace.S_VEHICLE)
    # LENGTH = 4.25
    # WIDTH = 1.7
    # HEIGHT = 1.7
    TIRE_RADIUS = 0.376
    TIRE_WIDTH = 0.25
    LATERAL_TIRE_TO_CENTER = 0.7

    FRONT_WHEELBASE = 1.385
    REAR_WHEELBASE = 1.11

    MASS = 800

    DEFAULT_LENGTH = 4.3  # meters
    DEFAULT_HEIGHT = 1.7  # meters
    DEFAULT_WIDTH = 1.7  # meters

class VaryingDynamicsVehicle(DefaultVehicle):
    @property
    def WIDTH(self):
        return self.config["width"] if self.config["width"] is not None else super(VaryingDynamicsVehicle, self).WIDTH

    @property
    def LENGTH(self):
        return self.config["length"] if self.config["length"] is not None else super(
            VaryingDynamicsVehicle, self
        ).LENGTH

    @property
    def HEIGHT(self):
        return self.config["height"] if self.config["height"] is not None else super(
            VaryingDynamicsVehicle, self
        ).HEIGHT

    @property
    def MASS(self):
        return self.config["mass"] if self.config["mass"] is not None else super(VaryingDynamicsVehicle, self).MASS

    def reset(
        self,
        random_seed=None,
        vehicle_config=None,
        position=None,
        heading: float = 0.0,  # In degree!
        *args,
        **kwargs
    ):

        assert "width" not in self.PARAMETER_SPACE
        assert "height" not in self.PARAMETER_SPACE
        assert "length" not in self.PARAMETER_SPACE
        should_force_reset = False
        if vehicle_config is not None:
            if vehicle_config["width"] is not None and vehicle_config["width"] != self.WIDTH:
                should_force_reset = True
            if vehicle_config["height"] is not None and vehicle_config["height"] != self.HEIGHT:
                should_force_reset = True
            if vehicle_config["length"] is not None and vehicle_config["length"] != self.LENGTH:
                should_force_reset = True
            if "max_engine_force" in vehicle_config and \
                vehicle_config["max_engine_force"] is not None and \
                vehicle_config["max_engine_force"] != self.config["max_engine_force"]:
                should_force_reset = True
            if "max_brake_force" in vehicle_config and \
                vehicle_config["max_brake_force"] is not None and \
                vehicle_config["max_brake_force"] != self.config["max_brake_force"]:
                should_force_reset = True
            if "wheel_friction" in vehicle_config and \
                vehicle_config["wheel_friction"] is not None and \
                vehicle_config["wheel_friction"] != self.config["wheel_friction"]:
                should_force_reset = True
            if "max_steering" in vehicle_config and \
                vehicle_config["max_steering"] is not None and \
                vehicle_config["max_steering"] != self.config["max_steering"]:
                self.max_steering = vehicle_config["max_steering"]
                should_force_reset = True
            if "mass" in vehicle_config and \
                vehicle_config["mass"] is not None and \
                vehicle_config["mass"] != self.config["mass"]:
                should_force_reset = True

        # def process_memory():
        #     import psutil
        #     import os
        #     process = psutil.Process(os.getpid())
        #     mem_info = process.memory_info()
        #     return mem_info.rss
        #
        # cm = process_memory()

        if should_force_reset:
            self.destroy()
            self.__init__(
                vehicle_config=vehicle_config,
                name=self.name,
                random_seed=self.random_seed,
                position=position,
                heading=heading,
                _calling_reset=False
            )

            # lm = process_memory()
            # print("{}:  Reset! Mem Change {:.3f}MB".format("1 Force Re-Init Vehicle", (lm - cm) / 1e6))
            # cm = lm

        assert self.max_steering == self.config["max_steering"]

        ret = super(VaryingDynamicsVehicle, self).reset(
            random_seed=random_seed, vehicle_config=vehicle_config, position=position, heading=heading, *args, **kwargs
        )

        # lm = process_memory()
        # print("{}:  Reset! Mem Change {:.3f}MB".format("2 Force Reset Vehicle", (lm - cm) / 1e6))
        # cm = lm

        return ret


class VaryingDynamicsBoundingBoxVehicle(VaryingDynamicsVehicle):
    def __init__(
        self, vehicle_config: dict = None, name: str = None, random_seed=None, position=None, heading=None, **kwargs
    ):

        # TODO(pzh): The above code is removed for now. How we get BUS label?
        #  vehicle_config has 'width' 'length' and 'height'
        # if vehicle_config["width"] < 0.0:
        #     self.SEMANTIC_LABEL = Semantics.CAR.label
        # else:
        #     self.SEMANTIC_LABEL = Semantics.BUS.label

        super(VaryingDynamicsBoundingBoxVehicle, self).__init__(
            vehicle_config=vehicle_config,
            name=name,
            random_seed=random_seed,
            position=position,
            heading=heading,
            **kwargs
        )

    def _add_visualization(self):
        if self.render:
            path, scale, offset, HPR = self.path

            # PZH: Note that we do not use model_collection as a buffer here.
            # if path not in BaseVehicle.model_collection:

            # PZH: Load a box model and resize it to the vehicle size
            car_model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))

            car_model.setTwoSided(False)
            BaseVehicle.model_collection[path] = car_model
            car_model.setScale((self.WIDTH, self.LENGTH, self.HEIGHT))
            # car_model.setZ(-self.TIRE_RADIUS - self.CHASSIS_TO_WHEEL_AXIS + self.HEIGHT / 2)
            car_model.setZ(0)
            # model default, face to y
            car_model.setHpr(*HPR)
            car_model.instanceTo(self.origin)

            show_contour = self.config["show_contour"] if "show_contour" in self.config else False
            if show_contour:
                # ========== Draw the contour of the bounding box ==========
                # Draw the bottom of the car first
                line_seg = LineSegs("bounding_box_contour1")
                zoffset = car_model.getZ()
                line_seg.setThickness(2)
                line_color = [1.0, 0.0, 0.0]
                out_offset = 0.02
                w = self.WIDTH / 2 + out_offset
                l = self.LENGTH / 2 + out_offset
                h = self.HEIGHT / 2 + out_offset
                line_seg.moveTo(w, l, h + zoffset)
                line_seg.drawTo(-w, l, h + zoffset)
                line_seg.drawTo(-w, l, -h + zoffset)
                line_seg.drawTo(w, l, -h + zoffset)
                line_seg.drawTo(w, l, h + zoffset)
                line_seg.drawTo(-w, l, -h + zoffset)
                line_seg.moveTo(-w, l, h + zoffset)
                line_seg.drawTo(w, l, -h + zoffset)

                line_seg.moveTo(w, -l, h + zoffset)
                line_seg.drawTo(-w, -l, h + zoffset)
                line_seg.drawTo(-w, -l, -h + zoffset)
                line_seg.drawTo(w, -l, -h + zoffset)
                line_seg.drawTo(w, -l, h + zoffset)
                line_seg.moveTo(-w, -l, 0 + zoffset)
                line_seg.drawTo(w, -l, 0 + zoffset)
                line_seg.moveTo(0, -l, h + zoffset)
                line_seg.drawTo(0, -l, -h + zoffset)

                line_seg.moveTo(w, l, h + zoffset)
                line_seg.drawTo(w, -l, h + zoffset)
                line_seg.moveTo(-w, l, h + zoffset)
                line_seg.drawTo(-w, -l, h + zoffset)
                line_seg.moveTo(-w, l, -h + zoffset)
                line_seg.drawTo(-w, -l, -h + zoffset)
                line_seg.moveTo(w, l, -h + zoffset)
                line_seg.drawTo(w, -l, -h + zoffset)
                line_np = NodePath(line_seg.create(True))
                line_material = Material()
                line_material.setBaseColor(LVecBase4(*line_color[:3], 1))
                line_np.setMaterial(line_material, True)
                line_np.reparentTo(self.origin)

            if self.config["random_color"]:
                material = Material()
                material.setBaseColor(
                    (
                        self.panda_color[0] * self.MATERIAL_COLOR_COEFF,
                        self.panda_color[1] * self.MATERIAL_COLOR_COEFF,
                        self.panda_color[2] * self.MATERIAL_COLOR_COEFF, 0.
                    )
                )
                material.setMetallic(self.MATERIAL_METAL_COEFF)
                material.setSpecular(self.MATERIAL_SPECULAR_COLOR)
                material.setRefractiveIndex(1.5)
                material.setRoughness(self.MATERIAL_ROUGHNESS)
                material.setShininess(self.MATERIAL_SHININESS)
                material.setTwoside(False)
                self.origin.setMaterial(material, True)

    def _add_wheel(self, pos: Vec3, radius: float, front: bool, left):
        wheel_np = self.origin.attachNewNode("wheel")
        self._node_path_list.append(wheel_np)

        # PZH: Skip the wheel model
        # if self.render:
        #     model = 'right_tire_front.gltf' if front else 'right_tire_back.gltf'
        #     model_path = AssetLoader.file_path("models", os.path.dirname(self.path[0]), model)
        #     wheel_model = self.loader.loadModel(model_path)
        #     wheel_model.setTwoSided(self.TIRE_TWO_SIDED)
        #     wheel_model.reparentTo(wheel_np)
        #     wheel_model.set_scale(1 * self.TIRE_MODEL_CORRECT if left else -1 * self.TIRE_MODEL_CORRECT)
        wheel = self.system.createWheel()
        wheel.setNode(wheel_np.node())
        wheel.setChassisConnectionPointCs(pos)
        wheel.setFrontWheel(front)
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
        return wheel


def random_vehicle_type(np_random, p=None):
    v_type = {
        "s": SVehicle,
        "m": MVehicle,
        "l": LVehicle,
        "xl": XLVehicle,
        "default": DefaultVehicle,
    }
    if p:
        assert len(p) == len(v_type), \
            "This function only allows to choose a vehicle from 6 types: {}".format(v_type.keys())
    prob = [1 / len(v_type) for _ in range(len(v_type))] if p is None else p
    return v_type[np_random.choice(list(v_type.keys()), p=prob)]


vehicle_type = {
    "s": SVehicle,
    "m": MVehicle,
    "l": LVehicle,
    "xl": XLVehicle,
    "default": DefaultVehicle,
    "static_default": StaticDefaultVehicle,
    "varying_dynamics": VaryingDynamicsVehicle,
    "varying_dynamics_bounding_box": VaryingDynamicsBoundingBoxVehicle,
    "traffic_default": TrafficDefaultVehicle
}

vehicle_class_to_type = inv_map = {v: k for k, v in vehicle_type.items()}
