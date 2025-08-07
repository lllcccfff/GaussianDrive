import os
import gc
import sys
import time
from typing import Optional, Union, Tuple
import gltf
from panda3d.core import AntialiasAttrib, loadPrcFileData, LineSegs, PythonCallbackObject, Vec3, NodePath
from metadrive.third_party.simplepbr import init

from metadrive.component.sensors.base_sensor import BaseSensor
from metadrive.constants import RENDER_MODE_OFFSCREEN, RENDER_MODE_NONE, RENDER_MODE_ONSCREEN, EDITION, CamMask, \
    BKG_COLOR
from metadrive.engine.asset_loader import initialize_asset_loader, close_asset_loader, randomize_cover, get_logo_file
from metadrive.engine.core.collision_callback import collision_callback
from metadrive.engine.core.draw import ColorLineNodePath, ColorSphereNodePath
from metadrive.engine.core.force_fps import ForceFPS
from metadrive.engine.core.onscreen_message import ScreenMessage
from metadrive.engine.core.physics_world import PhysicsWorld
from metadrive.engine.core.pssm import PSSM
from metadrive.engine.core.sky_box import SkyBox
from metadrive.engine.core.terrain import Terrain
from metadrive.engine.logger import get_logger
from metadrive.utils.utils import is_mac, setup_logger
import logging
import subprocess
from metadrive.utils.utils import is_port_occupied

logger = get_logger()


def _suppress_warning():
    loadPrcFileData("", "notify-level-glgsg fatal")
    loadPrcFileData("", "notify-level-pgraph fatal")
    loadPrcFileData("", "notify-level-pnmimage fatal")
    loadPrcFileData("", "notify-level-task fatal")
    loadPrcFileData("", "notify-level-thread fatal")
    loadPrcFileData("", "notify-level-device fatal")
    loadPrcFileData("", "notify-level-bullet fatal")
    loadPrcFileData("", "notify-level-display fatal")
    logging.getLogger('shapely.geos').setLevel(logging.CRITICAL)


# def _free_warning():
#     loadPrcFileData("", "notify-level-glgsg debug")
#     # loadPrcFileData("", "notify-level-pgraph debug")  # press 4 to use toggle analyze to do this
#     loadPrcFileData("", "notify-level-display debug")  # press 4 to use toggle analyze to do this
#     loadPrcFileData("", "notify-level-pnmimage debug")
#     loadPrcFileData("", "notify-level-thread debug")


# def attach_cover_image(window_width, window_height):
#     cover_file_path = randomize_cover()
#     image = OnscreenImage(image=cover_file_path, pos=(0, 0, 0), scale=(1, 1, 1))
#     if window_width > window_height:
#         scale = window_width / window_height
#     else:
#         scale = window_height / window_width
#     image.set_scale((scale, 1, scale))
#     image.setTransparency(True)
#     return image


# def attach_logo(engine):
#     cover_file_path = get_logo_file()
#     image = OnscreenImage(image=cover_file_path)
#     scale = 0.075
#     image.set_scale((scale * 3, 1, scale))
#     image.set_pos((0.8325 * engine.w_scale, 0, -0.94 * engine.h_scale))
#     image.set_antialias(AntialiasAttrib.MMultisample)
#     image.setTransparency(True)
#     return image


class EngineCore:
    DEBUG = False
    global_config = None

    def __init__(self, global_config):
        # if EngineCore.global_config is not None:
        #     # assert global_config is EngineCore.global_config, \
        #     #     "No allowed to change ptr of global config, which may cause issue"
        #     pass
        # else:
        config = global_config
        self.main_window_disabled = False

        self.pid = os.getpid()
        EngineCore.global_config = global_config
        self.mode = global_config["_render_mode"]
        self.pstats_process = None

        # Setup onscreen render
        if self.global_config["use_render"]:
            assert self.mode == RENDER_MODE_ONSCREEN, "Render mode error"
            # Warning it may cause memory leak, Pand3d Official has fixed this in their master branch.
            # You can enable it if your panda version is latest.
        else:
            self.global_config["show_coordinates"] = False
            if self.global_config["image_observation"]:
                assert self.mode == RENDER_MODE_OFFSCREEN, "Render mode error"
            else:
                assert self.mode == RENDER_MODE_NONE, "Render mode error"
                if self.global_config["show_interface"]:
                    # Disable useless camera capturing in none mode
                    self.global_config["show_interface"] = False

        # Setup some debug options
        # if self.global_config["headless_machine_render"]:
        #     # headless machine support
        #     # loadPrcFileData("", "load-display  pandagles2")
        #     # no further actions will be applied now!
        #     pass
        if self.global_config["debug"]:
            # debug setting
            # EngineCore.DEBUG = True
            # if self.global_config["debug_panda3d"]:
            #     _free_warning()
            # setup_logger(debug=True)
            # self.accept("1", self.toggleDebug)
            # self.accept("2", self.toggleWireframe)
            # self.accept("3", self.toggleTexture)
            # self.accept("4", self.toggleAnalyze)
            # self.accept("5", self.reload_shader)
            pass

        else:
            # only report fatal error when debug is False
            _suppress_warning()
            # a special debug mode
            if self.global_config["debug_physics_world"]:
                self.accept("1", self.toggleDebug)
                self.accept("4", self.toggleAnalyze)

        super(EngineCore, self).__init__(windowType=self.mode)

        # if not self.global_config["debug_physics_world"] \
        #         and (self.mode in [RENDER_MODE_ONSCREEN, RENDER_MODE_OFFSCREEN]):
        #     initialize_asset_loader(self)

        #     if not self.use_render_pipeline:
        #         # Display logo
        #         if self.mode == RENDER_MODE_ONSCREEN and (not self.global_config["debug"]):
        #             if self.global_config["show_logo"]:
        #                 self._window_logo = attach_logo(self)
        #                 self._loading_logo = attach_cover_image(
        #                     window_width=self.get_size()[0], window_height=self.get_size()[1]
        #                 )
        #                 for i in range(5):
        #                     self.graphicsEngine.renderFrame()
        #                 self.taskMgr.add(self.remove_logo, "remove _loading_logo in first frame")

        self.closed = False

        # physics world
        self.physics_world = PhysicsWorld(
            self.global_config["debug_static_world"], disable_collision=self.global_config["disable_collision"]
        )

        # collision callback
        self.physics_world.dynamic_world.setContactAddedCallback(PythonCallbackObject(collision_callback))

        # for real time simulation
        self.force_fps = ForceFPS(self)



    def step_physics_world(self):
        dt = self.global_config["physics_world_step_size"]
        self.physics_world.dynamic_world.doPhysics(dt, 1, dt)

    def _debug_mode(self):
        raise NotImplementedError

    def toggleAnalyze(self):
        print(self.physics_world.report_bodies())
        # self.worldNP.ls()

    def report_body_nums(self, task):
        logger.debug(self.physics_world.report_bodies())
        return task.done

    def close_engine(self):
        if hasattr(self, "_window_logo"):
            self._window_logo.removeNode()
        self.terrain.destroy()
        if self.sky_box:
            self.sky_box.destroy()
        self.physics_world.dynamic_world.clearContactAddedCallback()
        self.physics_world.destroy()
        self.destroy()
        close_asset_loader()
        # EngineCore.global_config.clear()
        EngineCore.global_config = None

        import sys
        if sys.version_info >= (3, 0):
            import builtins
        else:
            import __builtin__ as builtins
        if hasattr(builtins, "base"):
            del builtins.base
        from metadrive.base_class.base_object import BaseObject

        def find_all_subclasses(cls):
            """Find all subclasses of a given class, including subclasses of subclasses."""
            subclasses = cls.__subclasses__()
            for subclass in subclasses:
                # Recursively find subclasses of the current subclass
                subclasses.extend(find_all_subclasses(subclass))
            return subclasses

        gc.collect()
        all_classes = find_all_subclasses(BaseObject)
        for cls in all_classes:
            if hasattr(cls, "MODEL"):
                cls.MODEL = None
            elif hasattr(cls, "model_collections"):
                for node_path in cls.model_collections.values():
                    node_path.removeNode()
                cls.model_collections = {}
            elif hasattr(cls, "_MODEL"):
                for node_path in cls._MODEL.values():
                    node_path.cleanup()
                cls._MODEL = {}
            elif hasattr(cls, "TRAFFIC_LIGHT_MODEL"):
                for node_path in cls.TRAFFIC_LIGHT_MODEL.values():
                    node_path.removeNode()
                cls.TRAFFIC_LIGHT_MODEL = {}
        gc.collect()

    def toggle_help_message(self):
        if self.on_screen_message:
            self.on_screen_message.toggle_help_message()

    # def draw_line_2d(self, start_p, end_p, color, thickness: float):
    #     """
    #     Draw line use LineSegs coordinates system. Since a resolution problem is solved, the point on screen should be
    #     described by [horizontal ratio, vertical ratio], each of them are ranged in [-1, 1]
    #     :param start_p: 2d vec
    #     :param end_p: 2d vec
    #     :param color: 4d vec, line color
    #     :param thickness: line thickness
    #     """
    #     line_seg = LineSegs("interface")
    #     line_seg.setColor(*color)
    #     line_seg.moveTo(start_p[0] * self.w_scale, 0, start_p[1] * self.h_scale)
    #     line_seg.drawTo(end_p[0] * self.w_scale, 0, end_p[1] * self.h_scale)
    #     line_seg.setThickness(thickness)
    #     line_np = self.aspect2d.attachNewNode(line_seg.create(False))
    #     # TODO: line_np is not registered.
    #     return line_np

    # def remove_logo(self, task):
    #     alpha = self._loading_logo.getColor()[-1]
    #     if alpha < 0.1:
    #         self._loading_logo.destroy()
    #         self._loading_logo = None
    #         return task.done
    #     else:
    #         new_alpha = alpha - 0.08
    #         self._loading_logo.setColor((1, 1, 1, new_alpha))
    #         return task.cont

    # def _draw_line_3d(self, start_p: Union[Vec3, Tuple], end_p: Union[Vec3, Tuple], color, thickness: float):
    #     """
    #     This API is not official
    #     Args:
    #         start_p:
    #         end_p:
    #         color:
    #         thickness:

    #     Returns:

    #     """
    #     start_p = [*start_p]
    #     end_p = [*end_p]
    #     start_p[1] *= 1
    #     end_p[1] *= 1
    #     line_seg = LineSegs("interface")
    #     line_seg.moveTo(Vec3(*start_p))
    #     line_seg.drawTo(Vec3(*end_p))
    #     line_seg.setThickness(thickness)
    #     np = NodePath(line_seg.create(False))
    #     material = Material()
    #     material.setBaseColor(LVecBase4(*color[:3], 1))
    #     np.setMaterial(material, True)
    #     return np

    # def make_line_drawer(self, parent_node=None, thickness=1.0):
    #     if parent_node is None:
    #         parent_node = self.render
    #     drawer = ColorLineNodePath(parent_node, thickness=thickness)
    #     return drawer

    # def make_point_drawer(self, parent_node=None, scale=1.0):
    #     if parent_node is None:
    #         parent_node = self.render
    #     drawer = ColorSphereNodePath(parent_node, scale=scale)
    #     return drawer

    # def show_coordinates(self):
    #     if len(self.coordinate_line) > 0:
    #         return
    #     # x direction = red
    #     np_x = self._draw_line_3d(Vec3(0, 0, 0.1), Vec3(100, 0, 0.1), color=[1, 0, 0, 1], thickness=3)
    #     np_x.reparentTo(self.render)
    #     # y direction = blue
    #     np_y = self._draw_line_3d(Vec3(0, 0, 0.1), Vec3(0, 50, 0.1), color=[0, 1, 0, 1], thickness=3)
    #     np_y.reparentTo(self.render)

    #     np_y.hide(CamMask.AllOn)
    #     np_y.show(CamMask.MainCam)

    #     np_x.hide(CamMask.AllOn)
    #     np_x.show(CamMask.MainCam)

    #     self.coordinate_line.append(np_x)
    #     self.coordinate_line.append(np_y)

    # def remove_coordinates(self):
    #     for line in self.coordinate_line:
    #         line.detachNode()
    #         line.removeNode()

    # def set_coordinates_indicator_pos(self, pos):
    #     if len(self.coordinate_line) == 0:
    #         return
    #     for line in self.coordinate_line:
    #         line.setPos(pos[0], pos[1], 0)

    # @property
    # def use_render_pipeline(self):
    #     return self.global_config["render_pipeline"] and not self.mode == RENDER_MODE_NONE

    # def reload_shader(self):
    #     if self.render_pipeline is not None:
    #         self.render_pipeline.reload_shaders()

    # def setup_sensors(self):
    #     for sensor_id, sensor_cfg in self.global_config["sensors"].items():
    #         if sensor_id == "main_camera":
    #             # It is added when initializing main_camera
    #             continue
    #         if sensor_id in self.sensors:
    #             raise ValueError("Sensor id {} is duplicated!".format(sensor_id))
    #         cls = sensor_cfg[0]
    #         args = sensor_cfg[1:]
    #         assert issubclass(cls, BaseSensor), "{} is not a subclass of BaseSensor".format(cls.__name__)
    #         if issubclass(cls, ImageBuffer):
    #             self.add_image_sensor(sensor_id, cls, args)
    #         else:
    #             self.sensors[sensor_id] = cls(*args, self)

    # def get_sensor(self, sensor_id):
    #     if sensor_id not in self.sensors:
    #         raise ValueError("Can not get {}, available sensors: {}".format(sensor_id, self.sensors.keys()))
    #     return self.sensors[sensor_id]

    # def add_image_sensor(self, name: str, cls, args):
    #     sensor = cls(*args, engine=self, cuda=self.global_config["image_on_cuda"])
    #     assert isinstance(sensor, ImageBuffer), "This API is for adding image sensor"
    #     self.sensors[name] = sensor
