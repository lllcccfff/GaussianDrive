import os
import json
import torch
import threading

from cuda import cudart
import glfw
import ctypes
import numpy as np
import platform
import OpenGL.GL as gl
from imgui_bundle import imgui, imgui_toggle
from glm import mat4, mat3, mat4x3, vec3
from easydrive.utils.console_utils import *
from metadrive.visualizer.client import Client
from metadrive.visualizer.server import WebSocketServer

import cv2

class Visualizer:
    def __init__(
        self, H, W,
        mode = 'local', # client, server, local
        host = None,
        port = None,

    ):
        self.H = H
        self.W = W
        self.mode = mode
            
        if self.mode == 'client':
            self.lock = threading.Lock()
            self.client = Client(server_ip=host, server_port=port, lock=self.lock)
            self.client.input = self.sensor_controller.send_params()
            self.client.run()
        elif self.mode == 'server':
            self.lock = threading.Lock()
            self.server = WebSocketServer(host=host, port=port, lock=self.lock)
            self.server.run()
            
        self.window_title = 'Simple Visualizer'

        if self.mode in ['local', 'client']:
            self._init_glfw()
            self._init_opengl()
            self._init_imgui()
            self._init_quad()
            self._bind_callbacks()
        
    @property
    def window_address(self):
        window_address = ctypes.cast(self.window, ctypes.c_void_p).value
        return window_address

    def _init_opengl(self):
        gl.glViewport(0, 0, self.W, self.H)    # Use program point size
        # gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

        # # Performs face culling
        # gl.glEnable(gl.GL_CULL_FACE)
        # gl.glCullFace(gl.GL_BACK)

        # # Performs alpha trans testing
        # # gl.glEnable(gl.GL_ALPHA_TEST)
        # try: gl.glEnable(gl.GL_ALPHA_TEST)
        # except gl.GLError as e: pass

        # # Performs z-buffer testing
        # gl.glEnable(gl.GL_DEPTH_TEST)
        # # gl.glDepthMask(gl.GL_TRUE)
        # gl.glDepthFunc(gl.GL_LEQUAL)
        # # gl.glDepthRange(-1.0, 1.0)
        # gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # # Enable some masking tests
        # gl.glEnable(gl.GL_SCISSOR_TEST)

        # # Enable this to correctly render points
        # # https://community.khronos.org/t/gl-point-sprite-gone-in-3-2/59310
        # # gl.glEnable(gl.GL_POINT_SPRITE)  # MARK: ONLY SPRITE IS WORKING FOR NOW
        # try: gl.glEnable(gl.GL_POINT_SPRITE)  # MARK: ONLY SPRITE IS WORKING FOR NOW
        # except gl.GLError as e: pass
        # # gl.glEnable(gl.GL_POINT_SMOOTH) # MARK: ONLY SPRITE IS WORKING FOR NOW

        # # # Configure how we store the pixels in memory for our subsequent reading of the FBO to store the rendering into memory.
        # # # The second argument specifies that our pixels will be in bytes.
        # # gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)

    def _init_glfw(self):
        if not glfw.init():
            log(red('Could not initialize OpenGL context'))
            exit(1)

        # Decide GL+GLSL versions
        # GL 3.3 + GLSL 330
        # self.glsl_version = '#version 330'
        # glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        # glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        # glfw.window_hint(glfw.OPENGL_PROFILE, glfw.GLFW_OPENGL_CORE_PROFILE)  # // 3.2+ only
        # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, 1)  # 1 is gl.GL_TRUE

        if platform.system() == "Darwin":
            self.glsl_version = "#version 150"
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)  # // 3.2+ only
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, 1)
            glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, 0)  # disable osx scaling
        else:
            # GL 3.0 + GLSL 130
            self.glsl_version = "#version 130" # TODO: why? why not 330?
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
            # glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE) # // 3.2+ only
            # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        # glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, 1);

        # Create a windowed mode window and its OpenGL context
        window = glfw.create_window(self.W, self.H, self.window_title, None, None)
        if not window:
            glfw.terminate()
            log(red('Could not initialize window'))
            raise RuntimeError('Failed to initialize window in glfw')

        # Setting up the window
        glfw.make_context_current(window)
        glfw.swap_interval(self.cfg.use_vsync)  # disable vsync
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

        # TODO: set icon
        # icon = load_image(self.icon_file)
        # pixels = (icon * 255).astype(np.uint8)
        # height, width = icon.shape[:2]
        # glfw.set_window_icon(window, 1, [width, height, pixels])  # set icon for the window

        self.window = window

    def _init_imgui(self):
        imgui.create_context()
        self.io = imgui.get_io()

        # io.config_flags |= imgui.ConfigFlags_.nav_enable_keyboard  # Enable Keyboard Controls # NOTE: This will make imgui always want to capture keyboard
        # io.config_flags |= imgui.ConfigFlags_.nav_enable_gamepad # Enable Gamepad Controls
        self.io.config_flags |= imgui.ConfigFlags_.docking_enable  # Enable docking
        # io.config_flags |= imgui.ConfigFlags_.viewports_enable # Enable Multi-Viewport / Platform Windows
        # io.config_viewports_no_auto_merge = True
        # io.config_viewports_no_task_bar_icon = True

        # Setup Dear ImGui style
        imgui.style_colors_dark()
        # imgui.style_colors_classic()

        # When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
        style = imgui.get_style()
        style.tab_rounding = 4.0
        style.grab_rounding = 4.0
        style.child_rounding = 4.0
        style.frame_rounding = 4.0
        style.popup_rounding = 8.0
        style.window_rounding = 8.0
        style.scrollbar_rounding = 4.0
        window_bg_color = style.color_(imgui.Col_.window_bg)
        window_bg_color.w = 1.0
        style.set_color_(imgui.Col_.window_bg, window_bg_color)

        # You need to transfer the window address to imgui.backends.glfw_init_for_opengl
        # proceed as shown below to get it.
        imgui.backends.glfw_init_for_open_gl(self.window_address, True)
        imgui.backends.opengl3_init(self.glsl_version)

        io = imgui.get_io()
        self.default_font = io.fonts.add_font_from_file_ttf(self.font_default, self.font_size)
        self.italic_font = io.fonts.add_font_from_file_ttf(self.font_italic, self.font_size)
        self.bold_font = io.fonts.add_font_from_file_ttf(self.font_bold, self.font_size)
        io.fonts.build()

        # # Markdown initialization
        # options = imgui_md.MarkdownOptions()
        # # options.font_options.font_base_path = 'assets/fonts'
        # options.font_options.regular_size = self.font_size
        # imgui_md.initialize_markdown(options=options)
        # imgui_md.get_font_loader_function()() # requires imgui_hello
    
    def _init_quad(self):
        from easydrive.utils.opengl_utils import Quad
        self.quad = Quad(H=self.H, W=self.W)  # will blit this texture to screen if rendered

    def _bind_callbacks(self):
        glfw.set_window_user_pointer(self.window, self)  # set the user, for retrival

    def is_running(self):
        return not glfw.window_should_close(self.window) if self.mode in ['client', 'local'] else True
    
    def run(self, img : Union[np.ndarray, torch.Tensor] = None):
        if self.mode == 'server':
            return self._actuate_server(img)
        
        if self.mode == 'client':
            img = self._actuate_client(img)
        
        self._render(img)
                
    def _render(self, img):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        glfw.poll_events()

        imgui.backends.opengl3_new_frame()
        imgui.backends.glfw_new_frame()
        imgui.new_frame()

        # self.get_fps_and_frame_time()
        # self.get_device_and_memory()

        if img:
            self.quad.copy_to_texture(img)
            self.quad.draw()

        imgui.render()
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())
        
        glfw.swap_buffers(self.window)

    def _actuate_client(self):
        inputs = None
        with self.lock:
            self.client.input = inputs

        with self.lock:
            outputs = self.client.output
        return outputs
    
    def _actuate_server(self, img):
        with self.lock:
            inputs = self.server.input
        
        output = img
        with self.lock:
            # self.server.output = output.to('cpu', non_blocking=True)  # initiate async copy
            self.server.output = output  # initiate async copy


    def shutdown(self):
        imgui.backends.opengl3_shutdown()
        imgui.backends.glfw_shutdown()
        imgui.destroy_context()

        glfw.destroy_window(self.window)
        glfw.terminate()

    # def get_fps_and_frame_time(self):
    #     first_run = 'last_fps_update' not in self.static
    #     curr_time = time.perf_counter()
    #     if first_run:
    #         self.static.last_fps_update = curr_time
    #         self.static.frame_time = 1
    #         self.static.fps = 1
    #         self.static.acc_frame = 1
    #     elif curr_time - self.static.last_fps_update > self.update_fps_time:
    #         self.static.frame_time = (curr_time - self.static.last_fps_update) / self.static.acc_frame
    #         self.static.fps = 1 / self.static.frame_time  # in fps
    #         self.static.last_fps_update = curr_time
    #         self.static.acc_frame = 1
    #     else:
    #         self.static.acc_frame += 1
    #     return self.static.fps, self.static.frame_time

    # def get_device_and_memory(self):
    #     first_run = 'last_memory_update' not in self.static
    #     curr_time = time.perf_counter()
    #     if first_run or curr_time - self.static.last_memory_update > self.update_mem_time:
    #         try:
    #             self.static.name = torch.cuda.get_device_name()
    #             self.static.device = torch.cuda.current_device()
    #             self.static.memory = torch.cuda.max_memory_allocated()
    #         except:
    #             self.static.name = 'Unsupported'
    #             self.static.device = 'Unsupported'
    #             self.static.memory = -1
    #         self.static.last_memory_update = curr_time
    #     return self.static.name, self.static.device, self.static.memory
