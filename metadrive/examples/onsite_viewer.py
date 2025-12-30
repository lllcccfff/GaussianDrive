#!/usr/bin/env python3
"""
OnSite Viewer Script - 驾驶模拟器客户端

功能：
- 通过 libMulticastNetwork 连接 OnSite 服务器
- 接收渲染图像并在 GLFW 窗口中显示
- 通过键盘 (WASD) 控制，发送 VehicleControl 消息给 OnSite

用法：
    python -m metadrive.examples.onsite_viewer \
        --config_center 10.11.17.88:52009 \
        --field_id unique_fieldid \
        --net_interface eno2
"""

import argparse
import sys
import time
import platform

import numpy as np
import glfw
import OpenGL.GL as gl

import libMulticastNetwork

from api_reference.chassis.proto.chassis_enums_pb2 import VEHICLE_CONTROL
from api_reference.chassis.proto.chassis_messages_pb2 import VehicleControl
from api_reference.main.proto.messages_pb2 import Notify, ActorPrepare, ActorPrepareResult
from api_reference.main.proto.enums_pb2 import (
    MT_NOTIFY,
    NT_START_TEST,
    NT_ABORT_TEST,
    NT_FINISH_TEST,
    MT_ACTOR_PREPARE,
    MT_ACTOR_PREPARE_RESULT,
)
from get_ip import get_ip_address
from metadrive.viewer.manual_controller import KeyboardController

# ============ 全局状态 ============
recv_prepare = False
start_test = False
session_id = ""
actor_id = "drive_simulator"

# ============ 全局通道 ============
notify_channel = None
cmd_channel = None
prepare_channel = None
image_channel = None

# ============ 常量 ============
MAX_STEERING_RAD = 1.047  # 60度 = π/3


# ============ OnSiteViewer 类 ============
class OnSiteViewer:
    """OnSite 驾驶模拟器客户端 Viewer"""

    def __init__(self, H=720, W=1280):
        self.H = H
        self.W = W
        self.window_title = "OnSite Viewer"
        self.last_image = None

        self._init_glfw()
        self._init_opengl()
        self._init_texture()

    def _init_glfw(self):
        """初始化 GLFW 窗口"""
        if not glfw.init():
            print("Could not initialize OpenGL context")
            sys.exit(1)

        # 根据平台设置 OpenGL 版本
        if platform.system() == "Darwin":
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, 1)
            glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, 0)
        else:
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)

        # 创建窗口
        window = glfw.create_window(self.W, self.H, self.window_title, None, None)
        if not window:
            glfw.terminate()
            print("Could not initialize window")
            sys.exit(1)

        glfw.make_context_current(window)
        glfw.swap_interval(1)  # 启用 vsync
        self.window = window

    def _init_opengl(self):
        """初始化 OpenGL"""
        gl.glViewport(0, 0, self.W, self.H)
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)

    def _init_texture(self):
        """初始化纹理用于显示图像"""
        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def is_running(self):
        """检查窗口是否仍在运行"""
        return not glfw.window_should_close(self.window)

    def render(self, img):
        """渲染图像到窗口"""
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        glfw.poll_events()

        if img is not None:
            self.last_image = img
            self._draw_image(img)
        elif self.last_image is not None:
            self._draw_image(self.last_image)

        glfw.swap_buffers(self.window)

    def _draw_image(self, img):
        """绘制图像到屏幕"""
        if img is None:
            return

        # 确保图像是 uint8 格式
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        h, w = img.shape[:2]

        # 上传纹理
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGB,
            w, h, 0, gl.GL_BGR, gl.GL_UNSIGNED_BYTE, img
        )

        # 启用纹理并绘制全屏四边形
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 1); gl.glVertex2f(-1, -1)
        gl.glTexCoord2f(1, 1); gl.glVertex2f(1, -1)
        gl.glTexCoord2f(1, 0); gl.glVertex2f(1, 1)
        gl.glTexCoord2f(0, 0); gl.glVertex2f(-1, 1)
        gl.glEnd()
        gl.glDisable(gl.GL_TEXTURE_2D)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def shutdown(self):
        """关闭窗口"""
        if self.texture_id:
            gl.glDeleteTextures(1, [self.texture_id])
        glfw.destroy_window(self.window)
        glfw.terminate()


# ============ OnSite 通信函数 ============
def init_channels(args):
    """初始化 OnSite 通道"""
    global notify_channel, cmd_channel, prepare_channel, image_channel

    param = libMulticastNetwork.CreateChannelsParam()
    local_ip = get_ip_address(args.net_interface)

    param.config_center_addr = args.config_center
    param.local_ip = local_ip
    param.net_interface_name = args.net_interface
    param.field_id = args.field_id
    param.log_level = 1
    param.client_name = "drive_simulator"
    param.recv_self_msg = False

    channels = libMulticastNetwork.ChannelPtrVector()
    ret = libMulticastNetwork.create_channels(param, channels)
    if ret:
        print(f"create channels failed, ret: {ret}")
        sys.exit(1)

    channel_map = {}
    for c in channels:
        print(f"message channel name: {c.name()}, id: {c.id()}")
        channel_map[c.name()] = c

    notify_channel = channel_map["notify"]
    cmd_channel = channel_map["vehiclecontrol"]
    prepare_channel = channel_map["prepare"]
    image_channel = channel_map["camera"]

    if not libMulticastNetwork.InitImageDecoder():
        print("image decoder init error")
        sys.exit(1)


def get_prepare():
    """接收 ActorPrepare 消息"""
    global recv_prepare, session_id, actor_id

    ret, msg = prepare_channel.get()
    if msg is None:
        return

    if ret >= 0 and msg.type() == MT_ACTOR_PREPARE:
        recv_prepare = True
        data = libMulticastNetwork.getMessageData(msg)
        prepare_msg = ActorPrepare()
        prepare_msg.ParseFromString(data)
        session_id = prepare_msg.session_id
        print(f"Received prepare: session_id={session_id}")


def send_prepare_result():
    """发送 ActorPrepareResult"""
    print("Sending prepare result")
    result = ActorPrepareResult()
    result.session_id = session_id
    result.actor_id = actor_id
    result.result = True

    data = result.SerializeToString()
    ret = prepare_channel.put(MT_ACTOR_PREPARE_RESULT, len(data), data)
    if ret != 0:
        print("send prepare result error")


def process_notify():
    """处理 Notify 消息"""
    global start_test, recv_prepare

    ret, msg = notify_channel.get()
    if msg is None:
        return

    if ret >= 0 and msg.type() == MT_NOTIFY:
        notify = Notify()
        data = libMulticastNetwork.getMessageData(msg)
        notify.ParseFromString(data)

        if notify.type in [NT_ABORT_TEST, NT_FINISH_TEST]:
            print("Finish session")
            start_test = False
            recv_prepare = False
        elif notify.type == NT_START_TEST:
            print("Start session")
            start_test = True
        else:
            print(f"Notify: session={notify.session_id}, type={notify.type}")


def get_image():
    """接收图像"""
    msg = image_channel.get_image()
    if len(msg) == 0:
        return None

    # 获取最新的图像
    for image in msg:
        img = image.data.astype(np.uint8).reshape(image.height, image.width, 3)

    return img


def send_vehicle_control(steering, throttle_brake):
    """
    发送 VehicleControl 消息

    Args:
        steering: [-1, 1] 归一化方向盘值
        throttle_brake: [-1, 1] 正值油门，负值刹车
    """
    cmd = VehicleControl()

    # Steering: 归一化值 → 方向盘转角 (rad)
    cmd.steering_control.target_steering_wheel_angle = steering * MAX_STEERING_RAD

    # Throttle/Brake: 归一化值 → 踏板位置 (0-100)
    if throttle_brake >= 0:
        cmd.driving_control.target_accelerator_pedal_position = throttle_brake * 100.0
        cmd.brake_control.target_brake_pedal_position = 0.0
    else:
        cmd.driving_control.target_accelerator_pedal_position = 0.0
        cmd.brake_control.target_brake_pedal_position = -throttle_brake * 100.0

    data = cmd.SerializeToString()
    ret = cmd_channel.put(VEHICLE_CONTROL, len(data), data)
    if ret != 0:
        print("send vehicle control error")


# ============ 主函数 ============
def main():
    global recv_prepare, start_test

    # 解析命令行参数
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config_center", type=str, default="10.11.17.88:52009")
    arg_parser.add_argument("--field_id", type=str, default="unique_fieldid")
    arg_parser.add_argument("--net_interface", type=str, default="eno2")
    args = arg_parser.parse_args()

    # 初始化 OnSite 通道
    init_channels(args)

    # 初始化 Viewer 和控制器
    viewer = OnSiteViewer(H=720, W=1280)
    controller = KeyboardController(viewer.window)

    print("OnSite Viewer started. Use WASD to control.")
    print("Press ESC or close window to exit.")

    try:
        while viewer.is_running():
            # 1. 处理 Notify 消息
            process_notify()

            # 2. 握手阶段
            if not recv_prepare:
                get_prepare()
                viewer.render(None)
                time.sleep(0.1)
                continue

            if recv_prepare and not start_test:
                send_prepare_result()
                viewer.render(None)
                time.sleep(1)
                continue

            # 3. 运行阶段
            # 3.1 接收图像
            img = get_image()

            # 3.2 渲染图像到窗口
            viewer.render(img)

            # 3.3 获取键盘输入
            steering, throttle_brake = controller.process_input()

            # 3.4 发送 VehicleControl
            send_vehicle_control(steering, throttle_brake)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        viewer.shutdown()
        print("OnSite Viewer closed.")


if __name__ == "__main__":
    main()
