#!/usr/bin/env python3
"""OnSite remote viewer client.

Listens for gRPC SendImage calls, renders frames, and returns actions.
"""

import argparse
import logging
import platform
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

try:
    import grpc
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("grpcio is required for viewer_client.py") from exc

import glfw
import OpenGL.GL as gl

from metadrive.viewer.manual_controller import KeyboardController
from metadrive.utils.remote_viewer_proto import remote_viewer_pb2, remote_viewer_pb2_grpc

logger = logging.getLogger("onsite_viewer_client")


class OnSiteViewer:
    def __init__(self, height: int = 720, width: int = 1280) -> None:
        self.height = height
        self.width = width
        self.window_title = "OnSite Remote Viewer"
        self.last_image: Optional[np.ndarray] = None

        self._init_glfw()
        self._init_opengl()
        self._init_texture()

    def _init_glfw(self) -> None:
        if not glfw.init():
            raise RuntimeError("Could not initialize OpenGL context")

        if platform.system() == "Darwin":
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 2)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, 1)
            glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, 0)
        else:
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)

        window = glfw.create_window(self.width, self.height, self.window_title, None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Could not initialize window")

        glfw.make_context_current(window)
        glfw.swap_interval(1)
        self.window = window

    def _init_opengl(self) -> None:
        gl.glViewport(0, 0, self.width, self.height)
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)

    def _init_texture(self) -> None:
        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def is_running(self) -> bool:
        return not glfw.window_should_close(self.window)

    def render(self, img: Optional[np.ndarray]) -> None:
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        glfw.poll_events()

        if img is not None:
            self.last_image = img
            self._draw_image(img)
        elif self.last_image is not None:
            self._draw_image(self.last_image)

        glfw.swap_buffers(self.window)

    def _draw_image(self, img: np.ndarray) -> None:
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        height, width = img.shape[:2]

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGB,
            width,
            height,
            0,
            gl.GL_BGR,
            gl.GL_UNSIGNED_BYTE,
            img,
        )

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glBegin(gl.GL_QUADS)
        gl.glTexCoord2f(0, 1)
        gl.glVertex2f(-1, -1)
        gl.glTexCoord2f(1, 1)
        gl.glVertex2f(1, -1)
        gl.glTexCoord2f(1, 0)
        gl.glVertex2f(1, 1)
        gl.glTexCoord2f(0, 0)
        gl.glVertex2f(-1, 1)
        gl.glEnd()
        gl.glDisable(gl.GL_TEXTURE_2D)

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def shutdown(self) -> None:
        if self.texture_id:
            gl.glDeleteTextures(1, [self.texture_id])
        glfw.destroy_window(self.window)
        glfw.terminate()


def _decode_frame(frame) -> Optional[np.ndarray]:
    img = np.frombuffer(frame.data, dtype=np.uint8)
    expected = int(frame.width * frame.height * frame.channels)
    if img.size != expected:
        raise ValueError(
            f"Image size mismatch: got={img.size} expected={expected} "
            f"(w={frame.width} h={frame.height} c={frame.channels})"
        )
    img = img.reshape(frame.height, frame.width, frame.channels)
    if frame.format.upper() == "RGB":
        img = img[:, :, ::-1]
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="OnSite remote viewer client")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="listen host")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    max_bytes = 2048 * 2048 * 3
    logger.info("Using gRPC max message bytes: %d", max_bytes)

    viewer = OnSiteViewer(height=args.height, width=args.width)
    controller = KeyboardController(viewer.window)

    state_lock = threading.Lock()
    state = {"image": None, "action": [0.0, 0.0]}

    server_options = [
        ("grpc.max_send_message_length", max_bytes),
        ("grpc.max_receive_message_length", max_bytes),
    ]
    server = grpc.server(ThreadPoolExecutor(max_workers=2), options=server_options)
    servicer = remote_viewer_pb2_grpc.OnsiteViewerServiceServicer()

    def send_image(request, context):
        try:
            img = _decode_frame(request)
        except ValueError as exc:
            logger.warning("Image decode error: %s", exc)
            img = None
        if img is not None:
            with state_lock:
                state["image"] = img
        with state_lock:
            action = state["action"]
        return remote_viewer_pb2.Action(
            steering=float(action[0]),
            throttle_brake=float(action[1]),
        )

    servicer.SendImage = send_image
    remote_viewer_pb2_grpc.add_OnsiteViewerServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    logger.info("Listening for viewer server at %s:%s", args.host, args.port)

    try:
        while viewer.is_running():
            with state_lock:
                img = state["image"]
            viewer.render(img)
            steering, throttle_brake = controller.process_input()
            with state_lock:
                state["action"] = [float(steering), float(throttle_brake)]
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        server.stop(grace=1)
        viewer.shutdown()


if __name__ == "__main__":
    main()
