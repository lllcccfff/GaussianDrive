"""
data format: server received
 - action (steering, throttle_brake)

data format: server sends
 - image (raw bytes + shape + format)
"""

from __future__ import annotations

import time
import threading
from typing import Optional

import numpy as np
import torch

try:
    import grpc
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("grpcio is required for metadrive.viewer.server") from exc
from easydrive.utils.console_utils import log
from metadrive.utils.remote_viewer_proto import remote_viewer_pb2, remote_viewer_pb2_grpc


_DEFAULT_MAX_MESSAGE_BYTES = 2048 * 2048 * 3


class WebSocketServer:
    # Viewer should be used in conjuction with another runner, which explicitly handles model loading
    def __init__(self,
                 host: str = '0.0.0.0',
                 port: int = 1024,
                 lock: threading.Lock = None,
                 jpeg_quality: int = 75,
                 max_message_bytes: int = _DEFAULT_MAX_MESSAGE_BYTES,
                 **kwargs,
                 ):

        # gRPC related initialization
        self.host = host
        self.port = port
        self.max_message_bytes = max_message_bytes

        self.output = None
        self.input = None
        self.lock = lock or threading.Lock()
        self.jpeg_quality = jpeg_quality  # kept for backward compatibility
        self._grpc_channel = None
        self._grpc_stub = None
        self._stop = threading.Event()

        
    def run(self):
        server_thread = threading.Thread(target=self.server_thread, daemon=True)
        server_thread.start()
    
    def server_thread(self):
        server_options = [
            ("grpc.max_send_message_length", self.max_message_bytes),
            ("grpc.max_receive_message_length", self.max_message_bytes),
        ]
        target = f"{self.host}:{self.port}"
        while not self._stop.is_set():
            channel = grpc.insecure_channel(target, options=server_options)
            self._grpc_channel = channel
            self._grpc_stub = remote_viewer_pb2_grpc.OnsiteViewerServiceStub(channel)
            try:
                grpc.channel_ready_future(channel).result(timeout=5.0)
                log(f"Connected to viewer client at {target}, sending images")
                while not self._stop.is_set():
                    frame = self._pop_frame()
                    if frame is None:
                        time.sleep(0.001)
                        continue
                    try:
                        action = self._grpc_stub.SendImage(frame, timeout=5.0)
                    except grpc.RpcError as exc:
                        log(f"SendImage RPC error: {exc}")
                        break
                    with self.lock:
                        self.input = [float(action.steering), float(action.throttle_brake)]
            except grpc.FutureTimeoutError as exc:
                log(f"gRPC channel not ready: {exc}")
            except grpc.RpcError as exc:
                log(f"gRPC channel error: {exc}")
            except Exception as exc:
                log(f"Unexpected error in viewer server thread: {exc}")
            finally:
                if channel is not None:
                    channel.close()
                self._grpc_channel = None
                self._grpc_stub = None
            if not self._stop.is_set():
                time.sleep(1)

    def shutdown(self):
        self._stop.set()
        if self._grpc_channel is not None:
            self._grpc_channel.close()
            self._grpc_channel = None
        self._grpc_stub = None

    def _pop_frame(self) -> Optional[remote_viewer_pb2.Image]:
        with self.lock:
            output = self.output
            self.output = None
        if output is None:
            return None

        output_np = self._to_numpy(output)
        if output_np is None:
            return None

        if output_np.ndim == 3 and output_np.shape[-1] not in (1, 3, 4) and output_np.shape[0] in (1, 3, 4):
            output_np = np.transpose(output_np, (1, 2, 0))
        if output_np.ndim == 2:
            output_np = np.stack([output_np] * 3, axis=-1)

        output_np = self._to_uint8(output_np)
        height, width = output_np.shape[:2]
        channels = output_np.shape[2] if output_np.ndim == 3 else 1

        return remote_viewer_pb2.Image(
            data=output_np.tobytes(),
            width=int(width),
            height=int(height),
            channels=int(channels),
            format="RGB",
            timestamp_us=int(time.time() * 1e6),
        )

    @staticmethod
    def _to_numpy(output):
        if isinstance(output, torch.Tensor):
            if output.device.type != "cpu":
                output = output.detach().to("cpu", non_blocking=True)
            else:
                output = output.detach()
            return output.numpy()
        return np.asarray(output)

    @staticmethod
    def _to_uint8(output: np.ndarray) -> np.ndarray:
        if output.dtype == np.uint8:
            return output
        scale = 255.0 if output.max() <= 1.0 else 1.0
        return np.clip(output * scale, 0, 255).astype(np.uint8)
