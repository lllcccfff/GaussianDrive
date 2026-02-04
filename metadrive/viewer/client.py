"""
data format: client received
 - image (raw bytes + shape + format)
"""

import torch
import threading
import numpy as np
import grpc
from concurrent.futures import ThreadPoolExecutor
from metadrive.utils.remote_viewer_proto import remote_viewer_pb2, remote_viewer_pb2_grpc

# fmt: on

_DEFAULT_MAX_MESSAGE_BYTES = 2048 * 2048 * 3


class Client:
    def __init__(self,
                 server_ip='127.0.0.1', 
                 server_port=56789,
                 lock: threading.Lock = None,
                 max_message_bytes: int = _DEFAULT_MAX_MESSAGE_BYTES,
                 **kwargs,
                 ):
        self.server_ip = server_ip
        self.server_port = server_port
        self.max_message_bytes = max_message_bytes
        self.lock = lock or threading.Lock()
        self.output = None
        self.input = None
        self._last_action = [0.0, 0.0]
        self._grpc_server = None

    
    def run(self):
        client_thread = threading.Thread(target=self.client_thread, daemon=True)
        client_thread.start()

    def client_thread(self):
        server_options = [
            ("grpc.max_send_message_length", self.max_message_bytes),
            ("grpc.max_receive_message_length", self.max_message_bytes),
        ]
        self._grpc_server = grpc.server(ThreadPoolExecutor(max_workers=2), options=server_options)
        servicer = remote_viewer_pb2_grpc.OnsiteViewerServiceServicer()

        def send_image(request, context):
            tensor = self._decode_frame(request)
            if tensor is not None:
                with self.lock:
                    self.output = tensor

            with self.lock:
                action = self.input
                if action is None:
                    action = self._last_action
                else:
                    self._last_action = action
            return remote_viewer_pb2.Action(
                steering=float(action[0]),
                throttle_brake=float(action[1]),
            )

        servicer.SendImage = send_image
        remote_viewer_pb2_grpc.add_OnsiteViewerServiceServicer_to_server(servicer, self._grpc_server)
        self._grpc_server.add_insecure_port(f"{self.server_ip}:{self.server_port}")
        print(f"Listening for viewer server at {self.server_ip}:{self.server_port}")
        self._grpc_server.start()
        self._grpc_server.wait_for_termination()

    def shutdown(self):
        if self._grpc_server is not None:
            self._grpc_server.stop(grace=1)
            self._grpc_server = None

    @staticmethod
    def _decode_frame(frame):
        if frame is None or not frame.data:
            return None
        expected = int(frame.width * frame.height * frame.channels)
        img = np.frombuffer(frame.data, dtype=np.uint8)
        if img.size != expected:
            print(f"Image corrupted: got={img.size} expected={expected}")
            return None
        img = img.reshape(frame.height, frame.width, frame.channels)
        if frame.format.upper() == "BGR":
            img = img[:, :, ::-1]
        tensor = torch.from_numpy(img)
        if torch.cuda.is_available():
            tensor = tensor.to("cuda", non_blocking=True)
        return tensor
