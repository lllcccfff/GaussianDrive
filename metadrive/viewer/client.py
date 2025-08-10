"""
data format: client received
 - image
 - modal_list

 --- not implemented yet ---
 - server_frame, server_fps
 - server_gpu (name, vmem, peak_vmem)
"""

import cv2
import zlib
import torch
import asyncio
import threading
import socket
import numpy as np
from torchvision.io import encode_jpeg, decode_jpeg
import websockets

# fmt: on


class Client:
    def __init__(self,
                 server_ip='127.0.0.1', 
                 server_port=56789,
                 lock: threading.Lock = None,
                 **kwargs,
                 ):
        self.server_ip = server_ip
        self.server_port = server_port
        self.lock = lock
        self.output = None
        self.input = None

    
    @property
    def url(self):
        return f"ws://{self.server_ip}:{self.server_port}"
    
    def run(self):
        client_thread = threading.Thread(target=self.client_thread, daemon=True)
        client_thread.start()

    def client_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.client_loop())
        loop.run_forever()

    async def client_loop(self):
        async with websockets.connect(self.url) as websocket:

            while True:
                with self.lock:
                    input = self.input
                    self.input = None
                if input is not None:
                    assert isinstance(input, list) and len(input) == 2
                    action_bytes = np.array(input, dtype=np.float32).tobytes()
                    await websocket.send(action_bytes)

                buffer = await websocket.recv()
                if buffer is not None:
                    try:
                        # https://github.com/pytorch/vision/issues/4378
                        # Still not fixed even to this day? CUDA 12.1 seems fine
                        buffer = decode_jpeg(torch.from_numpy(np.frombuffer(buffer, np.uint8)), device='cuda')  # 10ms for 1080p...
                    except RuntimeError as e:
                        # buffer = decode_jpeg(torch.from_numpy(np.frombuffer(buffer, np.uint8)), device='cpu')  # 10ms for 1080p...
                        raise e
                    buffer = buffer.permute(1, 2, 0)

                    with self.lock:
                        self.output = buffer