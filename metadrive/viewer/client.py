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
        # 添加循环等待服务器连接的功能
        while True:
            try:
                async with websockets.connect(self.url) as websocket:
                    print(f"Connected to server at {self.url}")
                    send_task = asyncio.create_task(self._send_inputs(websocket))
                    recv_task = asyncio.create_task(self._recv_outputs(websocket))
                    _done, pending = await asyncio.wait({send_task, recv_task}, return_when=asyncio.FIRST_EXCEPTION)
                    for task in pending:
                        task.cancel()
            except (websockets.ConnectionClosed, ConnectionRefusedError, OSError) as e:
                print(f"Connection failed: {e}. Retrying in 1 seconds...")
                await asyncio.sleep(1)

    async def _send_inputs(self, websocket):
        try:
            while True:
                with self.lock:
                    _input = self.input
                    self.input = None
                if _input:
                    assert isinstance(_input, list) and len(_input) == 2
                    data = np.array(_input, dtype=np.float32).tobytes()
                    await websocket.send(data)
                # Yield control to allow recv task to run
                await asyncio.sleep(0)
        except websockets.ConnectionClosed:
            return

    async def _recv_outputs(self, websocket):
        try:
            while True:
                buffer = await websocket.recv()
                if buffer is not None:
                    try:
                        # https://github.com/pytorch/vision/issues/4378 Still not fixed even to this day? CUDA 12.1 seems fine
                        tensor = decode_jpeg(torch.from_numpy(np.frombuffer(buffer, np.uint8)), device='cuda')  # 10ms for 1080p...
                        tensor = tensor.permute(1, 2, 0)
                    except RuntimeError as e:
                        print("Image corruptted.")
                        continue

                    with self.lock:
                        self.output = tensor
        except websockets.ConnectionClosed:
            return
