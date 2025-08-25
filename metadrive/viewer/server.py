"""
data format: server received
 - colormaping
 - render modal

 === format ===
 frame int
 pause bool
 camera_pos torch.tensor 3
 camera_front torch.tensor 3
 image_width int
 image_height int
 fovxy_ratio float
 fovy float
"""

from __future__ import annotations

import os
import glm
import time
import zlib
import torch
import asyncio
import threading
import websockets
import numpy as np
import cv2
import torch.nn.functional as F

from copy import deepcopy
from typing import List, Union, Dict
from glm import vec3, vec4, mat3, mat4, mat4x3
from torchvision.io import encode_jpeg, decode_jpeg

from easydrive.utils.console_utils import log


class WebSocketServer:
    # Viewer should be used in conjuction with another runner, which explicitly handles model loading
    def __init__(self,
                 host: str = '0.0.0.0',
                 port: int = 1024,
                 lock: threading.Lock = None,
                 jpeg_quality: int = 75,
                 **kwargs,
                 ):

        # Socket related initialization
        self.host = host
        self.port = port

        self.output = None
        self.input = None
        self.lock = lock
        
        # self.stream = torch.cuda.Stream()
        self.jpeg_quality = jpeg_quality

        
    def run(self):
        server_thread = threading.Thread(target=self.server_thread, daemon=True)
        server_thread.start()
    
    def server_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        log('Preparing websocket server for sending images & receiving cameras')
        server = websockets.serve(self.server_loop, self.host, self.port)

        loop.run_until_complete(server)
        loop.run_forever()

    async def server_loop(self, websocket: websockets.WebSocket, _path: str):
        send_task = asyncio.create_task(self._send_outputs(websocket))
        recv_task = asyncio.create_task(self._recv_inputs(websocket))
        _done, pending = await asyncio.wait({send_task, recv_task}, return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()

    async def _send_outputs(self, websocket: websockets.WebSocket):
        try:
            while True:
                # self.stream.synchronize()  # waiting for the copy event to complete
                with self.lock:
                    output = self.output
                    self.output = None
                if output is not None:
                    if isinstance(output, np.ndarray):
                        output = torch.from_numpy(output)
                    output = output.permute(2, 0, 1).cpu()  # HWC -> CHW
                    data = encode_jpeg(output, quality=self.jpeg_quality).numpy().tobytes()
                    await websocket.send(data)
                await asyncio.sleep(0)
        except websockets.ConnectionClosed:
            return

    async def _recv_inputs(self, websocket: websockets.WebSocket):
        try:
            while True:
                response = await websocket.recv()
                if response is not None:
                    try:
                        action = np.frombuffer(response, dtype=np.float32).tolist()
                        assert len(action) == 2
                    except Exception as e:
                        print(f"Data corrupted from client: {e}")
                        continue
                    
                    with self.lock:
                        self.input = action
        except websockets.ConnectionClosed:
            return
