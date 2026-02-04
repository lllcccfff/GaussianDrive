#!/usr/bin/env python3
"""OnSite remote viewer server.

This process connects to the OnSite multicast network, receives images, and
forwards them to remote clients over gRPC. It also receives actions from the
remote client via gRPC and sends VehicleControl to the OnSite server.
"""

import argparse
import logging
import fcntl
import socket
import struct
import time
from typing import Optional

import numpy as np

try:
    import grpc
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("grpcio is required for viewer_server.py") from exc

import libMulticastNetwork

from metadrive.utils.onsite_proto.chassis.proto.chassis_enums_pb2 import VEHICLE_CONTROL
from metadrive.utils.onsite_proto.chassis.proto.chassis_messages_pb2 import VehicleControl
from metadrive.utils.onsite_proto.main.proto.enums_pb2 import (
    MT_NOTIFY,
    MT_ACTOR_PREPARE,
    MT_ACTOR_PREPARE_RESULT,
    NT_START_TEST,
    NT_ABORT_TEST,
    NT_FINISH_TEST,
)
from metadrive.utils.onsite_proto.main.proto.messages_pb2 import Notify, ActorPrepare, ActorPrepareResult

from metadrive.utils.remote_viewer_proto import remote_viewer_pb2, remote_viewer_pb2_grpc

logger = logging.getLogger("onsite_viewer_server")

MAX_STEERING_RAD = 1.047  # 60 degrees


def get_ip_address(ifname: str) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        return socket.inet_ntoa(
            fcntl.ioctl(
                sock.fileno(),
                0x8915,  # SIOCGIFADDR
                struct.pack("256s", bytes(ifname[:15], "utf-8")),
            )[20:24]
        )
    except Exception:
        return ""
    finally:
        sock.close()


class OnSiteBridge:
    def __init__(self, args: argparse.Namespace, action_state, grpc_client) -> None:
        self._args = args
        self._action_state = action_state
        self._grpc_client = grpc_client
        self._stop = False

        self._recv_prepare = False
        self._start_test = False
        self._session_id = ""
        self._actor_id = ""
        self._prepare_sent = False

        self._notify_channel = None
        self._cmd_channel = None
        self._prepare_channel = None
        self._image_channel = None

        self._init_channels()

    def _init_channels(self) -> None:
        param = libMulticastNetwork.CreateChannelsParam()
        local_ip = get_ip_address(self._args.net_interface)
        if not local_ip:
            raise RuntimeError(f"Failed to resolve IP for interface {self._args.net_interface}")

        param.config_center_addr = self._args.config_center
        param.local_ip = local_ip
        param.net_interface_name = self._args.net_interface
        param.field_id = self._args.field_id
        param.log_level = 1
        param.client_name = "apollo_testee"
        param.recv_self_msg = False

        channels = libMulticastNetwork.ChannelPtrVector()
        ret = libMulticastNetwork.create_channels(param, channels)
        if ret:
            raise RuntimeError(f"create channels failed, ret: {ret}")

        channel_map = {c.name(): c for c in channels}
        self._notify_channel = channel_map["notify"]
        self._cmd_channel = channel_map["vehiclecontrol"]
        self._prepare_channel = channel_map["prepare"]
        self._image_channel = channel_map["camera"]

        if not libMulticastNetwork.InitImageDecoder():
            raise RuntimeError("image decoder init error")

        logger.info("OnSite channels initialized")

    def stop(self) -> None:
        self._stop = True

    def _process_notify(self) -> None:
        ret, msg = self._notify_channel.get()
        if msg is None:
            return

        if ret >= 0 and msg.type() == MT_NOTIFY:
            notify = Notify()
            data = libMulticastNetwork.getMessageData(msg)
            notify.ParseFromString(data)

            if notify.type in [NT_ABORT_TEST, NT_FINISH_TEST]:
                logger.info("Finish session")
                self._start_test = False
                self._recv_prepare = False
                self._prepare_sent = False
                self._session_id = ""
            elif notify.type == NT_START_TEST:
                logger.info("Start session")
                self._start_test = True
            else:
                logger.info("Notify: session=%s type=%s", notify.session_id, notify.type)

    def _get_prepare(self) -> None:
        ret, msg = self._prepare_channel.get()
        if msg is None:
            return

        if ret >= 0 and msg.type() == MT_ACTOR_PREPARE:
            data = libMulticastNetwork.getMessageData(msg)
            prepare_msg = ActorPrepare()
            prepare_msg.ParseFromString(data)
            self._recv_prepare = True
            self._prepare_sent = False
            self._session_id = prepare_msg.session_id
            self._actor_id = prepare_msg.actor_id
            logger.info("Received prepare: session_id=%s actor_id=%s", self._session_id, self._actor_id)

    def _send_prepare_result(self) -> None:
        result = ActorPrepareResult()
        result.session_id = self._session_id
        result.actor_id = self._actor_id
        result.result = True

        data = result.SerializeToString()
        ret = self._prepare_channel.put(MT_ACTOR_PREPARE_RESULT, len(data), data)
        if ret != 0:
            logger.warning("send prepare result error")
        else:
            logger.info("Sent prepare result: session_id=%s actor_id=%s", self._session_id, self._actor_id)
            self._prepare_sent = True

    def _get_image(self) -> Optional[np.ndarray]:
        msg = self._image_channel.get_image()
        if len(msg) == 0:
            return None

        img = None
        for image in msg:
            img = image.data.astype(np.uint8).reshape(image.height, image.width, 3)
        return img

    def _send_vehicle_control(self, steering: float, throttle_brake: float) -> None:
        cmd = VehicleControl()
        cmd.steering_control.target_steering_wheel_angle = steering * MAX_STEERING_RAD

        if throttle_brake >= 0:
            cmd.driving_control.target_accelerator_pedal_position = throttle_brake * 100.0
            cmd.brake_control.target_brake_pedal_position = 0.0
        else:
            cmd.driving_control.target_accelerator_pedal_position = 0.0
            cmd.brake_control.target_brake_pedal_position = -throttle_brake * 100.0

        data = cmd.SerializeToString()
        ret = self._cmd_channel.put(VEHICLE_CONTROL, len(data), data)
        if ret != 0:
            logger.warning("send vehicle control error")

    def run(self) -> None:

        while not self._stop:
            self._process_notify()

            if not self._recv_prepare:
                self._get_prepare()
                time.sleep(0.05)
                continue

            if self._recv_prepare and not self._start_test:
                if not self._prepare_sent:
                    self._send_prepare_result()
                time.sleep(0.2)
                continue

            img = self._get_image()
            if img is None:
                continue

            frame = {
                "data": img.tobytes(),
                "width": img.shape[1],
                "height": img.shape[0],
                "channels": img.shape[2],
                "format": "BGR",
                "timestamp_us": int(time.time() * 1e6),
            }
            action = self._grpc_client.send_image(
                remote_viewer_pb2.Image(
                    data=frame["data"],
                    width=frame["width"],
                    height=frame["height"],
                    channels=frame["channels"],
                    format=frame["format"],
                    timestamp_us=frame["timestamp_us"],
                )
            )
            if action is not None:
                self._action_state["steering"] = float(action.steering)
                self._action_state["throttle_brake"] = float(action.throttle_brake)

            steering = float(self._action_state["steering"])
            throttle_brake = float(self._action_state["throttle_brake"])
            self._send_vehicle_control(steering, throttle_brake)


class OnsiteViewerGrpcClient:
    def __init__(self, host: str, port: int, max_message_bytes: int) -> None:
        self._target = f"{host}:{port}"
        self._options = [
            ("grpc.max_send_message_length", max_message_bytes),
            ("grpc.max_receive_message_length", max_message_bytes),
        ]
        self._channel = None
        self._stub = None

    def _connect(self) -> bool:
        if self._stub is not None:
            return True
        channel = grpc.insecure_channel(self._target, options=self._options)
        try:
            grpc.channel_ready_future(channel).result(timeout=1.0)
        except grpc.FutureTimeoutError as exc:
            logger.warning("gRPC channel not ready: %s", exc)
            channel.close()
            return False
        self._channel = channel
        self._stub = remote_viewer_pb2_grpc.OnsiteViewerServiceStub(channel)
        logger.info("Connected to viewer client at %s", self._target)
        return True

    def send_image(self, frame: remote_viewer_pb2.Image) -> Optional[remote_viewer_pb2.Action]:
        if not self._connect():
            return None
        try:
            return self._stub.SendImage(frame, timeout=1.0)
        except grpc.RpcError as exc:
            logger.warning("SendImage RPC error: %s", exc)
            self.close()
            return None

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None


def main() -> None:
    parser = argparse.ArgumentParser(description="OnSite remote viewer server")
    parser.add_argument("--config_center", type=str, default="www.zjvts.cn:52009")
    parser.add_argument("--field_id", type=str, default="unique_fieldid")
    parser.add_argument("--net_interface", type=str, default="eno2")
    parser.add_argument("--grpc_host", type=str, default="127.0.0.1", help="viewer client host")
    parser.add_argument("--grpc_port", type=int, default=50051, help="viewer client port")
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    action_state = {"steering": 0.0, "throttle_brake": 0.0}
    max_bytes = 2048 * 2048 * 3
    logger.info("Using gRPC max message bytes: %d", max_bytes)
    grpc_client = OnsiteViewerGrpcClient(args.grpc_host, args.grpc_port, max_bytes)
    bridge = OnSiteBridge(args, action_state, grpc_client)

    try:
        bridge.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        bridge.stop()
        grpc_client.close()


if __name__ == "__main__":
    main()
