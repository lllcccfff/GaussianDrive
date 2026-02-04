"""
OnSite Middleware for MetaDrive integration.

This module provides the OnSiteMiddleware class that encapsulates all OnSite communication logic,
including message sending/receiving and data format conversion between OnSite proto and MetaDrive.
"""

import json
import logging
import sys
import time
import numpy as np

import libMulticastNetwork

# Import proto messages and enums
from metadrive.misc.onsite_middleware.onsite_proto.chassis.proto.chassis_messages_pb2 import VehicleFeedback, VehicleControl
from metadrive.misc.onsite_middleware.onsite_proto.chassis.proto.chassis_enums_pb2 import VEHICLE_FEEDBACK, VEHICLE_CONTROL
from metadrive.misc.onsite_middleware.onsite_proto.main.proto.messages_pb2 import (
    PubRole, SubRole, Notify, ActorPrepare, ActorPrepareResult, SessionInfo
)
from metadrive.misc.onsite_middleware.onsite_proto.main.proto.enums_pb2 import (
    MT_PUBROLE, MT_SUBROLE, MT_NOTIFY, MT_SESSIONINFO,
    MT_ACTOR_PREPARE, MT_ACTOR_PREPARE_RESULT,
    NT_ABORT_TEST, NT_START_TEST, NT_FINISH_TEST, NT_DESTROY_ROLE
)

logger = logging.getLogger(__name__)


class OnSiteMiddleware:
    """
    OnSite communication middleware for MetaDrive.

    Encapsulates all OnSite communication logic, providing simple APIs for:
    - Receiving messages from OnSite server
    - Sending messages to OnSite server
    - Converting between OnSite proto format and MetaDrive format
    """

    # Constants for conversion
    MAX_STEERING_RAD = 1.047  # 60 degrees in radians

    def __init__(self, config_center, field_id, net_interface, local_ip):
        """
        Initialize OnSite middleware.

        Args:
            config_center: Config center address (e.g., "10.11.17.88:52009")
            field_id: Unique field ID (must match daemon and simulator)
            net_interface: Network interface name (e.g., "eno2")
            local_ip: Local IP address
        """
        self.config_center = config_center
        self.field_id = field_id
        self.net_interface = net_interface
        self.local_ip = local_ip

        # Channel references
        self.channels = None
        self.channel_map = {}
        self.prepare_channel = None
        self.notify_channel = None
        self.role_channel = None
        self.cmd_channel = None
        self.session_channel = None
        self.image_channel = None

        # Sequence counters
        self.vehicle_feedback_seq = 0
        self.image_seq = 0

        # Initialize channels
        self.initialize_channels()

    def initialize_channels(self):
        """
        Initialize multicast network channels.

        Raises:
            RuntimeError: If channel creation fails
        """
        param = libMulticastNetwork.CreateChannelsParam()
        param.config_center_addr = self.config_center
        param.local_ip = self.local_ip
        param.net_interface_name = self.net_interface
        param.field_id = self.field_id
        param.log_level = 1  # 1-info, 2-warning, 3-error
        param.client_name = "simulator"
        param.recv_self_msg = True

        self.channels = libMulticastNetwork.ChannelPtrVector()
        ret = libMulticastNetwork.create_channels(param, self.channels)

        if ret:
            raise RuntimeError(f"Failed to create channels, ret: {ret}")

        # Build channel map
        for c in self.channels:
            logger.info(f"Created channel: {c.name()}, id: {c.id()}")
            self.channel_map[c.name()] = c

        # Assign channel references
        self.prepare_channel = self.channel_map.get('prepare')
        self.notify_channel = self.channel_map.get('notify')
        self.role_channel = self.channel_map.get('pubrole')
        self.cmd_channel = self.channel_map.get('vehiclecontrol')
        self.session_channel = self.channel_map.get('sessioninfo')
        self.image_channel = self.channel_map.get('camera')

        # Initialize image decoder
        if not libMulticastNetwork.InitImageDecoder():
            raise RuntimeError("Failed to initialize image decoder")

        logger.info("OnSite middleware initialized successfully")

    def close(self):
        """Close all channels and cleanup resources."""
        logger.info("Closing OnSite middleware")
        # Channels are managed by libMulticastNetwork, no explicit cleanup needed

    # ==================== Receive Methods ====================

    def recv_actor_prepare(self):
        """
        Receive ActorPrepare message from OnSite server.

        Returns:
            tuple: (session_id, actor_id, brief_data) if message received, None otherwise
        """
        if self.prepare_channel is None:
            return None

        ret, msg = self.prepare_channel.get()
        if msg is None or ret < 0:
            return None

        if msg.type() == MT_ACTOR_PREPARE:
            data = libMulticastNetwork.getMessageData(msg)
            prepare_msg = ActorPrepare()
            prepare_msg.ParseFromString(data)

            session_id = prepare_msg.session_id
            actor_id = prepare_msg.actor_id

            # Parse brief_data if available
            brief_data = None
            if prepare_msg.archive_info.brief_data:
                try:
                    brief_data = json.loads(prepare_msg.archive_info.brief_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse brief_data: {e}")

            logger.info(f"Received ActorPrepare: session={session_id}, actor={actor_id}")
            return (session_id, actor_id, brief_data)

        return None

    def recv_notify(self):
        """
        Receive Notify message from OnSite server.

        Returns:
            Notify: Notify proto message if received, None otherwise
        """
        if self.notify_channel is None:
            return None

        ret, msg = self.notify_channel.get()
        if msg is None or ret < 0:
            return None

        if msg.type() == MT_NOTIFY:
            data = libMulticastNetwork.getMessageData(msg)
            notify = Notify()
            notify.ParseFromString(data)
            logger.debug(f"Received Notify: type={notify.type}, role_id={notify.role_id}")
            return notify

        return None

    def recv_all_notifies(self):
        """
        Receive all pending Notify messages from OnSite server.

        Returns:
            list: List of Notify proto messages
        """
        notifies = []
        while True:
            notify = self.recv_notify()
            if notify is None:
                break
            notifies.append(notify)
        return notifies

    def recv_pub_role(self):
        """
        Receive PubRole message from OnSite server.

        Returns:
            PubRole: PubRole proto message if received, None otherwise
        """
        if self.role_channel is None:
            return None

        ret, msg = self.role_channel.get()
        if msg is None or ret < 0:
            return None

        if msg.type() == MT_PUBROLE:
            data = libMulticastNetwork.getMessageData(msg)
            pub_role = PubRole()
            pub_role.ParseFromString(data)
            logger.debug(f"Received PubRole with {len(pub_role.s_roles)} roles")
            return pub_role

        return None

    def recv_vehicle_control(self):
        """
        Receive VehicleControl message and convert to MetaDrive action.

        Returns:
            list: [steering, throttle_brake] if message received, None otherwise
        """
        if self.cmd_channel is None:
            return None

        ret, msg = self.cmd_channel.get()
        if msg is None or ret < 0:
            return None

        if msg.type() == VEHICLE_CONTROL:
            data = libMulticastNetwork.getMessageData(msg)
            control = VehicleControl()
            control.ParseFromString(data)

            # Convert to MetaDrive action
            action = self._vehicle_control_to_action(control)
            logger.debug(f"Received VehicleControl: steering={action[0]:.3f}, throttle_brake={action[1]:.3f}")
            return action

        return None

    def recv_vehicle_feedback(self):
        """
        Receive VehicleFeedback message from OnSite server.
        Note: This is only received, not used for simulation state.

        Returns:
            VehicleFeedback: VehicleFeedback proto message if received, None otherwise
        """
        if self.cmd_channel is None:
            return None

        ret, msg = self.cmd_channel.get()
        if msg is None or ret < 0:
            return None

        if msg.type() == VEHICLE_FEEDBACK:
            data = libMulticastNetwork.getMessageData(msg)
            feedback = VehicleFeedback()
            feedback.ParseFromString(data)
            logger.debug("Received VehicleFeedback from OnSite")
            return feedback

        return None

    def recv_session_info(self):
        """
        Receive SessionInfo message from OnSite server.

        Returns:
            SessionInfo: SessionInfo proto message if received, None otherwise
        """
        if self.session_channel is None:
            return None

        ret, msg = self.session_channel.get()
        if msg is None or ret < 0:
            return None

        if msg.type() == MT_SESSIONINFO:
            data = libMulticastNetwork.getMessageData(msg)
            session_info = SessionInfo()
            session_info.ParseFromString(data)
            logger.debug("Received SessionInfo from OnSite")
            return session_info

        return None

    # ==================== Send Methods ====================

    def send_actor_prepare_result(self, session_id, actor_id, result=True):
        """
        Send ActorPrepareResult message to OnSite server.

        Args:
            session_id: Session ID from ActorPrepare
            actor_id: Actor ID
            result: Preparation result (default: True)
        """
        if self.prepare_channel is None:
            logger.warning("Prepare channel not available")
            return

        msg = ActorPrepareResult()
        msg.session_id = session_id
        msg.actor_id = actor_id
        msg.result = result

        data = msg.SerializeToString()
        length = len(data)
        ret = self.prepare_channel.put(MT_ACTOR_PREPARE_RESULT, length, data)

        if ret != 0:
            logger.error(f"Failed to send ActorPrepareResult, ret: {ret}")
        else:
            logger.info(f"Sent ActorPrepareResult: session={session_id}, result={result}")

    def send_sub_role(self, session_id):
        """
        Send SubRole message to OnSite server.
        Note: Only session_id is required, other fields are left empty.

        Args:
            session_id: Current session ID
        """
        if self.role_channel is None:
            logger.warning("Role channel not available")
            return

        msg = SubRole()
        msg.session_id = session_id
        # Other fields (role_types, role_ids, role_AOIs) are left empty

        data = msg.SerializeToString()
        length = len(data)
        ret = self.role_channel.put(MT_SUBROLE, length, data)

        if ret != 0:
            logger.error(f"Failed to send SubRole, ret: {ret}")
        else:
            logger.info(f"Sent SubRole: session={session_id}")

    def send_pub_role(self, ego_state, participants_states, last_received_pub_role, current_timestamp):
        """
        Send PubRole message to OnSite server with updated agent states.

        Args:
            ego_state: Dictionary with ego vehicle state
            participants_states: Dictionary of participant states {agent_id: state_dict}
            last_received_pub_role: Last received PubRole message (for preserving fields)
            current_timestamp: Current simulation timestamp in microseconds
        """
        if self.role_channel is None:
            logger.warning("Role channel not available")
            return

        msg = PubRole()

        # Add ego vehicle
        ego_role = self._agent_state_to_single_role(
            'actor', ego_state, last_received_pub_role, current_timestamp
        )
        if ego_role:
            msg.s_roles.append(ego_role)

        # Add participants
        for agent_id, state in participants_states.items():
            role = self._agent_state_to_single_role(
                agent_id, state, last_received_pub_role, current_timestamp
            )
            if role:
                msg.s_roles.append(role)

        data = msg.SerializeToString()
        length = len(data)
        ret = self.role_channel.put(MT_PUBROLE, length, data)

        if ret != 0:
            logger.error(f"Failed to send PubRole, ret: {ret}")
        else:
            logger.debug(f"Sent PubRole with {len(msg.s_roles)} roles")

    def send_vehicle_feedback(self, vehicle_state, current_timestamp, last_received_feedback=None):
        """
        Send VehicleFeedback message to OnSite server.

        Args:
            vehicle_state: Dictionary with vehicle state from MetaDrive
            current_timestamp: Current simulation timestamp in microseconds
            last_received_feedback: Last received VehicleFeedback (for preserving fields)
        """
        if self.cmd_channel is None:
            logger.warning("Command channel not available")
            return

        msg = self._vehicle_state_to_feedback(vehicle_state, current_timestamp, last_received_feedback)

        data = msg.SerializeToString()
        length = len(data)
        ret = self.cmd_channel.put(VEHICLE_FEEDBACK, length, data)

        if ret != 0:
            logger.error(f"Failed to send VehicleFeedback, ret: {ret}")
        else:
            logger.debug("Sent VehicleFeedback")

    def send_images(self, images, timestamp):
        """
        Send multiple images to OnSite server.

        Args:
            images: List of numpy arrays (H, W, 3) in BGR format
            timestamp: Timestamp in seconds
        """
        if self.image_channel is None:
            logger.warning("Image channel not available")
            return

        if not images:
            return

        py_images = []
        for img in images:
            py_img = libMulticastNetwork.PyImage()
            py_img.timestamp_sec = timestamp
            py_img.camera_timestamp = int(timestamp * 1e6)
            py_img.sequence_num = self.image_seq
            py_img.measurement_time = timestamp
            py_img.height = img.shape[0]
            py_img.width = img.shape[1]
            py_img.encoding = "bgr8"
            py_img.data = img.ravel()
            py_images.append(py_img)
            self.image_seq += 1

        ret = self.image_channel.put_image_simple(py_images)
        if ret != 0:
            logger.error(f"Failed to send images, ret: {ret}")
        else:
            logger.debug(f"Sent {len(py_images)} images")

    # ==================== Conversion Utility Functions ====================

    def _vehicle_control_to_action(self, control):
        """
        Convert VehicleControl proto message to MetaDrive action.

        Args:
            control: VehicleControl proto message

        Returns:
            list: [steering, throttle_brake] normalized to [-1, 1]
        """
        # Steering: normalize to [-1, 1]
        steering_angle = control.steering_control.target_steering_wheel_angle
        steering = np.clip(steering_angle / self.MAX_STEERING_RAD, -1.0, 1.0)

        # Throttle/Brake: combine pedal positions
        accelerator = control.driving_control.target_accelerator_pedal_position
        brake = control.brake_control.target_brake_pedal_position
        throttle_brake = (accelerator - brake) / 100.0
        throttle_brake = np.clip(throttle_brake, -1.0, 1.0)

        return [float(steering), float(throttle_brake)]

    def _euler_to_quaternion(self, roll, pitch, yaw):
        """
        Convert Euler angles to quaternion.

        Args:
            roll: Roll angle in radians
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians

        Returns:
            tuple: (x, y, z, w) quaternion
        """
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return (x, y, z, w)

    def _quaternion_to_matrix(self, position, quaternion):
        """
        Convert quaternion and position to 4x4 transform matrix.

        Args:
            position: Position proto message with x, y, z
            quaternion: Quaternion proto message with x, y, z, w

        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w

        # Quaternion to rotation matrix
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

        # Construct 4x4 transform matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [position.x, position.y, position.z]

        return T

    def _agent_state_to_single_role(self, agent_id, state, last_received_pub_role, current_timestamp):
        """
        Convert agent state to SingleRole proto message.

        Args:
            agent_id: Agent identifier
            state: Dictionary with agent state (position, velocity, heading, etc.)
            last_received_pub_role: Last received PubRole for preserving fields
            current_timestamp: Current timestamp in microseconds

        Returns:
            SingleRole proto message
        """
        from metadrive.misc.onsite_middleware.onsite_proto.main.proto.messages_pb2 import SingleRole

        role = SingleRole()
        role.id = agent_id
        role.name = agent_id

        # Preserve type and size from last received PubRole if available
        cached_role = None
        if last_received_pub_role:
            for r in last_received_pub_role.s_roles:
                if r.id == agent_id:
                    cached_role = r
                    break

        if cached_role:
            role.type = cached_role.type
            role.box.size.CopyFrom(cached_role.box.size)
        else:
            # Default values if no cached role
            role.type = 1  # Default vehicle type
            role.box.size.x = state.get('length', 4.5)
            role.box.size.y = state.get('width', 2.0)
            role.box.size.z = state.get('height', 1.5)

        # Position
        position = state.get('position', [0, 0, 0])
        role.box.bottom_center.x = position[0]
        role.box.bottom_center.y = position[1]
        role.box.bottom_center.z = position[2]

        # Rotation (from heading_theta to quaternion)
        heading = state.get('heading_theta', 0)
        quat = self._euler_to_quaternion(0, 0, heading)
        role.box.rotation.x = quat[0]
        role.box.rotation.y = quat[1]
        role.box.rotation.z = quat[2]
        role.box.rotation.w = quat[3]

        # Linear velocity
        velocity = state.get('velocity', [0, 0, 0])
        role.linear_speed.x = velocity[0]
        role.linear_speed.y = velocity[1]
        role.linear_speed.z = velocity[2]

        # Angular velocity
        angular_velocity = state.get('angular_velocity', 0)
        role.angular_speed.z = angular_velocity

        # Timestamp (microseconds to milliseconds)
        role.report_ts = current_timestamp // 1000

        return role

    def _vehicle_state_to_feedback(self, vehicle_state, current_timestamp, last_received_feedback=None):
        """
        Convert vehicle state to VehicleFeedback proto message.

        Args:
            vehicle_state: Dictionary with vehicle state from MetaDrive
            current_timestamp: Current timestamp in microseconds
            last_received_feedback: Last received feedback for preserving fields

        Returns:
            VehicleFeedback proto message
        """
        feedback = VehicleFeedback()

        # Header
        feedback.header.sim_ts = current_timestamp // 1000  # microseconds to milliseconds
        feedback.header.send_ts = int(time.time() * 1000)
        feedback.header.seq_no = self.vehicle_feedback_seq
        self.vehicle_feedback_seq += 1

        # Steering feedback
        feedback.steering_feedback.steering_wheel_angle = vehicle_state.get('steering_wheel_angle', 0.0)
        feedback.steering_feedback.steering_wheel_speed = vehicle_state.get('steering_wheel_speed', 0.0)
        feedback.steering_feedback.left_directive_wheel_angle = vehicle_state.get('left_directive_wheel_angle', 0.0)
        feedback.steering_feedback.right_directive_wheel_angle = vehicle_state.get('right_directive_wheel_angle', 0.0)

        # Driving feedback
        throttle_brake = vehicle_state.get('throttle_brake', 0.0)
        feedback.driving_feedback.accelerator_pedal_position = max(0, throttle_brake * 100)

        # Brake feedback
        feedback.brake_feedback.brake_pedal_position = max(0, -throttle_brake * 100)

        # BCM feedback
        feedback.bcm_feedback.vehicle_speed = vehicle_state.get('speed', 0.0) * 3.6  # m/s to km/h
        feedback.bcm_feedback.longitudinal_acceleration = vehicle_state.get('longitudinal_acceleration', 0.0)
        feedback.bcm_feedback.front_left_wheel_speed = vehicle_state.get('front_left_wheel_speed', 0.0)
        feedback.bcm_feedback.front_right_wheel_speed = vehicle_state.get('front_right_wheel_speed', 0.0)
        feedback.bcm_feedback.rear_left_wheel_speed = vehicle_state.get('rear_left_wheel_speed', 0.0)
        feedback.bcm_feedback.rear_right_wheel_speed = vehicle_state.get('rear_right_wheel_speed', 0.0)

        # Preserve fields from last received feedback if available
        if last_received_feedback:
            # Copy fields that MetaDrive cannot provide
            if last_received_feedback.HasField('driving_feedback'):
                feedback.driving_feedback.engine_rpm = last_received_feedback.driving_feedback.engine_rpm
            if last_received_feedback.HasField('gear_feedback'):
                feedback.gear_feedback.CopyFrom(last_received_feedback.gear_feedback)

        return feedback
