#!/usr/bin/env python3
"""
OnSite Integration Script for MetaDrive.

This script implements the complete communication flow with OnSite server:
1. Initial handshake (ActorPrepare, ActorPrepareResult, SubRole)
2. Main loop (receive PubRole/VehicleControl, step simulation, send updates)
3. Handle Notify messages for agent lifecycle management
"""

import argparse
import logging
import time
import sys
import os

# Add api_reference to path for proto imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'api_reference'))

from metadrive.envs.onsite_scenario_env import OnSiteScenarioEnv
from metadrive.middleware.onsite_middleware import OnSiteMiddleware
from metadrive.manager.agent_manager import AgentState

# Import proto enums for Notify types
from api_reference.main.proto.enums_pb2 import (
    NT_START_TEST, NT_RESUME_TEST, NT_INVALID, NT_ABORT_TEST,
    NT_FINISH_TEST, NT_DESTROY_ROLE, NT_ARRIVED_ROLE, NT_ROLLED,
    NT_PAUSE_TEST
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NotifyType -> AgentState mapping
NOTIFY_TO_STATE = {
    NT_START_TEST: AgentState.ALIVE,
    NT_RESUME_TEST: AgentState.ALIVE,
    NT_INVALID: AgentState.IDLE,
    NT_ABORT_TEST: AgentState.IDLE,
    NT_FINISH_TEST: AgentState.SUCCESS,
    NT_DESTROY_ROLE: AgentState.IDLE,
    NT_ARRIVED_ROLE: AgentState.SUCCESS,
    NT_ROLLED: AgentState.CRASH_OBJECT,
    NT_PAUSE_TEST: None,  # Ignore
}

# Global state variables
recv_prepare = False
start_test = False
session_id = ""


def process_notify(middleware, env):
    """
    Process Notify messages from OnSite server.

    OnSite sends Notify messages to control agent lifecycle and session state.
    This function collects all pending Notify messages and updates agent states accordingly.

    Args:
        middleware: OnSiteMiddleware instance
        env: OnSiteScenarioEnv instance
    """
    global start_test, recv_prepare

    # Collect all pending Notify messages
    notifies = middleware.recv_all_notifies()

    for notify in notifies:
        role_id = notify.role_id
        notify_type = notify.type

        logger.info(f"Received Notify: type={notify_type}, role_id={role_id}")

        # Map NotifyType to AgentState
        new_state = NOTIFY_TO_STATE.get(notify_type)

        if new_state is None:
            # Ignore this notify type
            continue

        # Handle session-level notifications
        if notify_type in [NT_ABORT_TEST, NT_FINISH_TEST]:
            logger.info(f"Session ended: {notify_type}")
            start_test = False
            recv_prepare = False
            continue
        elif notify_type == NT_START_TEST:
            logger.info(f"Session started: {notify_type}")
            start_test = True

        # Update agent state if agent exists
        if role_id in env.agent_managers:
            env.agent_managers[role_id].set_state(new_state)
            logger.info(f"Agent {role_id} state updated to {new_state}")


def get_prepare(middleware):
    """
    Receive ActorPrepare message from OnSite server.

    Args:
        middleware: OnSiteMiddleware instance

    Returns:
        tuple: (session_id, actor_id, brief_data) if received, None otherwise
    """
    global recv_prepare, session_id

    result = middleware.recv_actor_prepare()
    if result is None:
        return None

    session_id, actor_id, brief_data = result
    recv_prepare = True
    logger.info(f"Received ActorPrepare: session={session_id}, actor={actor_id}")

    return result


def send_prepare_result(middleware, actor_id):
    """
    Send ActorPrepareResult to OnSite server.

    Args:
        middleware: OnSiteMiddleware instance
        actor_id: Actor ID
    """
    middleware.send_actor_prepare_result(session_id, actor_id, result=True)
    logger.info(f"Sent ActorPrepareResult: session={session_id}")


def main_loop(env, middleware):
    """
    Main communication loop with OnSite server.

    Implements the three-phase protocol:
    1. Handshake: Wait for ActorPrepare, send ActorPrepareResult and SubRole
    2. Loop: Receive messages, step simulation, send updates
    3. Termination: Handle Notify messages for session end

    Args:
        env: OnSiteScenarioEnv instance
        middleware: OnSiteMiddleware instance
    """
    global recv_prepare, start_test

    logger.info("Starting main loop")

    while True:
        # Phase 1: Process Notify messages (at beginning of each iteration)
        process_notify(middleware, env)

        # Phase 2: Wait for ActorPrepare
        if not recv_prepare:
            get_prepare(middleware)
            time.sleep(0.1)
            continue

        # Phase 3: Send ActorPrepareResult and SubRole
        if recv_prepare and not start_test:
            send_prepare_result(middleware, middleware.actor_id)
            # Send SubRole (only session_id required)
            middleware.send_sub_role(session_id)
            logger.info("Sent SubRole, waiting for NT_START_TEST")
            time.sleep(1)
            continue

        # Phase 4: Main simulation loop
        # Receive messages from OnSite
        pub_role = middleware.recv_pub_role()
        vehicle_control = middleware.recv_vehicle_control()
        vehicle_feedback = middleware.recv_vehicle_feedback()  # Only receive, not use
        session_info = middleware.recv_session_info()  # Only receive, log

        if session_info:
            logger.debug("Received SessionInfo from OnSite")

        # Update participants from PubRole
        if pub_role:
            env.update_agents_from_pub_role(pub_role)

        # Execute simulation step
        action = vehicle_control if vehicle_control else [0.0, 0.0]
        obs, reward, terminated, truncated, info = env.step(action)

        # Get current timestamp
        current_timestamp = env.step_manager.current_timestamp

        # Send updated states to OnSite
        # 1. Send PubRole with all agent states
        all_states = env.get_all_agent_states()
        ego_state = all_states.get('actor')
        participants_states = {k: v for k, v in all_states.items() if k != 'actor'}

        if ego_state:
            middleware.send_pub_role(
                ego_state,
                participants_states,
                env.last_received_pub_role,
                current_timestamp
            )

        # 2. Send VehicleFeedback
        if ego_state:
            middleware.send_vehicle_feedback(
                ego_state,
                current_timestamp,
                vehicle_feedback  # Use received feedback for preserving fields
            )

        # 3. Send images
        if 'gaussian' in obs:
            timestamp_sec = current_timestamp / 1e6
            images_to_send = []
            for camera_name, images in obs['gaussian'].items():
                if len(images) > 0:
                    # Get the latest image
                    images_to_send.append(images[-1])
            if images_to_send:
                middleware.send_images(images_to_send, timestamp_sec)

        # Small delay to avoid busy loop
        time.sleep(0.01)


def main():
    """
    Main entry point for OnSite integration.

    Parses command-line arguments, initializes middleware and environment,
    and starts the main communication loop.
    """
    parser = argparse.ArgumentParser(description="MetaDrive OnSite Integration")
    parser.add_argument("--scene_config_directory", type=str, required=True,
                        help="Directory containing scene config files")
    parser.add_argument("--config_center", type=str, default="10.11.17.88:52009",
                        help="OnSite config center address")
    parser.add_argument("--field_id", type=str, default="unique_fieldid",
                        help="Unique field ID (must match daemon and simulator)")
    parser.add_argument("--net_interface", type=str, default="eno2",
                        help="Network interface name")
    parser.add_argument("--local_ip", type=str, default=None,
                        help="Local IP address (auto-detected if not specified)")
    parser.add_argument("--actor_id", type=str, default="metadrive_simulator",
                        help="Actor ID for this simulator")
    args = parser.parse_args()

    # Auto-detect local IP if not specified
    if args.local_ip is None:
        try:
            from api_reference.get_ip import get_ip_address
            args.local_ip = get_ip_address(args.net_interface)
            logger.info(f"Auto-detected local IP: {args.local_ip}")
        except Exception as e:
            logger.error(f"Failed to auto-detect local IP: {e}")
            logger.error("Please specify --local_ip manually")
            sys.exit(1)

    # Initialize OnSite middleware
    logger.info("Initializing OnSite middleware...")
    try:
        middleware = OnSiteMiddleware(
            config_center=args.config_center,
            field_id=args.field_id,
            net_interface=args.net_interface,
            local_ip=args.local_ip,
            actor_id=args.actor_id
        )
        logger.info("OnSite middleware initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OnSite middleware: {e}")
        sys.exit(1)

    # Initialize environment
    logger.info("Initializing MetaDrive environment...")
    try:
        # Import simulator interface
        from metadrive.simulator_interface import SimulatorInterface

        # Create model and environment
        model = SimulatorInterface()
        env_config = {
            "scene_config_directory": args.scene_config_directory,
            # Add other config as needed
        }
        env = OnSiteScenarioEnv(model, env_config)
        logger.info("MetaDrive environment initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize MetaDrive environment: {e}")
        sys.exit(1)

    # Run main loop
    try:
        main_loop(env, middleware)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        env.close()
        middleware.close()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
