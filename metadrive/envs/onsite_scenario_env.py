"""
OnSite Scenario Environment for MetaDrive.

This environment extends ScenarioEnv to support OnSite integration,
providing helper methods for state synchronization with OnSite server.
"""

import logging
import numpy as np
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.manager.agent_manager import AgentState

logger = logging.getLogger(__name__)


class OnSiteScenarioEnv(ScenarioEnv):
    """
    OnSite-integrated ScenarioEnv.

    Key features:
    - Helper methods for extracting agent states for OnSite messages
    - Helper methods for updating agents from OnSite messages
    - Middleware is managed externally, not held by env
    """

    def __init__(self, model, config=None):
        super().__init__(model, config)
        # Cache for last received PubRole (for preserving fields)
        self.last_received_pub_role = None

    def get_agent_state_dict(self, agent_name):
        """
        Get agent state as dictionary for OnSite message conversion.

        Args:
            agent_name: Name of the agent

        Returns:
            dict: Agent state with position, velocity, heading, etc.
        """
        if agent_name not in self.agent_managers:
            return None

        agent_mgr = self.agent_managers[agent_name]
        if agent_mgr.state != AgentState.ALIVE:
            return None

        vehicle = agent_mgr.controller

        return {
            'position': vehicle.position,
            'velocity': vehicle.velocity,
            'heading_theta': vehicle.heading_theta,
            'angular_velocity': vehicle.angular_velocity,
            'length': vehicle.LENGTH,
            'width': vehicle.WIDTH,
            'height': vehicle.HEIGHT,
            'steering_wheel_angle': vehicle.get_steering_wheel_angle(),
            'steering_wheel_speed': vehicle.get_steering_wheel_speed(),
            'left_directive_wheel_angle': vehicle.get_left_directive_wheel_angle(),
            'right_directive_wheel_angle': vehicle.get_right_directive_wheel_angle(),
            'throttle_brake': vehicle.throttle_brake,
            'speed': vehicle.speed,
            'longitudinal_acceleration': vehicle.get_longitudinal_acceleration(),
            'front_left_wheel_speed': vehicle.get_front_left_wheel_speed(),
            'front_right_wheel_speed': vehicle.get_front_right_wheel_speed(),
            'rear_left_wheel_speed': vehicle.get_rear_left_wheel_speed(),
            'rear_right_wheel_speed': vehicle.get_rear_right_wheel_speed(),
        }

    def get_all_agent_states(self):
        """
        Get all agent states as dictionary.

        Returns:
            dict: {agent_name: state_dict}
        """
        states = {}
        for agent_name in self.agent_managers.keys():
            state = self.get_agent_state_dict(agent_name)
            if state is not None:
                states[agent_name] = state
        return states

    def update_agent_from_pub_role_single(self, agent_name, role):
        """
        Update a single agent from PubRole SingleRole message.

        Args:
            agent_name: Name of the agent
            role: SingleRole proto message
        """
        if agent_name not in self.agent_managers:
            logger.warning(f"Agent {agent_name} not found in agent_managers")
            return

        agent_mgr = self.agent_managers[agent_name]
        if agent_mgr.state != AgentState.ALIVE:
            logger.debug(f"Agent {agent_name} not alive, skipping update")
            return

        # Convert quaternion and position to transform matrix
        from metadrive.middleware.onsite_middleware import OnSiteMiddleware
        middleware = OnSiteMiddleware.__new__(OnSiteMiddleware)  # Create instance without __init__

        transform = middleware._quaternion_to_matrix(
            role.box.bottom_center,
            role.box.rotation
        )

        # Create state_info for policy
        state_info = {
            'transform': transform,
            'velocity': [
                role.linear_speed.x,
                role.linear_speed.y,
                role.linear_speed.z
            ],
            'angular_velocity': role.angular_speed.z,
            'valid': True
        }

        # Update policy state
        agent_mgr.policy.set_state_info(state_info)

    def update_agents_from_pub_role(self, pub_role):
        """
        Update all agents from PubRole message.

        Args:
            pub_role: PubRole proto message
        """
        # Cache the received PubRole
        self.last_received_pub_role = pub_role

        # Update each agent
        for role in pub_role.s_roles:
            if role.id in self.agent_managers and role.id != 'actor':
                self.update_agent_from_pub_role_single(role.id, role)
