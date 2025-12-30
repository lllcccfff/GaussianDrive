"""
OnSite Dynamic Policy for MetaDrive integration.

This policy receives state updates from OnSite server and maintains dynamic simulation,
unlike ReplayPolicy which uses kinematic mode.
"""

import logging
from metadrive.policy.base_policy import BasePolicy

logger = logging.getLogger(__name__)


class OnSiteDynamicPolicy(BasePolicy):
    """
    OnSite dynamic policy: receives state updates from external source (OnSite server).

    Key differences from ReplayPolicy:
    - Does NOT call set_kinematic(True), maintains dynamic simulation
    - State is provided externally via set_state_info(), not from trajectory
    - act() returns state_info dict for controller.move()
    - Lifecycle controlled by external Notify messages, not trajectory timestamps
    """

    def reset(self, controller, seed, state, init_state, **kwargs):
        """
        Reset policy for OnSite mode.

        Args:
            controller: Vehicle controller
            seed: Random seed
            state: Not used in OnSite mode (state comes from external)
            init_state: Initial state dict with destination, spawn_timestamp, etc.
        """
        self.controller = controller
        self.seed(seed)

        # OnSite mode: no trajectory data, state provided externally
        self.trajectory = {}
        self.destination = init_state.get('destination', controller.position)
        self.static = False  # OnSite-controlled agents are not static
        self.spawn_timestamp = init_state.get('spawn_timestamp', 0)

        # CRITICAL: Do NOT call set_kinematic(True)
        # Keep dynamic mode for physics simulation
        # self.controller.set_kinematic(True)  # ReplayPolicy does this, we don't

        # Current state info (set externally via set_state_info)
        self.current_state_info = None

        logger.debug(f"OnSiteDynamicPolicy reset for {controller.name}")

    def act(self, *args, **kwargs):
        """
        Return state_info dict for controller.move().

        Returns:
            dict: state_info with transform, velocity, angular_velocity, valid
            None: if state_info not set or invalid
        """
        if self.current_state_info is None:
            logger.warning(f"OnSiteDynamicPolicy: current_state_info is None for {self.controller.name}")
            return None

        if not self.current_state_info.get('valid', False):
            logger.warning(f"OnSiteDynamicPolicy: current_state_info is invalid for {self.controller.name}")
            return None

        # Return state_info, controller will call move(state_info)
        return self.current_state_info

    def set_state_info(self, state_info):
        """
        Set current state info from external source (OnSite server).

        Args:
            state_info: dict with keys:
                - transform: 4x4 transformation matrix
                - velocity: [vx, vy, vz]
                - angular_velocity: float (yaw rate)
                - valid: bool
        """
        self.current_state_info = state_info

    @property
    def is_arrive(self):
        """
        OnSite mode: agent lifecycle controlled by Notify messages.
        Always return False to avoid destination-based termination.
        """
        return False

    @property
    def is_in_trajectory(self):
        """
        OnSite mode: no trajectory to check against.
        Always return True to avoid OUT_OF_ROAD termination.
        """
        return True

    @property
    def is_spawned(self):
        """
        OnSite mode: spawn controlled by Notify (NT_START_TEST).
        Always return False, external code uses AgentManager.set_state() to activate.
        """
        return False
