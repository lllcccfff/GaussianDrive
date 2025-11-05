import gymnasium as gym
import numpy as np

from metadrive.obs.observation_base import BaseObservation


class StateObservation(BaseObservation):
    """
    Simple state observation returning a dict with:
    - position: [x, y]
    - velocity: [vx, vy]
    """

    def __init__(self, config=None):
        super().__init__(config or {})
        self.controller = None
        # Generous bounds for meters and m/s
        self._pos_low = -1e6
        self._pos_high = 1e6
        self._vel_low = -1e3
        self._vel_high = 1e3

    def reset(self, controller, seed=None, **kwargs):
        self.controller = controller

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "position": gym.spaces.Box(self._pos_low, self._pos_high, shape=(2,), dtype=np.float32),
                "velocity": gym.spaces.Box(self._vel_low, self._vel_high, shape=(2,), dtype=np.float32),
            }
        )

    def observe(self):
        p = self.controller.position
        v = self.controller.velocity
        return {
            "position": np.array([p[0], p[1]], dtype=np.float32),
            "velocity": np.array([v[0], v[1]], dtype=np.float32),
        }

    def destroy(self):
        self.controller = None
        super().destroy()
