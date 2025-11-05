import math
import numpy as np

from metadrive.obs.observation_base import BaseObservation


class SurroundingObservation(BaseObservation):
    """
    Collect surrounding dynamic objects and express them in ego coordinates.

    observe() returns a list of dicts per surrounding object:
    - position: [x, y] in ego frame
    - velocity: [vx, vy] in ego frame
    - heading: relative heading in radians (object heading minus ego heading)
    - size: [length, width]
    """

    def __init__(self, config):
        super().__init__(config)
        self.collector = None
        self.controller = None

    def reset(self, collector, controller, **kwargs):
        self.collector = collector
        self.controller = controller

    @property
    def observation_space(self):
        # Variable-size list; return a placeholder Box to satisfy interface.
        import gymnasium as gym

        return gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)

    def observe(self):
        objs = self.collector()  # dict[name] -> controller
        ego_T = self.controller.transform
        ego_T_inv = np.linalg.inv(ego_T)
        ego_R_inv = ego_T_inv[:3, :3]
        ego_heading = self.controller.heading_theta

        surrounding = []
        for name, ctrl in objs.items():
            if ctrl is self.controller:
                continue

            # Relative transform in ego frame
            T_rel = ego_T_inv @ ctrl.transform
            pos_ego = T_rel[:2, 3]

            # Velocity transform to ego frame (use rotation part only)
            v_world = ctrl.velocity  # [vx, vy]
            v_world3 = np.array([float(v_world[0]), float(v_world[1]), 0.0], dtype=np.float32)
            v_ego3 = ego_R_inv @ v_world3
            v_ego = v_ego3[:2]

            # Relative heading
            rel_heading = self._wrap_pi(ctrl.heading_theta - ego_heading)

            # Size from controller
            length = ctrl.LENGTH
            width = ctrl.WIDTH
            size = [float(length), float(width)]

            surrounding.append(
                {
                    "position": [float(pos_ego[0]), float(pos_ego[1])],
                    "velocity": [float(v_ego[0]), float(v_ego[1])],
                    "heading": float(rel_heading),
                    "size": size,
                    "type": self.collector.metadrive_type,
                }
            )

        return surrounding

    @staticmethod
    def _wrap_pi(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def destroy(self):
        super().destroy()
        self.collector = None
        self.controller = None
