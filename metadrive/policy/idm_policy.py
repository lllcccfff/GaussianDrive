# TODO: how to lane shifting?
# TODO: how to use map?
# TODO: parameter space
# TODO: pid steering

import math
import numpy as np
import gymnasium as gym

from metadrive.policy.base_policy import BasePolicy
from metadrive.type import MetaDriveType
from metadrive.utils.navigation_utils import nearest_front_index


class IDMPolicy(BasePolicy):
    ACC_FACTOR = 2.0
    DEACC_FACTOR = 3.0
    DELTA = 4.0
    lookahead_path_length = 50
    LANE_WIDTH = 3.5

    def __init__(self, step_manager, config=None):
        super().__init__(step_manager, config)
        self.front_distance = float(self.config["front_distance"]) if "front_distance" in self.config else 5.0
        self.react_time = float(self.config["react_time"]) if "react_time" in self.config else 1.0

        # Sample speeds (m/s)
        self.max_speed = float(
            gym.spaces.Box(low=np.array([25.0]), high=np.array([50.0]), dtype=np.float32).sample()[0]
        )
        self.max_turning_speed = float(
            gym.spaces.Box(low=np.array([10.0]), high=np.array([20.0]), dtype=np.float32).sample()[0]
        )

    def reset(self, controller, seed, state, init_state, **kwargs):
        if controller.metadrive_type != MetaDriveType.VEHICLE:
            raise ValueError("IDMPolicy can only be used for vehicle agents.")
        self.controller = controller
        self.seed(seed)

        timestamp_list = sorted(state.keys())
        self.spawn_timestamp = timestamp_list[0]

        self.destination = init_state["destination"]

    def act(self, observation, *args, **kwargs):
        nav = observation["navigation"]
        surround = observation["surrounding"]

        turn_signal = nav["turn_signal"]
        path = nav["waypoint"]
        pts = np.asarray(path, dtype=np.float32)
        cumlen = nav["cummulative_length"]

        if len(path) < 2:
            return 0.0, 0.1

        v0 = self.max_turning_speed if turn_signal != 0 else self.max_speed
        v = self.controller.speed_km_h

        # Ego pose and heading
        ego_xy = np.array([self.controller.position[0], self.controller.position[1]], dtype=np.float32)
        h = self.controller.heading
        heading_vec = np.array([float(h[0]), float(h[1])], dtype=np.float32)

        # nearest forward index for ego; if none, return zeros
        rel_all = pts - ego_xy[None, :]
        if not np.any((rel_all @ heading_vec) >= 0.0):
            self.is_arrive = True
            return 0.0, 0.0
        front_idx = int(nearest_front_index(pts, ego_xy, heading_vec))

        # Free road acceleration
        a_free = self.ACC_FACTOR * (1.0 - (v / max(v0, 1e-3)) ** self.DELTA)

        # Select closest lead object in ego frame: x>0 and |y|<= lane width/2
        lead = None
        for obj in surround:
            px = obj["position"][0]
            py = obj["position"][1]
            if px <= 0 or abs(py) > self.LANE_WIDTH * 0.5:
                continue
            if lead is None or px < lead["position"][0]:
                lead = obj

        a_int = 0.0
        # Rotation from ego to world
        R = np.array([[heading_vec[0], -heading_vec[1]], [heading_vec[1], heading_vec[0]]])
        if lead is not None:
            # Project lead to path to get arclen gap and tangent
            pos_ego = np.array([lead["position"]])
            pos_world = ego_xy + R @ pos_ego
            obj_closest_idx = np.argmin(np.sum((pts - pos_world[None, :]) ** 2, axis=1))
            delta_dist = float(cumlen[obj_closest_idx] - cumlen[front_idx])
            ego_len = self.controller.LENGTH
            obj_len = lead["size"][0]
            s0 = self.front_distance if lead["type"] == MetaDriveType.VEHICLE else 2.0

            # Gap s computed from path arclen minus half lengths
            s = max(1e-3, delta_dist - 0.5 * ego_len - 0.5 * obj_len)

            # Tangent at object's nearest point on path

            k0 = max(0, obj_closest_idx - 1)
            k1 = min(len(pts) - 1, obj_closest_idx + 1)
            t_vec = pts[k1] - pts[k0]
            path_dir = t_vec / (np.linalg.norm(t_vec) + 1e-9)

            # Tangential velocities (km/h)
            v_obj_world = R @ np.array([lead["velocity"]])
            v_obj_t = np.dot(v_obj_world, path_dir) * 3.6
            dv = max(0.0, v - v_obj_t)

            s_star = s0 + v * self.react_time + v * dv / (2.0 * math.sqrt(self.ACC_FACTOR * self.DEACC_FACTOR))
            a_int = self.ACC_FACTOR * (s_star / s) ** 2

        a_cmd = a_free - a_int

        # Steering from lookahead path
        steering = 0.0
        if len(pts) >= 3:
            target = pts[front_idx]
            to_target = target - ego_xy
            n = np.linalg.norm(to_target)
            if n > 1e-6:
                to_target /= n
                cross_z = heading_vec[0] * to_target[1] - heading_vec[1] * to_target[0]
                dot_h = np.clip(float(np.dot(heading_vec, to_target)), -1.0, 1.0)
                ang = math.atan2(cross_z, dot_h)
                ang_limit = math.radians(float(self.controller.max_steering))
                steering = float(np.clip(ang / ang_limit, -1.0, 1.0))

        return steering, a_cmd
