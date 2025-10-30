import math
import numpy as np
import gymnasium as gym
from trajdata import VectorMap
from metadrive.obs.observation_base import BaseObservation
from metadrive.base_class.randomizable import Randomizable
from metadrive.utils.navigation_utils import nearest_front_index


class NavigationObservation(BaseObservation, Randomizable):
    trajdata_map: VectorMap

    def __init__(self, config):
        BaseObservation.__init__(self, config)
        Randomizable.__init__(self, None)
        self.navigating_type = config.get("navigating_type", "expert_following")  # lane_following, destination_following, expert_following
        self.early_signal_distance = float(config.get("early_signal_distance", 10.0))  # meters
        self.turn_threshold_deg = float(config.get("turn_threshold_deg", 25.0))  # degrees

        self.controller = None
        self.trajdata_map = None
        self.init_state = None
        self.state = None

        self._path_xy = None
        self._path_cumlen = None

    def reset(self, trajdata_map: VectorMap, init_state, state, controller, timestamp_range=None, seed=None, **kwargs):
        if self.navigating_type in ["lane_following", "destination_following"]:
            assert isinstance(trajdata_map, VectorMap), "trajdata_map must be provided for lane_following or destination_following navigation type."

        if seed is not None:
            self.seed(int(seed))

        self.controller = controller
        self.trajdata_map = trajdata_map
        self.init_state = init_state
        self.state = state

        if self.navigating_type == "expert_following":
            self._build_expert_path()
        elif self.navigating_type == "lane_following":
            self._build_lane_follow_path()
        elif self.navigating_type == "destination_following":
            self._build_destination_path()
        else:
            raise ValueError(f"Unknown navigating_type: {self.navigating_type}")
        
        self.destination = self._path_xy[-1] if self._path_xy is not None else init_state["destination"]

    def observe(self):
        return {
            'map': self.map, 
            'turn_signal': self._get_turn_signal(), 
            'waypoint': self._path_xy,
            'cummulative_length': self._path_cumlen
        }
    
    def _get_turn_signal(self):
        if len(self._path_xy) < 3:
            return 0

        ego_xy = self._vehicle_xy(self.controller)
        heading_vec = self._ego_heading_vec(self.controller)
        i0 = self.nearest_front_index(self._path_xy, ego_xy, heading_vec)
        if i0 >= len(self._path_xy):
            return 0
        
        j = self._first_index_by_arclen(self._path_cumlen, i0, self.early_signal_distance)
        if j ==  len(self._path_cumlen) or j == 0:
            return 0

        thr = math.radians(self.turn_threshold_deg)
        for k in range(j - 1, i0 + 1, -1):
            p0 = self._path_xy[k - 1]
            p1 = self._path_xy[k]
            p2 = self._path_xy[k + 1]
            v1 = p1 - p0
            v2 = p2 - p1
            if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                continue
            ang = self._signed_angle(v1, v2)
            if abs(ang) >= thr:
                return -1 if ang > 0 else 1
        return 0

    @property
    def observation_space(self):
        return gym.spaces.Discrete(3)

    def destroy(self):
        self._path_xy = None
        self._path_cumlen = None
        self.controller = None
        self.trajdata_map = None
        self.init_state = None
        self.state = None

    # ---------- path builders ----------
    def _build_expert_path(self):
        points = []
        for ts in sorted(self.state.keys()):
            pos = self.state[ts]["position"]
            x, y = float(pos[0]), float(pos[1])
            points.append([x, y])
        self._set_path(points)

    def _build_lane_follow_path(self):
        spawn_xy = self._xy2(self.init_state["spawn_position"])
        spawn_heading = float(self.init_state["spawn_heading"])

        start_lane = self.trajdata_map.nearest_lane_id(spawn_xy, spawn_heading)
        succs = self.trajdata_map.successors(start_lane)
        if len(succs) > 0:
            next_lane = self.np_random.choice(succs)
            lane_seq = [start_lane, next_lane]
        else:
            lane_seq = [start_lane]
        path_pts = self._concat_centerlines(lane_seq, spawn_xy, spawn_heading)
        self._set_path(path_pts)

    def _build_destination_path(self):
        spawn_xy = self._xy2(self.init_state["spawn_position"])
        spawn_heading = float(self.init_state["spawn_heading"])
        dest_xy = self._xy2(self.init_state["destination"])

        start_lane = self.trajdata_map.nearest_lane_id(spawn_xy, spawn_heading)
        goal_lane = self.trajdata_map.nearest_lane_id(dest_xy, None)

        if start_lane == goal_lane:
            lane_seq = [start_lane]
        else:
            lane_seq = self._bfs_lane_seq(start_lane, goal_lane)
            if len(lane_seq) == 0:
                lane_seq = [start_lane]
        path_pts = self._concat_centerlines(lane_seq, spawn_xy, spawn_heading)
        self._set_path(path_pts)

    # ---------- small utils ----------
    @staticmethod
    def _vehicle_xy(vehicle):
        pos = vehicle.position
        return np.array([float(pos[0]), float(pos[1])], dtype=np.float32)

    @staticmethod
    def _xy2(p):
        return float(p[0]), float(p[1])

    def _set_path(self, pts):
        path = np.asarray(pts, dtype=np.float32)
        if len(path) < 3:
            self._path_xy = None
            self._path_cumlen = None
            return
        seg = np.linalg.norm(path[1:] - path[:-1], axis=1)
        self._path_xy = path
        self._path_cumlen = np.concatenate([[0.0], np.cumsum(seg)])

    @staticmethod
    def _first_index_by_arclen(cumlen, i0, ahead_len):
        target = cumlen[i0] + max(0.0, ahead_len)
        idx = np.searchsorted(cumlen, target, side="right")
        return int(idx)
    
    @staticmethod
    def _ego_heading_vec(vehicle):
        h = vehicle.heading  # (cos, sin)
        return np.array([float(h[0]), float(h[1])], dtype=np.float32)

    @staticmethod
    def _signed_angle(v1, v2):
        v1n = v1 / np.linalg.norm(v1)
        v2n = v2 / np.linalg.norm(v2)
        dot = np.clip(float(np.dot(v1n, v2n)), -1.0, 1.0)
        ang = math.acos(dot)
        cross_z = v1n[0] * v2n[1] - v1n[1] * v2n[0]
        return ang if cross_z > 0 else -ang

    def _concat_centerlines(self, lane_seq, start_xy, start_heading):
        pts = []
        for idx, lane_id in enumerate(lane_seq):
            cl = np.asarray(self.trajdata_map.lane_centerline(lane_id), dtype=np.float32)
            if idx == 0:
                heading_vec = np.array([math.cos(start_heading), math.sin(start_heading)], dtype=np.float32)
                start_idx = nearest_front_index(cl, np.asarray(start_xy), heading_vec)
                cl = cl[start_idx:]
            if len(pts) > 0 and len(cl) > 0:
                if np.allclose(pts[-1], cl[0]):
                    pts.extend(cl[1:])
                else:
                    pts.extend(cl)
            else:
                pts.extend(cl)
        return pts

    def _bfs_lane_seq(self, start_lane, goal_lane):
        if start_lane == goal_lane:
            return [start_lane]
        from collections import deque
        q = deque([start_lane])
        parent = {start_lane: None}
        visited = {start_lane}
        while len(q) > 0:
            u = q.popleft()
            for v in self.trajdata_map.successors(u):
                if v in visited:
                    continue
                parent[v] = u
                if v == goal_lane:
                    seq = [v]
                    while parent[seq[-1]] is not None:
                        seq.append(parent[seq[-1]])
                    seq.reverse()
                    return seq
                visited.add(v)
                q.append(v)
        return []
