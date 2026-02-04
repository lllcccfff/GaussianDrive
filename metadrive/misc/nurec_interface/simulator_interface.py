from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

from metadrive.misc.nurec_interface.grpc_client import NurecGrpcClient
from metadrive.misc.nurec_interface.nurec_parser import (
    load_rig_data,
    load_tracks_data,
    parse_camera_params,
    parse_ego_poses,
    parse_tracking_data,
    parse_world_to_nre,
)


class SimulatorInterface:
    def __init__(self, zNear: float = 0.0001, zFar: float = 1000.0, grpc_host: str = "localhost",
                 grpc_port: int = 50051, grpc_timeout_s: float = 60.0, resolution_scale: float = 1.0) -> None:
        self.zNear = zNear
        self.zFar = zFar
        self.resolution_scale = resolution_scale
        self._grpc = NurecGrpcClient(host=grpc_host, port=grpc_port, timeout_s=grpc_timeout_s)
        self._cached_ts: Optional[int] = None
        self._camera_models: Dict[str, Dict[str, Any]] = {}
        self._default_camera_name: Optional[str] = None
        self._world_to_nre: Optional[np.ndarray] = None

    def load_metadata(
        self,
        cfg_path: str | Path,
    ) -> Tuple[str, Any, list[int], Dict[str, Dict[str, Any]], Dict[int, list[list[float]]],
               Dict[str, Dict[str, Any]], Optional[str]]:
        cfg_path = Path(cfg_path)
        cfg = self._load_cfg(cfg_path)
        rig = load_rig_data(cfg["rig_trajectories_path"])
        self._world_to_nre = parse_world_to_nre(rig)
        camera_params, camera_models = parse_camera_params(rig, resolution_scale=self.resolution_scale)
        self._camera_models = camera_models
        self._default_camera_name = next(iter(camera_models)) if camera_models else None
        ego_poses, timestamps = parse_ego_poses(rig)
        timestamp_range = [int(min(timestamps)), int(max(timestamps))]
        tracks = load_tracks_data(cfg.get("sequence_tracks_path"))
        tracking_data = parse_tracking_data(tracks, apply_world_to_nre=False, world_to_nre=self._world_to_nre)
        bk_ground_model_path = None
        return (
            cfg["scene_name"],
            cfg,
            timestamp_range,
            camera_params,
            ego_poses,
            tracking_data,
            bk_ground_model_path,
        )

    def load_model(self, cfg: Any) -> None:
        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path:
            self._grpc.load_model(ckpt_path)
        return None

    def update_scene(self, timestamp: int, object_poses: Dict[str, Any]) -> None:
        self._cached_ts = int(timestamp)
        if not object_poses:
            return
        for object_id, pose in object_poses.items():
            pose_np = np.array(pose, dtype=np.float64)
            pose_flat = pose_np.reshape(-1).tolist()
            self._grpc.set_traffic_pose(object_id=str(object_id), pose_4x4=pose_flat)

    def render(self, K: Any, H: int, W: int, extrinsics: Any) -> np.ndarray:
        if self._cached_ts is None:
            raise ValueError("render() called before update_scene(); timestamp is required")
        k_mat = np.array(K, dtype=np.float64)
        fx = float(k_mat[0, 0])
        fy = float(k_mat[1, 1])
        cx = float(k_mat[0, 2])
        cy = float(k_mat[1, 2])
        world_to_camera = np.array(extrinsics, dtype=np.float64)
        camera_to_world = np.linalg.inv(world_to_camera)[:3, :4].reshape(-1).tolist()
        camera_name, camera_model = self._select_camera_model(W, H, cx, cy)
        camera_model_type = camera_model["type"] if camera_model else None
        ftheta_params = None
        if camera_model_type == "ftheta":
            ftheta_params = camera_model["parameters"]
        # response = self._grpc.render(
        #     camera_to_world=camera_to_world,
        #     fx=fx,
        #     fy=fy,
        #     cx=cx,
        #     cy=cy,
        #     width=int(W),
        #     height=int(H),
        #     camera_model=camera_model_type,
        #     ftheta_params=ftheta_params,
        #     time_s=float(self._cached_ts) / 1_000_000.0,
        # )
        response = self._grpc.render(
            camera_to_world=camera_to_world,
            fx=1492.82,
            fy=1492.82,
            cx=400,
            cy=300,
            width=int(800),
            height=int(600),
            camera_model='pinhole',
            ftheta_params=None,
            time_s=float(self._cached_ts) / 1_000_000.0,
        )
        rgb = np.frombuffer(response.rgb_image.rgb_data, dtype=np.uint8)
        rgb = rgb.reshape((response.rgb_image.height, response.rgb_image.width, 3))

        target_h = int(H)
        target_w = int(W)
        if rgb.shape[0] != target_h or rgb.shape[1] != target_w:
            # Top-left align; pad to bottom-right (or crop if larger).
            padded = np.zeros((target_h, target_w, 3), dtype=rgb.dtype)
            copy_h = min(target_h, rgb.shape[0])
            copy_w = min(target_w, rgb.shape[1])
            padded[:copy_h, :copy_w] = rgb[:copy_h, :copy_w]
            rgb = padded

        return rgb

    def _select_camera_model(self, width: int, height: int, cx: float, cy: float) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not self._camera_models:
            return "", None
        for name, model in self._camera_models.items():
            params = model["parameters"]
            if params.get("resolution") == [width, height] and params.get("principal_point") == [cx, cy]:
                return name, model
        default_name = self._default_camera_name or next(iter(self._camera_models))
        return default_name, self._camera_models[default_name]

    @staticmethod
    def _load_cfg(cfg_path: Path) -> Dict[str, Any]:
        if not cfg_path.exists():
            raise FileNotFoundError(f"Scene config not found: {cfg_path}")
        data = yaml.safe_load(cfg_path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Scene config must be a mapping: {cfg_path}")
        required = ("scene_name", "scene_root", "rig_trajectories_path")
        for key in required:
            if key not in data:
                raise ValueError(f"Missing {key} in scene config: {cfg_path}")
        return data
