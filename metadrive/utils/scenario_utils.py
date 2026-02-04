import copy
import os
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import figure

from metadrive.component.static_object.traffic_object import TrafficCone, TrafficBarrier
from metadrive.component.traffic_participants.cyclist import Cyclist
from metadrive.component.traffic_participants.pedestrian import Pedestrian
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import DATA_VERSION, DEFAULT_AGENT
from metadrive.engine import get_logger
from metadrive.type import MetaDriveType
from metadrive.utils.math import compute_angular_velocity, norm, wrap_to_pi

NP_ARRAY_DECIMAL = 3
VELOCITY_DECIMAL = 1  # velocity can not be set accurately
MIN_LENGTH_RATIO = 0.8

logger = get_logger()


def dict_recursive_remove_array_and_set(d):
    if isinstance(d, np.ndarray):
        return d.tolist()
    if isinstance(d, set):
        return tuple(d)
    if isinstance(d, dict):
        for k in d.keys():
            d[k] = dict_recursive_remove_array_and_set(d[k])
    return d


def draw_map(map_features, show=False):
    figure(figsize=(8, 6), dpi=500)
    for key, value in map_features.items():
        if MetaDriveType.is_lane(value.get("type", None)):
            plt.scatter([x[0] for x in value["polyline"]], [y[1] for y in value["polyline"]], s=0.1)
        elif value.get("type", None) == "road_edge":
            plt.scatter([x[0] for x in value["polyline"]], [y[1] for y in value["polyline"]], s=0.1, c=(0, 0, 0))
        # elif value.get("type", None) == "road_line":
        #     plt.scatter([x[0] for x in value["polyline"]], [y[1] for y in value["polyline"]], s=0.5, c=(0.8,0.8,0.8))
    if show:
        plt.show()


def get_type_from_class(obj_class):
    if issubclass(obj_class, BaseVehicle) or obj_class is BaseVehicle:
        return MetaDriveType.VEHICLE
    elif issubclass(obj_class, Pedestrian) or obj_class is Pedestrian:
        return MetaDriveType.PEDESTRIAN
    elif issubclass(obj_class, Cyclist) or obj_class is Cyclist:
        return MetaDriveType.CYCLIST
    elif issubclass(obj_class, TrafficBarrier) or obj_class is TrafficBarrier:
        return MetaDriveType.TRAFFIC_BARRIER
    elif issubclass(obj_class, TrafficCone) or obj_class is TrafficCone:
        return MetaDriveType.TRAFFIC_CONE
    else:
        return MetaDriveType.OTHER


def _convert_type_to_string(nested):
    if isinstance(nested, type):
        return (nested.__module__, nested.__name__)
    if isinstance(nested, (list, tuple)):
        return [_convert_type_to_string(v) for v in nested]
    if isinstance(nested, dict):
        return {k: _convert_type_to_string(v) for k, v in nested.items()}
    return nested


def find_light_manager_name(manager_info):
    """
    Find the light_manager in real data manager
    """
    for manager_name in manager_info:
        if "LightManager" in manager_name:
            return manager_name
    return None


def find_traffic_manager_name(manager_info):
    """
    Find the traffic_manager in real data manager
    """
    for manager_name in manager_info:
        if "TrafficManager" in manager_name and manager_name != "PGTrafficManager":
            return manager_name
    return None


def find_data_manager_name(manager_info):
    """
    Find the data_manager
    """
    for manager_name in manager_info:
        if "DataManager" in manager_name:
            return manager_name
    return None


def get_max_valid_indicis(track, current_index):
    """
    Find the invalid timestep and get the trajectory before that step
    """
    states = track["state"]
    assert states["valid"][current_index], "Current index should be valid"
    end = len(states["valid"])
    for i, valid in enumerate(states["valid"][current_index + 1:], current_index + 1):
        if not valid:
            end = i
            break
    return current_index, end


# def parse_object_state(object_dict, time_idx, check_last_state=False, sim_time_interval=0.1, include_z_position=False):
#     """
#     Parse object state of one time step
#     """
#     states = object_dict["state"]
#
#     epi_length = len(states["position"])
#     if time_idx < 0:
#         time_idx = epi_length + time_idx
#
#     if time_idx >= len(states["position"]):
#         time_idx = len(states["position"]) - 1
#     if check_last_state:
#         for current_idx in range(time_idx):
#             p_1 = states["position"][current_idx][:2]
#             p_2 = states["position"][current_idx + 1][:2]
#             if norm(p_1[0] - p_2[0], p_1[1] - p_2[1]) > 100:
#                 time_idx = current_idx
#                 break
#
#     ret = {k: v[time_idx] for k, v in states.items()}
#
#     if include_z_position:
#         ret["position"] = states["position"][time_idx]
#     else:
#         ret["position"] = states["position"][time_idx, :2]
#
#     ret["velocity"] = states["velocity"][time_idx]
#
#     ret["heading_theta"] = states["heading"][time_idx]
#
#     ret["heading"] = ret["heading_theta"]
#
#     # optional keys with scalar value:
#     for k in ["length", "width", "height"]:
#         if k in states:
#             ret[k] = float(states[k][time_idx].item())
#
#     ret["valid"] = states["valid"][time_idx]
#     if time_idx < len(states["position"]) - 1 and states["valid"][time_idx] and states["valid"][time_idx + 1]:
#         angular_velocity = compute_angular_velocity(
#             initial_heading=states["heading"][time_idx],
#             final_heading=states["heading"][time_idx + 1],
#             dt=sim_time_interval
#         )
#         ret["angular_velocity"] = angular_velocity
#     else:
#         ret["angular_velocity"] = 0
#
#     # Retrieve vehicle type
#     ret["vehicle_class"] = None
#     if "spawn_info" in object_dict["metadata"]:
#         type_module, type_cls_name = object_dict["metadata"]["spawn_info"]["type"]
#         import importlib
#         module = importlib.import_module(type_module)
#         cls = getattr(module, type_cls_name)
#         ret["vehicle_class"] = cls
#
#     return ret


def parse_full_trajectory(object_dict):
    """
    Parse object states for a whole trajectory
    """
    positions = object_dict["state"]["position"]
    index = len(positions)
    for current_idx in range(len(positions) - 1):
        p_1 = positions[current_idx][:2]
        p_2 = positions[current_idx + 1][:2]
        if norm(p_1[0] - p_2[0], p_1[1] - p_2[1]) > 100:
            index = current_idx
            break
    positions = positions[:index]
    trajectory = copy.deepcopy(positions[:, :2])

    return trajectory


def parse_object_state(poses, idx, check_last_state=True, include_z_position=False):
    """
    Parse object state from 4x4 ego-to-world transformation matrices
    matrix_list: List of 4x4 numpy arrays representing ego2world transforms
    """
    ts_list = sorted(poses.keys())

    epi_length = len(poses)
    if idx < 0:
        idx = epi_length + idx
    if idx >= epi_length:
        idx = epi_length - 1

    if check_last_state:
        for current_idx in ts_list[:-1]:
            if current_idx >= idx:
                break
            pos_1 = poses[ts_list[current_idx]][:3, 3][:2]  # Extract translation
            pos_2 = poses[ts_list[current_idx + 1]][:3, 3][:2]
            if norm(pos_1[0] - pos_2[0], pos_1[1] - pos_2[1]) > 100:
                idx = current_idx
                break

    current_matrix = poses[ts_list[idx]]

    # Extract position from translation column
    position = current_matrix[:3, 3]
    if not include_z_position:
        position = position[:2]
    if not isinstance(position, list):
        position = position.tolist()

    # Extract heading from rotation matrix (assuming Z-up)
    heading_theta = torch.arctan2(current_matrix[1, 0], current_matrix[0, 0]).item()

    # Calculate velocity from position difference
    if idx == 0:
        prev_idx = idx
        idx = idx + 1
    else:
        prev_idx = idx - 1
    sim_time_interval = (ts_list[idx] - ts_list[prev_idx]) * 1e-6
    prev_matrix = poses[ts_list[prev_idx]]
    curr_matrix = poses[ts_list[idx]]

    prev_pos = prev_matrix[:3, 3][:2]
    curr_pos = curr_matrix[:3, 3][:2]
    velocity = (curr_pos - prev_pos) / sim_time_interval

    prev_theta = torch.arctan2(prev_matrix[1, 0], prev_matrix[0, 0]).item()
    curr_theta = torch.arctan2(curr_matrix[1, 0], curr_matrix[0, 0]).item()
    angular_velocity = compute_angular_velocity(prev_theta, curr_theta, sim_time_interval)

    ret = {
        "position": position,
        "velocity": velocity,
        "heading_theta": heading_theta,
        "angular_velocity": angular_velocity,
        "transform": current_matrix,
        "valid": True,  # Assume valid if matrix exists
        "vehicle_class": None
    }

    return ret

