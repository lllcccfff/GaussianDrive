import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from trajdata.utils import map_utils

def render_vehicle_trajectories(vec_map,
                                trajectories,
                                vehicle_boxes,
                                vehicle_colors,
                                box_min,
                                box_max,
                                resolution=2.0,
                                figsize=(8, 8)):
    """
    Args:
        vec_map (VectorMap): trajdata.maps.vec_map.VectorMap 实例。
        trajectories (Sequence[np.ndarray]): 每条轨迹 shape=(T_i, 2) 的世界坐标序列。
        vehicle_boxes (Sequence[Tuple[float, float]]): 每辆车的 (length, width)。
        vehicle_colors (Sequence[str]): 每辆车对应的颜色，可为 Matplotlib 颜色字符串。
        box_min (Tuple[float, float]): 想要截取的世界坐标左下角 (x_min, y_min)。
        box_max (Tuple[float, float]): 想要截取的世界坐标右上角 (x_max, y_max)。
        resolution (float): 栅格化时的像素/米，越大越清晰。
        figsize (Tuple[int, int]): Matplotlib figure 大小。
    """
    assert len(trajectories) == len(vehicle_boxes) == len(vehicle_colors), \
        "输入的轨迹/包络尺寸/颜色数量需要一致"

    # 1) 栅格化整张地图并获取世界→像素的齐次变换矩阵 (vec_map.rasterize, see src/trajdata/maps/vec_map.py:449-563)
    map_img, raster_from_world = vec_map.rasterize(
        resolution=resolution,
        return_tf_mat=True,
        incl_centerlines=True,
        incl_lane_edges=True,
        incl_lane_area=True,
    )

    # 2) 把感兴趣的世界窗口映射到像素坐标，裁掉不需要的内容
    bbox_corners = np.array(
        [
            [box_min[0], box_min[1]],
            [box_max[0], box_min[1]],
            [box_max[0], box_max[1]],
            [box_min[0], box_max[1]],
        ]
    )
    bbox_pix = map_utils.transform_points(bbox_corners, raster_from_world)
    x0, x1 = np.floor(bbox_pix[:, 0].min()), np.ceil(bbox_pix[:, 0].max())
    y0, y1 = np.floor(bbox_pix[:, 1].min()), np.ceil(bbox_pix[:, 1].max())
    x0, x1 = int(max(x0, 0)), int(min(x1, map_img.shape[1]))
    y0, y1 = int(max(y0, 0)), int(min(y1, map_img.shape[0]))
    cropped_img = map_img[y0:y1, x0:x1]

    def world_to_cropped_pix(points_xy: np.ndarray) -> np.ndarray:
        pts = map_utils.transform_points(points_xy, raster_from_world)
        pts[:, 0] -= x0
        pts[:, 1] -= y0
        return pts

    def rectangle_from_center(center_xy: np.ndarray, length: float, width: float, heading: float):
        # 先在局部坐标系生成矩形，再旋转到指定朝向
        rect = np.array(
            [
                [-length / 2, -width / 2],
                [-length / 2, width / 2],
                [length / 2, width / 2],
                [length / 2, -width / 2],
            ]
        )
        rot = np.array(
            [
                [np.cos(heading), -np.sin(heading)],
                [np.sin(heading), np.cos(heading)],
            ]
        )
        return rect @ rot.T + center_xy

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(cropped_img, origin="lower")

    for traj, (length, width), color in zip(trajectories, vehicle_boxes, vehicle_colors):
        traj = np.asarray(traj, dtype=np.float32)
        if traj.shape[0] == 0:
            continue

        # 3) 起点矩形
        start = traj[0]
        if traj.shape[0] > 1:
            heading = np.arctan2(traj[1, 1] - traj[0, 1], traj[1, 0] - traj[0, 0])
        else:
            heading = 0.0

        rect_world = rectangle_from_center(start, length, width, heading)
        rect_pix = world_to_cropped_pix(rect_world)
        ax.add_patch(
            Polygon(rect_pix, closed=True, facecolor=color, alpha=0.6, edgecolor="k", linewidth=1.0, zorder=3)
        )

        # 4) 轨迹折线
        traj_pix = world_to_cropped_pix(traj)
        ax.plot(traj_pix[:, 0], traj_pix[:, 1], color=color, linewidth=2.5, zorder=4)
        ax.scatter(traj_pix[0, 0], traj_pix[0, 1], color=color, s=30, zorder=5)

    ax.set_xlim(0, cropped_img.shape[1])
    ax.set_ylim(0, cropped_img.shape[0])
    ax.axis("off")
    return fig, ax