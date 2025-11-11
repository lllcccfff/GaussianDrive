import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from trajdata.utils import map_utils
import cv2
import numpy as np

def project_points_on_image(img, k, w2c, h, w, points, color=(0, 255, 0)):
    """
    Args:
        img: np.ndarray (H_img, W_img, 3) 背景图
        k: np.ndarray (3, 3) 内参矩阵
        w2c: np.ndarray (4, 4) 世界到相机变换矩阵
        h, w: int 图像尺寸，用于视口检查
        points: np.ndarray (N, 3) 世界坐标点
        color: tuple RGB 颜色

    Return:
        out_img: 投影结果图
    """

    # === 1. 将点从世界坐标转到相机坐标 ===
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    pts_cam = (R @ points.T + t.reshape(3, 1)).T  # (N,3)

    # 只保留在前方的点
    mask_front = pts_cam[:, 2] > 1e-6
    pts_cam = pts_cam[mask_front]

    # === 2. 投影到像素坐标 ===
    pts_norm = pts_cam / pts_cam[:, 2:3]
    pts_pix = (k @ pts_norm.T).T  # (N,3)
    u = pts_pix[:, 0]
    v = pts_pix[:, 1]

    # === 3. 过滤在图像外的点 ===
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[valid].astype(int)
    v = v[valid].astype(int)

    out_img = img.copy()

    # === 4. 画点 ===
    for (x, y) in zip(u, v):
        cv2.circle(out_img, (x, y), radius=2, color=color, thickness=-1)

    # === 5. 连线 ===
    if len(u) >= 2:
        pts_2d = np.stack([u, v], axis=1)
        for i in range(len(pts_2d) - 1):
            cv2.line(out_img, tuple(pts_2d[i]), tuple(pts_2d[i + 1]), color=color, thickness=1)

    return out_img

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