import cv2
import numpy as np
from scipy.interpolate import griddata


def get_z_topview(xyz_robot, xlim, ylim, scale):
    x_ws = np.arange(int(xlim[0] * scale), int(xlim[1] * scale))
    y_ws = np.arange(int(ylim[0] * scale), int(ylim[1] * scale))
    xx_ws, yy_ws = np.meshgrid(x_ws, y_ws)
    points = xyz_robot[:2].T
    # shape: (W*H, 2)
    values = xyz_robot[2]
    # shape: (W*H)
    z_ws = griddata(points * scale, values, (xx_ws, yy_ws), method="nearest")
    return z_ws


def visualize_z_ws(z_ws_blur, grasp_point):
    # scale for visualization
    z_min = -0.14
    z_max = 0.0
    z_ws_scaled = (z_ws_blur - z_min) / (z_max - z_min + 1e-6)
    z_ws_scaled = np.dstack([z_ws_scaled, z_ws_scaled, z_ws_scaled])
    z_ws_scaled[grasp_point[0], grasp_point[1], :] = [0, 0, 1]
    z_ws_scaled = np.fliplr(np.rot90(z_ws_scaled, k=1))

    # cv2.imshow("z workspace", z_ws_scaled)
    cv2.waitKey(1)


def get_highest_point(z_ws, xlim, ylim, scale, visualize, mask_corners=None):
    z_ws_blur = cv2.GaussianBlur(z_ws, (5, 5), 0)

    if mask_corners is None or np.sum(mask_corners) < 5:
        x_high, y_high = np.where(
            z_ws_blur >= (z_ws_blur.min() + (z_ws_blur.max() - z_ws_blur.min()) * 0.95)
        )
    else:
        z_ws_blur[mask_corners == 0.0] = -1000.0
        x_high, y_high = np.where(
            z_ws_blur >= (z_ws_blur.min() + (z_ws_blur.max() - z_ws_blur.min()) * 0.95)
        )

    ind = np.random.randint(len(x_high))
    grasp_point = (x_high[ind], y_high[ind])

    if visualize:
        visualize_z_ws(z_ws_blur, grasp_point)

    xyz_robot_highest = [
        grasp_point[1] / scale + xlim[0],
        grasp_point[0] / scale + ylim[0],
        z_ws[grasp_point],
    ]
    return xyz_robot_highest
