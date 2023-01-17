#!/usr/bin/env python

import time

import cv2
import numpy as np
from robot import Robot
from scipy.spatial.transform import Rotation as R

robot = Robot()

a = 0.05
v = 0.05

tool_orientation_euler = [180, 0, 180]
tool_orientation = R.from_euler("xyz", tool_orientation_euler, degrees=True).as_rotvec()

pose0 = np.hstack([[-0.505, 0.2, 0.2], tool_orientation])  # left
# pose0 = np.hstack([[-0.13, -0.39, 0.06], tool_orientation])  # right

robot.urc_l.movel_wait(pose0, a=a, v=v)
# ---------------------------------------------

# Callback function for clicking on OpenCV window
click_point_pix = ()
camera_color_img, camera_depth_img = robot.get_camera_image()


def mouseclick_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global camera, robot, click_point_pix

        click_point_pix = (x, y)
        d = camera_depth_img[y][x]
        target_position = robot.transform_l.get_robot_from_depth(x, y, d)

        print("TARGET POS: ", target_position)
        print("CLICK POINT: ", click_point_pix)

        pose = np.hstack([target_position, tool_orientation])
        # urc.movel_wait(pose, a=a, v=v)


# Show color and depth frames
cv2.namedWindow("color")
cv2.setMouseCallback("color", mouseclick_callback)
cv2.namedWindow("depth")

while True:
    camera_color_img, camera_depth_img = robot.get_camera_image()
    bgr_data = camera_color_img  # cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
    if len(click_point_pix) != 0:
        bgr_data = cv2.circle(bgr_data, click_point_pix, 7, (0, 0, 255), 2)
    cv2.imshow("color", bgr_data)
    cv2.imshow("depth", camera_depth_img / 1100.0)

    if cv2.waitKey(1) == ord("c"):
        break

cv2.destroyAllWindows()
