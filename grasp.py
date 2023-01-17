#!/usr/bin/env python

import os
import time

import cv2
import numpy as np
from robot import Robot
from scipy.spatial.transform import Rotation as R
from util import get_z_topview, get_highest_point

robot = Robot()


def move_away():
    robot.urc_l.movel_wait(robot.pose_away_l)
    robot.urc_r.movel_wait(robot.pose_away_r)


def setup_robot():
    # move_away()

    tool_orientation = get_rotvec_vertical(180)
    pose_up = np.hstack([[-0.46, -0.24, 0.18], tool_orientation])

    acc = 0.2
    vel = 0.2
    robot.urc_l.movel_wait(pose_up, a=acc, v=vel)
    return pose_up


def get_rotvec_vertical(theta):
    # Args: theta in degrees
    tool_orientation_euler = [180, 0, 180]
    tool_orientation_euler[2] = theta
    tool_orientation = R.from_euler(
        "xyz", tool_orientation_euler, degrees=True
    ).as_rotvec()
    return tool_orientation


def move_record(pose, a=0.2, v=0.2):
    robot.urc_l.movel_nowait(pose, a=a, v=v)
    robot.urc_l.clear_history()
    while True:
        if robot.urc_l.check_stopped():
            break
        time.sleep(0.1)


def close_gripper_l():
    robot.grc.set_left(0.7)
    time.sleep(0.5)
    print("Closing gripper " * 20)


def open_gripper_l():
    robot.grc.set_left(0.0)
    time.sleep(0.5)
    print("Opening gripper " * 20)


def visualize_crop(camera_color_img, crop_x, crop_y):
    color_crop = camera_color_img[crop_y[0] : crop_y[1], crop_x[0] : crop_x[1]]
    cv2.imshow("color_crop", color_crop)
    cv2.waitKey(1)


def get_highest_robot(camera_color_img, camera_depth_img, visualize):
    crop_x = [700, 1100]
    crop_y = [400, 800]

    if visualize:
        visualize_crop(camera_color_img, crop_x, crop_y)

    xyz_robot = robot.transform_l.get_robot_from_depth_array(
        camera_depth_img, crop_x, crop_y
    )

    xlim = [-0.65, -0.30]
    ylim = [-0.40, -0.05]
    scale = 100
    z_ws = get_z_topview(xyz_robot, xlim, ylim, scale)
    xyz_robot_highest = get_highest_point(z_ws, xlim, ylim, scale, visualize)

    return xyz_robot_highest


def pick_highest(camera_color_img, camera_depth_img, pose_up, visualize=True):
    tool_orientation = pose_up[3:]
    xyz_robot_highest = get_highest_robot(camera_color_img, camera_depth_img, visualize)
    xyz_robot_highest[2] -= 0.02
    if xyz_robot_highest[2] < -0.088:
        print("WARNNING: Reach Z Limit, set to minimal Z")
        xyz_robot_highest[2] = -0.088
    pose = np.hstack([xyz_robot_highest, tool_orientation])
    print("move to highest point", pose)

    move_record(pose)


def home(pose_up):
    theta = np.random.randint(180) + 90
    tool_orientation = get_rotvec_vertical(theta)

    random_perturb_rng = 0.05
    pose_up = np.hstack([[-0.46, -0.24, 0.18], tool_orientation])
    pose_up[0] += np.random.uniform(-0.5, 0.5) * random_perturb_rng
    pose_up[1] += np.random.uniform(-0.5, 0.5) * random_perturb_rng

    move_record(pose_up)
    return pose_up


def rotate(pose_up):
    theta = np.random.randint(180) + 90
    tool_orientation = get_rotvec_vertical(theta)
    pose_up = np.hstack([pose_up[:3], tool_orientation])
    move_record(pose_up)
    return pose_up


def main():
    pose_up = setup_robot()

    seq_id = 0
    sequence = ["pick_highest", "close_gripper_l", "home", "rotate", "open_gripper_l"]

    while True:
        if seq_id % len(sequence) == 0:
            time.sleep(0.5)

        camera_color_img, camera_depth_img = robot.camera.get_image()

        cv2.imshow("color", camera_color_img)
        cv2.imshow("depth", camera_depth_img / 1100.0)
        cv2.waitKey(1)

        c = sequence[seq_id % len(sequence)]
        seq_id += 1

        if c == "close_gripper_l":
            close_gripper_l()
        if c == "open_gripper_l":
            open_gripper_l()
        elif c == "home":
            pose_up = home(pose_up)
        elif c == "pick_highest":
            pick_highest(camera_color_img, camera_depth_img, pose_up)
        elif c == "rotate":
            pose_up = rotate(pose_up)


if __name__ == "__main__":
    main()
