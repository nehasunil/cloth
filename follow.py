import os
import time

import cv2
import numpy as np
from robot import Robot
from scipy.spatial.transform import Rotation as R

robot = Robot()


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


def clip_vel(vel, vel_max):
    v_norm = np.sum(vel[:2] ** 2) ** 0.5
    if v_norm > vel_max:
        vel[:2] = vel[:2] / v_norm * vel_max
    return vel


curr_width = 0.87
open_counter = 0
closed_counter = 0
perception_threshold_width = 0.89
err = 0
theta_last = 0


def follow(robot):
    gs = robot.gs

    # joints_vertical_l = [0.238, -1.493, 1.319, -1.392, -1.583, -1.326]
    # joints_horizontal_r = [0.48, -1.608, 2.486, -0.888, 0.482, 1.568]
    # robot.urc_l.movej_joints(joints_vertical_l, a=2, v=0.2, wait=True)
    # robot.urc_r.movej_joints(joints_horizontal_r, a=2, v=0.2, wait=True)

    tool_orientation = get_rotvec_vertical(-90)
    pose_up = np.hstack([[-0.46, -0.24, 0.33], tool_orientation])
    move_record(pose_up)

    pose0_r = [-0.16, -0.63, 0.2, 1.2, -1.2, 1.19]
    robot.urc_r.movel_wait(pose0_r)

    # grc.follow_gripper_pos = 0.97
    robot.grc.set_right(0.87)
    perception_threshold_width = 0.88
    delta_width = 0.02

    time.sleep(3)

    depth_queue = []

    tm = 0
    dt = 0.03

    a = 0.2
    vel_max = 0.04

    tm = time.time()
    cnt = 0

    def get_action(xy, theta):
        global curr_width, perception_threshold_width, open_counter, err, theta_last, closed_counter
        kp = -0.03
        kp_th = 0.08
        kp_w = -0.20
        alpha = 1

        # kp = -0.06

        # print("xy", xy)
        x_current = xy[0]
        x_desired = 0.3
        # x_desired = 0.7

        if curr_width < perception_threshold_width:
            closed_counter = 0
            open_counter += 1
            if open_counter > 13:
                curr_width += 0.8
                alpha = 0.5
                print("CLOSING", curr_width)
            else:
                print("OPENING", curr_width)
                curr_width = curr_width + kp_w * np.abs(err)
        else:
            open_counter = 0
            closed_counter += 1
            if closed_counter > 7:
                print("OPENING", curr_width)
                theta_last = theta
                err = x_current - x_desired
                curr_width = curr_width + kp_w * np.abs(err)

        robot.grc.set_right(np.clip(curr_width, 0.75, 0.9))

        # vel = [0, alpha * (x_current - x_desired) * kp, -0.01, 0.005 * (xy[0]), 0, 0]
        vel = [0, alpha * err * kp, -0.01, kp_th * theta_last, 0, 0]
        vel = np.array(vel)

        # Workspace Bounds
        ur_pose = robot.urc_r.getl_rt()
        if ur_pose[2] < -0.04:
            vel[2] = 0.0
        if ur_pose[1] < -0.7:
            vel[1] = max(vel[1], 0.0)
        if ur_pose[1] > -0.4:
            vel[1] = min(vel[1], 0.0)

        return vel

    while True:
        img = gs.stream.image

        # get pose image
        pose_img = gs.pc.pose_img
        # pose_img = gs.pc.frame_large
        if pose_img is None:
            continue

        slip_index_realtime = robot.gs3.tc.slip_index_realtime
        # print("slip_index_realtime", slip_index_realtime)
        tracking_img = robot.gs3.tc.tracking_img
        if tracking_img is not None:
            cv2.imshow("tracking", tracking_img)
            cv2.waitKey(1)

        pose = gs.pc.pose
        cv2.imshow("pose", pose_img)
        cv2.waitKey(1)

        vel = np.array([0, 0, -0.01, 0, 0, 0])

        # if gs.pc.inContact:
        if True:

            if pose is not None:
                xy = [pose[0], pose[1]]
                theta = pose[2]
            else:
                gs.pc.inContact = False
                print("no pose estimate")
                print("log saved: ", logger.save_logs())
                continue

        # vel = clip_vel(vel, vel_max)
        vel = clip_vel(get_action(xy, theta), vel_max)
        # print(vel)

        if np.abs(slip_index_realtime) > 0.6:
            vel = np.array([0, 0, 0, 0, 0, 0])

        # print("time", time.time() - tm)
        tm = time.time()

        robot.urc_r.speedl(vel, a=a, t=dt * 2)
        time.sleep(dt)

        c = cv2.waitKey(1) & 0xFF
        if c == ord("q"):
            break


if __name__ == "__main__":
    follow(robot)
