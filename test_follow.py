import os
import time

import cv2
import numpy as np
from robot import Robot
from scipy.spatial.transform import Rotation as R

# robot = Robot()


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


class Controller_Follow:
    def __init__(self, robot):
        self.robot = robot
        self.class_buffer = []
        self.cnt = 0
        self.min_w = 0.75
        self.max_w = 0.9
        self.cmd_history = [0] * 100
        self.pose_buffer = []

    def get_action(self, xy, theta, class_id):
        print("xy", xy[0])
        robot = self.robot
        self.cnt += 1

        t = np.sin(self.cnt / 10 * np.pi) / 2 + 0.5
        w = t * (self.max_w - self.min_w) + self.min_w
        w = 0.88
        print(w)
        robot.grc.set_right(w)
        self.cmd_history.append(w)
        self.cmd_history = self.cmd_history[-20:]

        if self.cmd_history[-4] >= 0.88:
            # self.class_buffer.append((class_id == 2 ) * 1.0)
            self.class_buffer.append((class_id == 2 or xy[0] < 0.2) * 1.0)
            self.class_buffer = self.class_buffer[-10:]
        print("class_id", class_id)

        ratio_all_fabric = np.mean(self.class_buffer)
        # print("ratio_all_fabric", ratio_all_fabric)

        kp = -0.03
        x_current = xy[0]
        if class_id == 2:
            x_current = -0.2
        self.pose_buffer.append(x_current)
        self.pose_buffer = self.pose_buffer[-3:]
        x_current = np.mean(self.pose_buffer)
        # if x_current > 0.6:
        #     robot.grc.set_right(0.5)
        #     time.sleep(0.05)
        #     robot.grc.set_right(0.88)
        #     time.sleep(0.15)
        #     self.pose_buffer = []
        x_desired = 0.25
        # x_desired = 0.4
        print("x_current", x_current)

        # kp = 0.03
        # x_current = ratio_all_fabric
        # x_desired = 0.5

        # vel = [0, alpha * (x_current - x_desired) * kp, -0.01, 0.005 * (xy[0]), 0, 0]
        err = x_current - x_desired
        vel = [0, err * kp, -0.01, 0, 0, 0]
        # vel = [0, 0, -0.01, 0, 0, 0]
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


def test_fold(robot):
    pose_fold_1_l = robot.urc_l.getl_rt()
    pose_fold_1_r = robot.urc_r.getl_rt()
    pose_fold_1_l[:3] = [-0.462, -0.239, 0.177]
    pose_fold_1_r[:3] = [-0.158, -0.41, 0.188]
    # pose_fold_1_l = [-0.462, -0.239, 0.167, -2.21, 2.198, -0.023]
    # pose_fold_1_r = [-0.158, -0.39, 0.178, 1.219, -1.219, 1.219]

    move_together(robot, pose_fold_1_l, pose_fold_1_r)

    DX = 0.1
    DZ = 0.05
    pose_fold_2_l = pose_fold_1_l.copy()
    pose_fold_2_l[0] += DX
    pose_fold_2_l[2] -= DZ

    pose_fold_2_r = pose_fold_1_r.copy()
    pose_fold_2_r[0] -= DX
    pose_fold_2_r[2] -= DZ
    move_together(robot, pose_fold_2_l, pose_fold_2_r)

    DX = 0.12
    DZ = 0.12
    pose_fold_3_l = pose_fold_1_l.copy()
    pose_fold_3_r = pose_fold_1_r.copy()
    pose_fold_3_l[0] += DX
    pose_fold_3_l[2] -= DZ
    pose_fold_3_r[0] -= DX
    pose_fold_3_r[2] -= DZ
    move_together(robot, pose_fold_3_l, pose_fold_3_r)

    DX = 0.05
    DZ = 0.24
    pose_fold_3_l = pose_fold_1_l.copy()
    pose_fold_3_r = pose_fold_1_r.copy()
    pose_fold_3_l[0] += DX
    pose_fold_3_l[2] -= DZ
    pose_fold_3_r[0] -= DX
    pose_fold_3_r[2] -= DZ
    move_together(robot, pose_fold_3_l, pose_fold_3_r)


def move_together(robot, pose_l, pose_r, a=1, v=0.1):
    robot.urc_l.movel_nowait(pose_l, a=a, v=v)
    robot.urc_r.movel_nowait(pose_r, a=a, v=v)
    robot.urc_l.wait_following(len_p=5, threshold=5e-6)
    robot.urc_r.wait_following(len_p=5, threshold=5e-6)


def follow(robot):
    gs = robot.gs

    # c = input()
    # if c == "c":
    #     robot.grc.set_left(0.1)
    #     time.sleep(0.2)
    #     input()
    #     robot.grc.set_left(0.6)
    # robot.grc.set_left(0.6)
    #
    # # joints_vertical_l = [0.238, -1.493, 1.319, -1.392, -1.583, -1.326]
    # # joints_horizontal_r = [0.48, -1.608, 2.486, -0.888, 0.482, 1.568]
    # # robot.urc_l.movej_joints(joints_vertical_l, a=2, v=0.2, wait=True)
    # # robot.urc_r.movej_joints(joints_horizontal_r, a=2, v=0.2, wait=True)
    # #
    # robot.grc.set_right(0.5)
    # #
    # # tool_orientation = get_rotvec_vertical(-90)
    # # pose_up = np.hstack([[-0.46, -0.24, 0.33], tool_orientation])
    # # move_record(pose_up)
    # #
    # pose0_r = [-0.16, -0.63, 0.2, 1.2, -1.2, 1.19]
    # robot.urc_r.movel_wait(pose0_r)
    # time.sleep(1)

    robot.grc.set_right(0.83)
    time.sleep(1)

    for i in range(10):
        robot.gs3.tc.flag_reset_flow = True
        slip_index_realtime = robot.gs3.tc.slip_index_realtime
        print("slip_index_realtime", np.abs(slip_index_realtime))
        time.sleep(0.1)

    depth_queue = []

    tm = 0
    dt = 0.03

    a = 0.2
    vel_max = 0.04
    slip_index_realtime_thresh = 3

    tm = time.time()
    cnt = 0

    controller = Controller_Follow(robot)
    cnt_stop = 0

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
        class_id = gs.pc.class_id
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

        vel = controller.get_action(xy, theta, class_id)
        # vel = clip_vel(vel, vel_max)
        # print(vel)

        print("slip_index_realtime", np.abs(slip_index_realtime))
        if np.abs(slip_index_realtime) > 3.5:
            vel = np.array([0, 0, 0, 0, 0, 0])
            cnt_stop += 1
            if cnt_stop >= 2:
                break
        else:
            cnt_stop = 0
        # vel = np.array([0, 0, 0, 0, 0, 0])

        # print("time", time.time() - tm)
        tm = time.time()

        robot.urc_r.speedl(vel, a=a, t=dt * 4)
        time.sleep(dt)

        c = cv2.waitKey(1) & 0xFF
        if c == ord("q"):
            break


if __name__ == "__main__":
    while True:
        # pose0_l = [-0.46, -0.229, 0.329, -2.114, 2.32, 0.002]
        # robot.urc_l.movel_wait(pose0_l)
        follow(robot)
        # test_fold(robot)
        # input()
