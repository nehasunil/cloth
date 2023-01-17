#! /usr/bin/env python
# -*- coding: utf-8

import time
import cv2
import numpy as np
from math import pi, sin, cos, asin, acos
import csv
import random
from perception.wedge.gelsight.util.Vis3D import ClassVis3D
from robot import Robot

robot = Robot()

# pose0 = np.array([-0.505 - 0.1693, -0.219, 0.235, -1.129, -1.226, 1.326])
# pose0 = np.array([-0.667, -0.196, 0.228, 1.146, -1.237, -1.227])
# pose0 = np.array([-0.581, -0.259, 0.227, 1.146, -1.237, -1.226])
# pose0 = np.array([-0.58, -0.385, 0.226, 1.146, -1.237, -1.225])
# pose0 = np.array([-0.558, -0.384, 0.235, 1.146, -1.237, -1.224])
pose0 = np.array([-0.568, -0.384, 0.253, 1.146, -1.236, -1.224])
ang = 30


def test_combined():
    gs = robot.gs

    a = 0.15
    v = 0.08
    robot.urc_l.movel_wait(pose0, a=a, v=v)
    # time.sleep(2)

    c = input()

    robot.grc.set_right(0.93)
    time.sleep(0.5)

    cnt = 0
    dt = 0.05
    pos_x = 0.5

    tm_key = time.time()
    noise_acc = 0.0
    flag_record = False
    tm = 0
    start_tm = time.time()

    vel = [0.00, 0.008, 0, 0, 0, 0]

    while True:
        img = gs.stream.image

        # get pose image
        pose_img = gs.pc.pose_img
        if pose_img is None:
            continue

        pose = gs.pc.pose
        cv2.imshow("pose", pose_img)

        if gs.pc.inContact:
            a = 0.02
            v = 0.02

            fixpoint_x = pose0[0]
            fixpoint_y = pose0[1] - 0.133
            pixel_size = 0.2e-3
            ur_pose = robot.urc_l.getl_rt()
            ur_xy = np.array(ur_pose[:2])

            x = 0.1 - pose[0] - 0.5 * (1 - 2 * pose[1]) * np.tan(pose[2])
            alpha = (
                np.arctan(ur_xy[0] - fixpoint_x)
                / (ur_xy[1] - fixpoint_y)
                * np.cos(np.pi * ang / 180)
            )

            print("x: ", x, "; input: ", x * pixel_size)

            # K = np.array([6528.5, 0.79235, 2.18017]) #10 degrees
            # K = np.array([7012, 8.865, 6.435]) #30 degrees
            # K = np.array([1383, 3.682, 3.417])
            K = np.array([862689, 42.704, 37.518])

            state = np.array([[x * pixel_size], [pose[2]], [alpha]])
            phi = -K.dot(state)

            # noise = random.random() * 0.07 - 0.02
            # a = 0.8
            # noise_acc = a * noise_acc + (1 - a) * noise
            # phi += noise_acc

            target_ur_dir = phi + alpha
            limit_phi = np.pi / 3
            target_ur_dir = max(-limit_phi, min(target_ur_dir, limit_phi))
            if abs(target_ur_dir) == limit_phi:
                print("reached phi limit")
            v_norm = 0.02
            vel = np.array(
                [
                    v_norm * sin(target_ur_dir) * cos(np.pi * ang / 180),
                    v_norm * cos(target_ur_dir),
                    v_norm * sin(target_ur_dir) * sin(np.pi * -ang / 180),
                    0,
                    0,
                    0,
                ]
            )

            # if x < -0.2:
            #     print("regrasp")
            #     rx_regrasp()

            if ur_pose[0] < -0.7:
                vel[0] = max(vel[0], 0.0)
                print("reached x limit")
            if ur_pose[0] > -0.4:
                vel[0] = min(vel[0], 0.0)
                print("reached x limit")
            if ur_pose[2] < 0.15:
                vel[2] = 0.0
                print("reached z limit")
            if ur_pose[1] > 0.34:
                print("end of workspace")
                gs.pc.inContact = False
                vel[0] = min(vel[0], 0.0)
                vel[1] = 0.0

            # print("sliding vel ", vel[0], "posx ", pos_x)

            vel = np.array(vel)
            robot.urc_l.speedl(vel, a=a, t=dt * 2)

            time.sleep(dt)

        else:
            print("no pose estimate")
            break

        c = cv2.waitKey(1) & 0xFF
        if c == ord("q"):
            break


if __name__ == "__main__":
    test_combined()
