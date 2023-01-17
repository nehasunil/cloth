import os
import time

import matplotlib.pyplot as plt
import numpy as np
from robot import Robot


def test_flip(robot):

    # pos_list = [
    #     [0, 20, 0.4, 0.4],
    #     [-5, 5, 1, 1],
    #     [-20, 0, 0.4, 0.4],
    #     [0, 0, 0, 0],
    # ]
    pos_list = [
        [0, 20, 0.0, 0.0],
        [-20, 0, 0, 0],
    ]

    k = -1
    for i in range(10000):
        c = input()
        robot.grc.follow_dc_pos = [pos_list[k][0], pos_list[k][1]]
        robot.grc.follow_gripper_pos = [pos_list[k][2], pos_list[k][3]]

        if c == "o":
            k = -1
        else:
            k = (k + 1) % len(pos_list)


def test_continuous(robot):
    gripper_width = 0
    inc = 0.02

    for i in range(10000):
        time.sleep(1.0 / 30)
        robot.grc.follow_gripper_pos = [gripper_width, gripper_width]
        gripper_width = gripper_width + inc

        if gripper_width > 1 or gripper_width < 0:
            inc *= -1
            gripper_width = gripper_width + inc


def set_gripper_r(robot, width):
    gripper_id = 1
    width_cmd = np.clip(width, 0, 1)
    robot.grc.set_right(width)


def close_gripper_l(robot):
    robot.grc.set_left(0.7)
    robot.grc.follow_dc_pos = [-5, 5]
    # robot.grc.follow_dc_pos = [0, 10]
    time.sleep(0.5)


def open_gripper_l(robot):
    robot.grc.set_left(0.0)
    time.sleep(0.5)


def test_gripper(robot):
    while True:
        # set_gripper_r(robot, 0)
        open_gripper_l(robot)
        c = input()
        # set_gripper_r(robot, 0.1)
        # close_gripper_l(robot)
        robot.grc.set_left(0.35)
        # set_gripper_r(robot, 0.95)
        c = input()


def test_gripper_l(robot):
    while True:
        open_gripper_l(robot)
        c = input()
        close_gripper_l(robot)
        c = input()


def main():
    robot = Robot()
    test_flip(robot)
    # test_continuous(robot)
    # test_gripper(robot)
    # test_gripper_l(robot)
    del robot


if __name__ == "__main__":
    main()
