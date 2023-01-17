import time

import numpy as np
from robot import Robot
from scipy.spatial.transform import Rotation as R


def setup_robot():
    robot = Robot()
    return robot


def format_pose(pose):
    return "[" + ", ".join([str(int(p * 1000) / 1000.0) for p in pose]) + "]"


def get_poses(robot):
    print("urc_l pose", format_pose(robot.urc_l.getl_rt()))
    print("urc_r pose", format_pose(robot.urc_r.getl_rt()))


def test_thread(robot):
    pose0_l = np.array([-0.41, -0.13, 0.15, 0.0, 3.14, 0.0])
    robot.urc_l.movel_wait(pose0_l)

    # pose0_r = np.array([0.07, -0.41, 0.15, 0.0, 3.14, 0.0])
    pose0_r = np.array([-0.02, -0.41, 0.15, -2.2, 2.2, 0.0])
    robot.urc_r.movel_wait(pose0_r)

    pose_l = pose0_l.copy()
    pose_r = pose0_r.copy()

    for i in range(480):
        pose_l[2] = pose_l[2] + np.sin(i / 60 * np.pi) * 0.002
        pose_r[2] = pose_r[2] + np.sin(i / 60 * np.pi) * 0.002
        robot.urc_l.pose_following = pose_l
        robot.urc_r.pose_following = pose_r
        time.sleep(0.01)


def test_movel(robot):
    # pose0_r = np.array([-0.13, -0.4, 0.07, 0.0, np.pi / 2, 0.0])
    pose_away_r = np.array([-0.42, -0.16, 0.05, -2.2, 2.2, 0.0])

    euler_angle_yxz = [-90, 0, 90]
    tool_orientation = R.from_euler("yxz", euler_angle_yxz, degrees=True).as_rotvec()
    pose0_r = np.hstack([[-0.11, -0.45, 0.29], tool_orientation])

    robot.urc_r.movel_wait(pose0_r)
    # robot.urc_r.movel_wait(pose_away_r)


def home(robot):
    # pose0_l = np.array([-0.41, -0.13, 0.15, 0.0, 3.14, 0.0])
    # pose0_r = np.array([-0.02, -0.41, 0.15, -2.2, 2.2, 0.0])
    pose_away_l = np.array([-0.39, 0.16, 0.07, 2.22, 2.22, 0.0])
    pose_away_r = np.array([-0.42, -0.16, 0.05, -2.2, 2.2, 0.0])

    robot.urc_l.movel_wait(pose_away_l)
    robot.urc_r.movel_wait(pose_away_r)


def test_transformation(robot):
    pose_l = robot.urc_l.getl_rt()
    L_p = pose_l[:3]
    R_p = robot.transform_l2r(L_p)
    print("L_p", L_p)
    print("R_p", R_p)


def get_joints(robot):
    joints_l = robot.urc_l.rob.getj()
    print(format_pose(joints_l))
    joints_r = robot.urc_r.rob.getj()
    print(format_pose(joints_r))


def test_orientation_transition(robot):
    pose_vertical_r = [-0.2, -0.3, 0.03, 2.22, 2.22, 0.00]
    pose_horizontal_r = np.array([-0.16, -0.42, 0.12, 1.2, -1.2, 1.2])

    joints_vertical_r = [0.672, -1.654, 2.417, -2.331, -1.575, 0.665]
    joints_horizontal_r = [0.48, -1.608, 2.486, -0.888, 0.482, 1.568]

    # a = 3
    # v = 1

    a = 1
    v = 0.3

    robot.urc_r.movej_joints(joints_vertical_r, a=a, v=v)
    robot.urc_r.movej_joints(joints_horizontal_r, a=a, v=v)
    get_joints(robot)

    # robot.urc_r.movej_pose(pose_vertical_r, a=a, v=v)
    # robot.urc_r.movej_pose(pose_horizontal_r, a=a, v=v)


def test_orientation_transition_l(robot):
    pose_horizontal_l = [-0.51, 0.05, 0.17, 1.22, 1.22, -1.22]
    pose_vertical_l = [-0.46, -0.23, 0.33, 0.0, 3.14, 0.0]

    joints_vertical_l = [0.238, -1.493, 1.319, -1.392, -1.583, -1.326]
    joints_horizontal_l = [-0.728, -0.988, 1.494, -0.533, -0.721, -1.552]

    # a = 3
    # v = 1

    a = 1
    v = 0.1

    robot.urc_l.movej_joints(joints_horizontal_l, a=a, v=v)
    robot.urc_l.movej_joints(joints_vertical_l, a=a, v=v)

    # robot.urc_l.movej_pose(pose_horizontal_l, a=a, v=v)
    # robot.urc_l.movej_pose(pose_vertical_l, a=a, v=v)
    get_joints(robot)


def test_dual_arm_transition(robot):
    pose_horizontal_l = [-0.51, -0.11, 0.17, 1.22, 1.22, -1.22]
    pose_pregrasp_l = [-0.46, -0.23, 0.18, 0.0, 3.14, 0.0]

    pose_vertical_r = [-0.2, -0.3, 0.03, 2.22, 2.22, 0.00]
    pose_pregrasp_r = np.array([-0.16, -0.42, 0.12, 1.2, -1.2, 1.2])

    a = 3
    v = 0.5

    # a = 1
    # v = 0.1

    for i in range(10):
        robot.urc_l.movej_pose(pose_horizontal_l, a=a, v=v, wait=False)
        robot.urc_r.movej_pose(pose_vertical_r, a=a, v=v * 2, wait=False)

        robot.urc_l.wait_following()
        robot.urc_r.wait_following()

        robot.urc_l.movej_pose(pose_pregrasp_l, a=a, v=v, wait=False)
        robot.urc_r.movej_pose(pose_pregrasp_r, a=a, v=v * 2, wait=False)

        robot.urc_l.wait_following()
        robot.urc_r.wait_following()


def test_unfold(robot):
    pose_test_r = [-0.159, -0.53, 0.05, 1.2, -1.2, 1.189]
    robot.urc_r.movel_wait(pose_test_r)
    input()
    robot.grc.set_right(0.9)
    time.sleep(1)

    pose_corner_r = [-0.159, -0.384, 0.315, 1.22, -1.22, 1.22]
    robot.urc_r.movel_wait(pose_corner_r)


def move_together(robot, pose_l, pose_r, a=1, v=0.1):
    robot.urc_l.movel_nowait(pose_l, a=a, v=v)
    robot.urc_r.movel_nowait(pose_r, a=a, v=v)
    robot.urc_l.wait_following(len_p=5, threshold=5e-6)
    robot.urc_r.wait_following(len_p=5, threshold=5e-6)


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


def test_rotate(robot):
    joints_start_l = [0.267, -1.505, 1.332, -1.394, -1.585, 3.465]
    joints_end_l = [0.268, -1.507, 1.335, -1.395, -1.585, -2.921]

    a = 2
    v = 0.2

    while True:
        robot.urc_l.movej_joints(joints_start_l, a=a, v=v * 3)
        input()
        robot.urc_l.movej_joints(joints_end_l, a=a, v=v)
        input()


def main():
    robot = setup_robot()

    # test_thread(robot)
    get_poses(robot)
    # test_movel(robot)
    # home(robot)
    # test_transformation(robot)
    # test_orientation_transition(robot)
    # test_orientation_transition_l(robot)
    # test_dual_arm_transition(robot)
    # get_joints(robot)
    # test_unfold(robot)
    # test_fold(robot)
    # test_rotate(robot)

    del robot


if __name__ == "__main__":
    main()
