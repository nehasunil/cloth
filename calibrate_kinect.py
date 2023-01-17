#!/usr/bin/env python

import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from controller.ur5.ur_controller import UR_Controller
from mpl_toolkits.mplot3d import Axes3D
from perception.kinect.kinect_camera import Kinect
from scipy import optimize

world2camera = np.eye(4)

a = 0.3
v = 0.3


def calibrate_single_arm(
    urc,
    pose0,
    tool_orientation,
    workspace_limits,
    checkerboard_offset_from_tool,
    log_dir,
):

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir + "real", exist_ok=True)

    cam_intrinsics_origin = np.array(
        [
            [895.71514724, 0.0, 952.43775937],
            [0.0, 901.8924785, 608.72008223],
            [0.0, 0.0, 1.0],
        ]
    )
    cam_intrinsics = np.array(
        [
            [887.78753662, 0.0, 936.86175935],
            [0.0, 885.96539307, 634.48526542],
            [0.0, 0.0, 1.0],
        ]
    )  # New Camera Intrinsic Matrix
    dist = np.array([[0.08116987, -0.10769384, 0.01815891, -0.00474316, 0.03952777]])
    camera = Kinect(cam_intrinsics_origin, dist)

    urc.movel_wait(pose0, a=a, v=v)
    print(", ".join([str("{:.3f}".format(_)) for _ in urc.getl_rt()]))

    calib_grid_step = 0.05
    # calib_grid_step = 0.2

    # ---------------------------------------------

    # Construct 3D calibration grid across workspace
    gridspace_x = np.linspace(
        workspace_limits[0][0],
        workspace_limits[0][1],
        int(1 + (workspace_limits[0][1] - workspace_limits[0][0]) // calib_grid_step),
    )
    gridspace_y = np.linspace(
        workspace_limits[1][0],
        workspace_limits[1][1],
        int(1 + (workspace_limits[1][1] - workspace_limits[1][0]) // calib_grid_step),
    )
    gridspace_z = np.linspace(
        workspace_limits[2][0],
        workspace_limits[2][1],
        int(1 + (workspace_limits[2][1] - workspace_limits[2][0]) // calib_grid_step),
    )
    calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(
        gridspace_x, gridspace_y, gridspace_z
    )
    num_calib_grid_pts = (
        calib_grid_x.shape[0] * calib_grid_x.shape[1] * calib_grid_x.shape[2]
    )
    calib_grid_x.shape = (num_calib_grid_pts, 1)
    calib_grid_y.shape = (num_calib_grid_pts, 1)
    calib_grid_z.shape = (num_calib_grid_pts, 1)
    calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)

    measured_pts = []
    observed_pts = []
    observed_pix = []

    # Move robot to home pose
    print("Connecting to robot...")

    # Move robot to each calibration point in workspace
    print("Collecting data...")
    for calib_pt_idx in range(num_calib_grid_pts):
        tool_position = calib_grid_pts[calib_pt_idx, :]
        # robot.move_to(tool_position, tool_orientation)
        pose = np.hstack([tool_position, tool_orientation])
        urc.movel_wait(pose, a=a, v=v)

        time.sleep(1)
        #
        # Find checkerboard center
        checkerboard_size = (3, 3)
        refine_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        # camera_color_img, camera_depth_img = robot.get_camera_data()
        camera_color_img, camera_depth_img = camera.get_image()

        # bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
        bgr_color_data = camera_color_img
        gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_RGB2GRAY)
        checkerboard_found, corners = cv2.findChessboardCorners(
            gray_data, checkerboard_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH
        )
        if checkerboard_found:
            corners_refined = cv2.cornerSubPix(
                gray_data, corners, (3, 3), (-1, -1), refine_criteria
            )

            # Get observed checkerboard center 3D point in camera space
            checkerboard_pix = np.round(corners_refined[4, 0, :]).astype(int)
            checkerboard_z = camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]]
            checkerboard_x = np.multiply(
                checkerboard_pix[0] - cam_intrinsics[0][2],
                checkerboard_z / cam_intrinsics[0][0],
            )
            checkerboard_y = np.multiply(
                checkerboard_pix[1] - cam_intrinsics[1][2],
                checkerboard_z / cam_intrinsics[1][1],
            )
            if checkerboard_z == 0:
                continue

            # Save calibration point and observed checkerboard center
            observed_pts.append([checkerboard_x, checkerboard_y, checkerboard_z])
            tool_position = tool_position + checkerboard_offset_from_tool

            measured_pts.append(tool_position)
            observed_pix.append(checkerboard_pix)

            # Draw and display the corners
            # vis = cv2.drawChessboardCorners(robot.camera.color_data, checkerboard_size, corners_refined, checkerboard_found)
            vis = cv2.drawChessboardCorners(
                bgr_color_data, (1, 1), corners_refined[4, :, :], checkerboard_found
            )
            cv2.imwrite(log_dir + ("%06d.png" % len(measured_pts)), vis)
            cv2.imshow("Calibration", vis)
            cv2.waitKey(10)

    # Move robot back to home pose
    # robot.go_home()

    measured_pts = np.asarray(measured_pts)
    observed_pts = np.asarray(observed_pts)
    observed_pix = np.asarray(observed_pix)

    # Estimate rigid transform with SVD (from Nghia Ho)
    def get_rigid_transform(A, B):
        assert len(A) == len(B)
        N = A.shape[0]
        # Total points
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - np.tile(centroid_A, (N, 1))  # Centre the points
        BB = B - np.tile(centroid_B, (N, 1))
        H = np.dot(np.transpose(AA), BB)  # Dot is matrix multiplication for array
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:  # Special reflection case
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = np.dot(-R, centroid_A.T) + centroid_B.T
        return R, t

    def get_rigid_transform_error(z_scale):
        # global measured_pts, observed_pts, observed_pix, world2camera, camera, cam_intrinsics
        global world2camera

        # Apply z offset and compute new observed points using camera intrinsics
        observed_z = observed_pts[:, 2:] * z_scale
        observed_x = np.multiply(
            observed_pix[:, [0]] - cam_intrinsics[0][2],
            observed_z / cam_intrinsics[0][0],
        )
        observed_y = np.multiply(
            observed_pix[:, [1]] - cam_intrinsics[1][2],
            observed_z / cam_intrinsics[1][1],
        )
        new_observed_pts = np.concatenate((observed_x, observed_y, observed_z), axis=1)

        # Estimate rigid transform between measured points and new observed points
        R, t = get_rigid_transform(
            np.asarray(measured_pts), np.asarray(new_observed_pts)
        )
        t.shape = (3, 1)
        world2camera = np.concatenate(
            (np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0
        )

        # Compute rigid transform error
        registered_pts = np.dot(R, np.transpose(measured_pts)) + np.tile(
            t, (1, measured_pts.shape[0])
        )
        error = np.transpose(registered_pts) - new_observed_pts
        error = np.sum(np.multiply(error, error))
        rmse = np.sqrt(error / measured_pts.shape[0])
        return rmse

    # Optimize z scale w.r.t. rigid transform error
    print("Calibrating...")
    z_scale_init = 1
    optim_result = optimize.minimize(
        get_rigid_transform_error, np.asarray(z_scale_init), method="Nelder-Mead"
    )
    camera_depth_offset = optim_result.x

    # Save camera optimized offset and camera pose
    print("Saving...")
    np.savetxt(
        log_dir + "real/camera_depth_scale.txt", camera_depth_offset, delimiter=" "
    )
    get_rigid_transform_error(camera_depth_offset)
    camera_pose = np.linalg.inv(world2camera)
    np.savetxt(log_dir + "real/camera_pose.txt", camera_pose, delimiter=" ")
    print("Done.")


def calibrate_left_arm(urc_l, urc_r):
    ################ configurations for left arm ################
    pose0_l = np.array([-0.718, 0, 0.05, 0.26, -np.pi / 2, 0.0])
    tool_orientation_l = [0.0, -np.pi / 2, 0.0]  # [0,-2.22,2.22] # [2.22,2.22,0]
    checkerboard_offset_from_tool_l = [0.115, 0.00021, 0.0319]
    # from robot wrist center to checkerboard: 0.020 m
    # from gripper center to checkerboard: 0.0319 m
    # from robot wrist center to gripper center: -0.0119 m
    workspace_limits_l = np.asarray([[-0.845, -0.718], [-0.40, 0], [0.05, 0.26]])
    # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    log_dir_l = "logs/calibration_ur5_l/"

    # TODO: move right arm to a safe place
    pose_away_r = np.array([-0.42, -0.16, 0.05, -2.2, 2.2, 0.0])
    urc_r.movel_wait(pose_away_r, v=v, a=a)

    calibrate_single_arm(
        urc_l,
        pose0_l,
        tool_orientation_l,
        workspace_limits_l,
        checkerboard_offset_from_tool_l,
        log_dir_l,
    )


def calibrate_right_arm(urc_l, urc_r):
    ################ configurations for right arm ################
    pose0_r = np.array([-0.13, -0.4, 0.07, 0.0, np.pi / 2, 0.0])
    tool_orientation_r = [0.0, np.pi / 2, 0.0]
    # checkerboard_offset_from_tool_r = [0.115, 0.00021, 0.0319]
    checkerboard_offset_from_tool_r = [-0.094, 0.0, 0.01575]
    # from robot wrist center to checkerboard: 0.020 m
    # from gripper center to checkerboard: 0.0319 m
    # from robot wrist center to gripper center: -0.0119 m
    workspace_limits_r = np.asarray(
        [[-0.04, 0.23], [-0.62, -0.44], [-0.06, 0.15]]
    )  # TODO: measure exact number
    # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    log_dir_r = "logs/calibration_ur5_r/"

    # move left arm to a safe place
    pose_away_l = np.array([-0.39, 0.16, 0.07, 2.22, 2.22, 0.0])
    urc_l.movel_wait(pose_away_l, v=v, a=a)

    calibrate_single_arm(
        urc_r,
        pose0_r,
        tool_orientation_r,
        workspace_limits_r,
        checkerboard_offset_from_tool_r,
        log_dir_r,
    )


def setup_urc():
    urc_l = UR_Controller(HOST="10.42.0.121")
    urc_r = UR_Controller(HOST="10.42.0.2")
    urc_l.start()
    urc_r.start()
    return urc_l, urc_r


def kill_urc(urc_l, urc_r):
    urc_l.flag_terminate = True
    urc_l.join()
    urc_r.flag_terminate = True
    urc_r.join()


def calibrate_dual_arm():
    urc_l, urc_r = setup_urc()
    calibrate_left_arm(urc_l, urc_r)
    calibrate_right_arm(urc_l, urc_r)
    kill_urc(urc_l, urc_r)


if __name__ == "__main__":
    calibrate_dual_arm()
