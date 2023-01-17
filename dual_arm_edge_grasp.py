#!/usr/bin/env python

import glob
import os
import time
from csv import writer

import cv2
import numpy as np
import skimage
import torch
from classification_examples import Grip_Classifier
from logger import Logger, draw_grasp_point
from perception.infer_affordance.affordance import Affordance
from perception.infer_affordance.affordance_trainer import AffordanceTrainer
from perception.segmentation.run import Run
from perception.segmentation.segment import Segmentation
from policy import Policy_Affordance, Policy_Heuristic, get_upper_mask
from robot import Robot
from scipy.spatial.transform import Rotation as R
from test_follow import follow, test_fold
from util import get_highest_point, get_z_topview

flag_logging = True
flag_logging = False

robot = Robot()
segment = Segmentation()
afford = Affordance()
# fn_afford = "/home/gelsight/Code/Fabric/src/perception/infer_affordance/models/affordance_ft_0607"
# fn_afford = "/home/gelsight/Code/Fabric/src/perception/infer_affordance/models/nosim_0609/best_19_0.600.pt"
fn_afford = "/home/gelsight/Code/Fabric/src/perception/infer_affordance/models/finetune_0610/best_1_0.700.pt"
# fn_afford = "/home/gelsight/Code/Fabric/src/perception/infer_affordance/models/20220609_unetlowres_blackboxes.pth"
# fn_afford = "/home/gelsight/Code/Fabric/src/perception/infer_affordance/models/affordance_smooth_ft_0607"
# fn_afford = "/home/gelsight/Code/Fabric/src/perception/infer_affordance/models/ft_0607_noise/100_0.750.pt"
# fn_afford = "/home/gelsight/Code/Fabric/src/perception/infer_affordance/models/ft_0607_noise_0.5_smooth/260_0.700.pt"
afford.model.load_state_dict(torch.load(fn_afford, map_location=afford.device))
grip_classifier = Grip_Classifier(robot)
policy_affordance = Policy_Affordance()
policy_heuristic = Policy_Heuristic()
logger = Logger(logging_directory="data/affordance/20220602", iteration=-1)
affordance_trainer = AffordanceTrainer(logger, afford.model, afford.device, afford)
affordance_trainer.best_eval = 0.65

fn_afford_list = [
    "/home/gelsight/Code/Fabric/src/perception/infer_affordance/models/finetune_0610/best_1_0.700.pt",
    "segmentation",
    "",
    "/home/gelsight/Code/Fabric/src/perception/infer_affordance/models/nosim_0609/best_19_0.600.pt",
]
# model_names = ["no_sim", "sim_to_real", "finetune", "ft_time_averaged"]
# model_names = ["finetune_lowres", "sim_to_real", "no_sim"]
# threshold_list = [0.3, 0.3, 0.3]

model_names = ["finetune_lowres", "segmentation", "sim_to_real", "no_sim"]
threshold_list = [0.3, 0.3, 0.3, 0.3]


click_point_pix = ()
click_position_r = None

pose_horizontal_r = np.array([-0.16, -0.42, 0.12, 1.2, -1.2, 1.2])
pose_vertical_r = np.array([-0.2, -0.3, 0.03, 2.22, 2.22, 0.00])


crop_x_table = [700, 1100]
crop_y_table = [400, 800]
crop_x_seg = [150, 800]
crop_y_seg = [600, 1250]
crop_x_affordance = [100, 550]
crop_y_affordance = [700, 1150]


def move_away():
    # robot.urc_l.movel_wait(robot.pose_away_l)
    # robot.urc_r.movel_wait(pose_horizontal_r)
    robot.urc_r.movej_pose(pose_vertical_r)


def setup_robot():
    # move_away()

    tool_orientation = get_rotvec_vertical(180)
    pose_up = np.hstack([[-0.46, -0.24, 0.18], tool_orientation])

    acc = 0.2
    vel = 0.2
    # robot.urc_l.movej_pose(pose_up, a=acc, v=vel)
    return pose_up


def get_rotvec_vertical(theta):
    # Args: theta in degrees
    tool_orientation_euler = [180, 0, 180]
    tool_orientation_euler[2] = theta
    tool_orientation = R.from_euler(
        "xyz", tool_orientation_euler, degrees=True
    ).as_rotvec()
    return tool_orientation


def move_record(pose, a=3, v=0.3):
    robot.urc_l.movel_nowait(pose, a=a, v=v)
    robot.urc_l.clear_history()
    while True:
        if robot.urc_l.check_stopped():
            break
        time.sleep(0.1)


def dual_arm_movej_pose(joints_l, joints_r, a=3, v=0.3):
    robot.urc_l.movej_joints(joints_l, a=a, v=v * 2, wait=False)
    robot.urc_r.movej_joints(joints_r, a=a, v=v * 2, wait=False)
    robot.urc_l.wait_following()
    robot.urc_r.wait_following()


def pregrasp_right():
    # pose_horizontal_l = [-0.51, 0.05, 0.17, 1.22, 1.22, -1.22]
    joints_horizontal_l = [-0.728, -0.988, 1.494, -0.533, -0.721, -1.552]
    # pose_vertical_r = [-0.2, -0.3, 0.03, 2.22, 2.22, 0.00]
    joints_vertical_r = [0.672, -1.654, 2.417, -2.331, -1.575, 0.665]

    dual_arm_movej_pose(joints_horizontal_l, joints_vertical_r, a=3, v=0.3)


def pregrasp_left():
    # pose_vertical_l = [-0.46, -0.23, 0.33, 0.0, 3.14, 0.0]
    joints_vertical_l = [0.238, -1.493, 1.319, -1.392, -1.583, -1.326]
    # pose_horizontal_r = np.array([-0.16, -0.42, 0.12, 1.2, -1.2, 1.2])
    joints_horizontal_r = [0.48, -1.608, 2.486, -0.888, 0.482, 1.568]

    pose_horizontal_safe_r = robot.urc_r.getl_rt()
    pose_horizontal_safe_r[1] = -0.42
    robot.urc_r.movel_wait(pose_horizontal_safe_r)

    pose_horizontal_safe_l = robot.urc_l.getl_rt()
    pose_horizontal_safe_l[2] = 0.33
    robot.urc_l.movel_wait(pose_horizontal_safe_l)

    dual_arm_movej_pose(joints_vertical_l, joints_horizontal_r, a=3, v=0.3)


def get_xyz_robot(camera_depth_img, click_point_pix):
    x, y = click_point_pix[0], click_point_pix[1]
    d = camera_depth_img[y][x]
    target_position = robot.transform_r.get_robot_from_depth(x, y, d)
    return target_position


def get_xyz_robot_l(camera_depth_img, click_point_pix):
    x, y = click_point_pix[0], click_point_pix[1]
    d = camera_depth_img[y][x]
    target_position = robot.transform_l.get_robot_from_depth(x, y, d)
    return target_position


def move_up_r():
    pose_up_r = [-0.11, -0.67, 0.39, 2.21, 2.22, 0.0]
    robot.urc_r.movel_wait(pose_up_r, a=1, v=0.1)
    time.sleep(0.5)


def mouseclick_callback(event, x, y, flags, param):
    # Callback function for clicking on OpenCV window
    if event == cv2.EVENT_LBUTTONDOWN:
        global robot, click_point_pix, click_position_r
        camera_color_img, camera_depth_img = robot.camera.get_image()

        click_point_pix = (x, y)
        target_position = get_xyz_robot(camera_depth_img, click_point_pix)

        print("TARGET POS: ", target_position)
        print("CLICK POINT: ", click_point_pix)

        click_position_r = target_position.copy()
        # urc.movel_wait(pose, a=a, v=v)


def setup_click_grasp():
    cv2.namedWindow("color")
    cv2.setMouseCallback("color", mouseclick_callback)
    cv2.namedWindow("depth")


def close_gripper_l():
    robot.grc.set_left(0.7)
    robot.grc.follow_dc_pos = [-5, 5]
    time.sleep(0.5)


def open_gripper_l():
    robot.grc.set_left(0.0)
    time.sleep(0.5)


def open_gripper_wide_l():
    robot.grc.set_left(0.0)
    robot.grc.follow_dc_pos = [10, -10]
    # robot.grc.follow_dc_pos = [0, 0]
    time.sleep(0.5)


def open_gripper_r():
    robot.grc.set_right(0.0)
    time.sleep(0.5)


def close_gripper_r():
    robot.grc.set_right(0.9)
    time.sleep(0.5)


def home_r():
    robot.urc_r.movel_wait(pose_horizontal_r)


def visualize_crop(camera_color_img, crop_x, crop_y):
    color_crop = camera_color_img[crop_y[0] : crop_y[1], crop_x[0] : crop_x[1]]
    cv2.imshow("color_crop", color_crop)
    cv2.waitKey(1)


def get_highest_robot(
    camera_color_img, camera_depth_img, visualize=False, mask_corners=None
):
    crop_x_table = [700, 1100]
    crop_y_table = [400, 800]

    if visualize:
        visualize_crop(camera_color_img, crop_x_table, crop_y_table)

    xyz_robot = robot.transform_l.get_robot_from_depth_array(
        camera_depth_img, crop_x_table, crop_y_table
    )

    xlim = [-0.65, -0.30]
    ylim = [-0.40, -0.05]
    scale = 100
    z_ws = get_z_topview(xyz_robot, xlim, ylim, scale)

    mask_corners_topview = None
    if mask_corners is not None:
        xym = xyz_robot.copy()
        mask_corners_transformed = np.fliplr(np.rot90(mask_corners, k=3))
        xym[2] = mask_corners_transformed.reshape((-1))
        mask_corners_topview = get_z_topview(xym, xlim, ylim, scale)

    xyz_robot_highest = get_highest_point(
        z_ws, xlim, ylim, scale, visualize, mask_corners=mask_corners_topview
    )

    return xyz_robot_highest


def get_corner_robot(camera_color_img, camera_depth_img, segmentation):
    target_position = get_xyz_robot_l(camera_depth_img, click_point_pix)


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


def pick_highest_r(camera_color_img, camera_depth_img, pose_up, visualize=True):
    tool_orientation = pose_up[3:]
    xyz_robot_highest = get_highest_robot(camera_color_img, camera_depth_img, visualize)

    L_p = np.array(xyz_robot_highest[:3])
    R_p = robot.transform_l2r(L_p)
    xyz_robot_highest[:3] = R_p

    xyz_robot_highest[2] -= 0.02
    if xyz_robot_highest[2] < -0.098:
        print("WARNNING: Reach Z Limit, set to minimal Z")
        xyz_robot_highest[2] = -0.098
    pose = np.hstack([xyz_robot_highest, tool_orientation])
    print("move to highest point", pose)

    pose_higher = pose.copy()
    pose_higher[2] = 0.02
    robot.urc_r.movel_wait(pose_higher)
    robot.urc_r.movel_wait(pose)


def pick_corners(
    camera_color_img, camera_depth_img, pose_up, mask_corners, visualize=True
):
    mask_corner_tableview = get_cropped_images(
        camera_depth_img,
        mask_corners,
        crop_x_seg,
        crop_y_seg,
        crop_y_table,
        crop_x_table,
    )

    tool_orientation = pose_up[3:]
    xyz_robot_highest = get_highest_robot(
        camera_color_img,
        camera_depth_img,
        visualize,
        mask_corners=mask_corner_tableview,
    )
    xyz_robot_highest[2] -= 0.02
    if xyz_robot_highest[2] < -0.098:
        print("WARNNING: Reach Z Limit, set to minimal Z")
        xyz_robot_highest[2] = -0.098
    pose = np.hstack([xyz_robot_highest, tool_orientation])
    print("move to highest point", pose)

    move_record(pose)


def home(pose_up):
    theta = np.random.randint(180) + 90
    tool_orientation = get_rotvec_vertical(theta)

    random_perturb_rng = 0.05
    pose_up = np.hstack([[-0.46, -0.24, 0.3], tool_orientation])
    pose_up[0] += np.random.uniform(-0.5, 0.5) * random_perturb_rng
    pose_up[1] += np.random.uniform(-0.5, 0.5) * random_perturb_rng

    move_record(pose_up)
    return pose_up


def rotate(theta):
    # theta = np.random.randint(180) + 90
    pose = robot.urc_l.getl_rt()
    tool_orientation = get_rotvec_vertical(theta)
    pose[3:] = tool_orientation
    move_record(pose, a=1, v=0.1)
    return pose


# def rotate_init():
#     # theta = np.random.randint(180) + 90
#     pose = robot.urc_l.getl_rt()
#     tool_orientation = get_rotvec_vertical(theta)
#     pose[3:] = tool_orientation
#     move_record(pose, a=1, v=0.1)
#     return pose


def get_tool_orientation_side_r():
    # horizontal from right
    euler_angle_yxz = [-90, 0, 90]
    tool_orientation = R.from_euler("yxz", euler_angle_yxz, degrees=True).as_rotvec()
    return tool_orientation


def side_grasp(pose_r):
    # adjust for centered grasp
    pose_r[1] += 0.01
    # pose_r[1] -= 0.01
    pose_r_pregrasp = pose_r.copy()
    pose_r_pregrasp[1] = robot.urc_r.getl_rt()[1]
    robot.urc_r.movel_wait(pose_r_pregrasp)
    robot.urc_r.movel_wait(pose_r)


def side_grasp_adjust():
    pose_r = robot.urc_r.getl_rt()
    # adjust for centered grasp
    robot.grc.set_right(0.4)
    time.sleep(0.2)
    pose_r[1] -= 0.02
    robot.urc_r.movel_wait(pose_r)
    robot.grc.set_right(0.88)
    time.sleep(0.2)


def side_grasp_l(pose_l):
    # adjust for centered grasp
    # pose_r[1] += 0.01
    pose_l_pregrasp = pose_l.copy()
    pose_l_pregrasp[1] = robot.urc_l.getl_rt()[1]
    robot.urc_l.movel_wait(pose_l_pregrasp)
    robot.urc_l.movel_wait(pose_l)


def check_safety_side_grasp(pose_r):
    SAFE_DY = 0.06
    SAFE_DZ = -0.03
    position_l_in_r = robot.get_l_in_r()
    if pose_r[2] > position_l_in_r[2] + SAFE_DZ:
        print(
            f"WARNING: target position {pose_r[:3]} may collide with the left arm at {position_l_in_r[1]}"
        )
        return False
    if (
        pose_r[0] < -0.3
        or pose_r[0] > 0
        or pose_r[1] < -0.7
        or pose_r[1] > -0.45
        or pose_r[2] < -0.05
        or pose_r[2] > 0.26
    ):
        print(
            f"WARNING: target position out of safety boundary, target pose {pose_r[:3]}"
        )
        return False
    print(f"move to target pose {pose_r[:3]}")
    side_grasp(pose_r)
    return True


def grasp_r(click_position_r):
    tool_orientation = get_tool_orientation_side_r()
    pose_r = np.hstack([click_position_r, tool_orientation])
    print("click_position_r", click_position_r)
    # move_record(pose_r)
    result = check_safety_side_grasp(pose_r)
    return result


def grasp_corner_l(camera_depth_img):
    xyz_lowest = get_lowest_point(
        camera_depth_img, crop_x_affordance, crop_y_affordance
    )
    pose = robot.urc_l.getl_rt()
    print("current_pose", pose)
    pose[:3] = xyz_lowest
    # pose[1] -= 0.01
    pose[2] += 0.02
    print("lowest_point", pose)
    side_grasp_l(pose)


def visualize_prediction(y_pred, depth, win_name="affordance"):
    y_pred_vis = y_pred  # / 0.5
    print(y_pred.min(), y_pred.max())
    y_pred_vis = np.clip(y_pred_vis * 2, 0, 1)
    y_pred_vis = (y_pred_vis * 255).astype(np.uint8)
    im_color = cv2.applyColorMap(y_pred_vis, cv2.COLORMAP_JET)
    combined = im_color.copy() * 1.0
    for c in range(3):
        combined[:, :, c] = (depth / 2) + (im_color[:, :, c] / 255.0 / 2)
    cv2.imshow(win_name, combined)
    cv2.waitKey(1)


def normalize_depth(depth, vmin, vmax):
    depth = np.clip(depth, vmin, vmax)
    depth_normalized = (depth - vmin) / (vmax - vmin) - 0.5
    return depth_normalized


def get_cloth_mask(camera_depth_img, crop_x, crop_y):
    depth_crop = camera_depth_img[crop_x[0] : crop_x[1], crop_y[0] : crop_y[1]]

    xyz_robot = robot.transform_l.get_robot_from_depth_array(
        camera_depth_img, crop_y, crop_x
    )
    print("xyz_robot.shape", xyz_robot.shape)
    # xyz_robot: (3, W*H)
    cloth_mask = (
        (xyz_robot[2] > -0.1)
        & (xyz_robot[2] < 0.35)
        & (xyz_robot[0] < -0.3)
        & (xyz_robot[0] > -0.6)
        & (xyz_robot[1] < -0.1)
        & (xyz_robot[1] > -0.35)
    )
    cloth_mask = cloth_mask.reshape((depth_crop.shape[0], depth_crop.shape[1])).T

    return cloth_mask


def filter_cloth_mask(mask, depth_crop):
    mask = mask.reshape((depth_crop.shape[0], depth_crop.shape[1]))
    # cv2.imshow("filtered_mask", mask * 1.0)
    mask = mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = mask.reshape((-1))
    mask = mask.astype(bool)
    return mask


def get_lowest_point(camera_depth_img, crop_x, crop_y):
    depth_crop = camera_depth_img[crop_x[0] : crop_x[1], crop_y[0] : crop_y[1]]

    xyz_robot = robot.transform_l.get_robot_from_depth_array(
        camera_depth_img, crop_y, crop_x
    )
    # xyz_robot: (3, W*H)
    cloth_mask = (
        (xyz_robot[2] > -0.05)
        & (xyz_robot[2] < 0.35)
        & (xyz_robot[0] < -0.3)
        & (xyz_robot[0] > -0.6)
        & (xyz_robot[1] < -0.1)
        & (xyz_robot[1] > -0.35)
    )
    cloth_mask = filter_cloth_mask(cloth_mask, depth_crop)

    xyz_robot_cloth = xyz_robot[:, cloth_mask]
    print(
        "xyz_robot_cloth.shape",
        xyz_robot_cloth.shape,
        np.sum(cloth_mask),
        xyz_robot.shape,
    )

    print(xyz_robot_cloth, len(xyz_robot_cloth[2]))
    if len(xyz_robot_cloth[2]) == 0:
        raise Exception("no fabric in hand")

    min_idx = np.argmin(xyz_robot_cloth[2])
    xyz = xyz_robot_cloth[:, min_idx]

    cloth_mask = cloth_mask.reshape((depth_crop.shape[0], depth_crop.shape[1])).T
    # cv2.imshow("cloth_mask", cloth_mask * 1.0)
    cv2.waitKey(1)
    return xyz


def get_safety_mask(camera_depth_img, crop_x, crop_y):
    depth_crop = camera_depth_img[crop_x[0] : crop_x[1], crop_y[0] : crop_y[1]]

    xyz_robot = robot.transform_r.get_robot_from_depth_array(
        camera_depth_img, crop_y, crop_x
    )
    print("xyz_robot.shape", xyz_robot.shape)
    # xyz_robot: (3, W*H)

    SAFE_DY = 0.06
    SAFE_DZ = -0.03
    position_l_in_r = robot.get_l_in_r()
    print("position_l_in_r", position_l_in_r)

    mask_no_collision = xyz_robot[2] < position_l_in_r[2] + SAFE_DZ
    mask_in_boundry = (
        (xyz_robot[0] > -0.3)
        & (xyz_robot[0] < 0)
        & (xyz_robot[1] > -0.7)
        & (xyz_robot[1] < -0.45)
        & (xyz_robot[2] > -0.05)
        & (xyz_robot[2] < 0.26)
    )
    safety_mask = mask_no_collision & mask_in_boundry
    safety_mask = filter_cloth_mask(safety_mask, depth_crop)
    safety_mask = safety_mask.reshape((depth_crop.shape[0], depth_crop.shape[1])).T

    return safety_mask


def visualize_affordance(camera_color_img, camera_depth_img, visualize=True):
    color_crop = camera_color_img[
        crop_x_affordance[0] : crop_x_affordance[1],
        crop_y_affordance[0] : crop_y_affordance[1],
    ]
    depth_crop = camera_depth_img[
        crop_x_affordance[0] : crop_x_affordance[1],
        crop_y_affordance[0] : crop_y_affordance[1],
    ]
    cloth_mask = get_cloth_mask(camera_depth_img, crop_x_affordance, crop_y_affordance)
    cloth_mask = filter_cloth_mask(cloth_mask, depth_crop)
    cloth_mask = cloth_mask.reshape(depth_crop.shape[0], depth_crop.shape[1])
    # depth_crop[~cloth_mask] = 0

    color_resize = cv2.resize(color_crop, (224, 224))
    depth_resize = cv2.resize(depth_crop, (224, 224))
    mask_resize = cv2.resize(cloth_mask.astype(np.uint8), (224, 224))

    # depth_normalized = normalize_depth(depth_resize, 500, 1200)
    depth_normalized = normalize_depth(depth_resize, 0, 700)

    depth_masked = depth_normalized
    depth_masked[mask_resize == 0] = 0
    affordance = afford.get_affordance(depth_masked)
    affordance[mask_resize == 0] = 0

    print("affordance.max()", affordance.max())

    logger.update_camera(camera_color_img, camera_depth_img, depth_masked)
    safety_mask = get_safety_mask(
        camera_depth_img, crop_x_affordance, crop_y_affordance
    )
    safety_mask_resize = cv2.resize(safety_mask.astype(np.uint8), (224, 224))
    visualize_affordance_masked(affordance, depth_normalized, safety_mask_resize)

    if visualize:
        # cv2.imshow("cloth_mask", cloth_mask * 1.0)
        cv2.imshow("color_crop", color_resize)
        cv2.imshow("depth_crop", depth_normalized + 0.5)
        # cv2.imshow("depth_mask", depth_normalized + 0.5)
        # cv2.imshow("depth_crop", depth_crop / 1100.0)
        visualize_prediction(affordance, depth_normalized)
        cv2.waitKey(1)

    return affordance


def visualize_affordances(camera_color_img, camera_depth_img, visualize=True):
    global afford
    for model_id, fn_afford in enumerate(fn_afford_list):
        if fn_afford != "":
            afford.model.load_state_dict(
                torch.load(fn_afford, map_location=afford.device)
            )
        else:
            afford = Affordance()

        color_crop = camera_color_img[
            crop_x_affordance[0] : crop_x_affordance[1],
            crop_y_affordance[0] : crop_y_affordance[1],
        ]
        depth_crop = camera_depth_img[
            crop_x_affordance[0] : crop_x_affordance[1],
            crop_y_affordance[0] : crop_y_affordance[1],
        ]
        cloth_mask = get_cloth_mask(
            camera_depth_img, crop_x_affordance, crop_y_affordance
        )
        cloth_mask = filter_cloth_mask(cloth_mask, depth_crop)
        cloth_mask = cloth_mask.reshape(depth_crop.shape[0], depth_crop.shape[1])

        color_resize = cv2.resize(color_crop, (224, 224))
        depth_resize = cv2.resize(depth_crop, (224, 224))
        mask_resize = cv2.resize(cloth_mask.astype(np.uint8), (224, 224))

        # depth_normalized = normalize_depth(depth_resize, 500, 1200)
        depth_normalized = normalize_depth(depth_resize, 0, 700)

        depth_masked = depth_normalized
        depth_masked[mask_resize == 0] = 0
        affordance = afford.get_affordance(depth_masked)
        affordance[mask_resize == 0] = 0

        print("affordance.max()", affordance.max())

        logger.update_camera(camera_color_img, camera_depth_img, depth_masked)
        safety_mask = get_safety_mask(
            camera_depth_img, crop_x_affordance, crop_y_affordance
        )
        safety_mask_resize = cv2.resize(safety_mask.astype(np.uint8), (224, 224))
        # visualize_affordance_masked(affordance, depth_normalized, safety_mask_resize)

        if visualize:
            # cv2.imshow("cloth_mask", cloth_mask * 1.0)
            cv2.imshow("color_crop", color_resize)
            cv2.imshow("depth_crop", depth_normalized + 0.5)
            # cv2.imshow("depth_mask", depth_normalized + 0.5)
            # cv2.imshow("depth_crop", depth_crop / 1100.0)
            if model_id == 2:
                visualize_affordance_masked(
                    affordance, depth_normalized, safety_mask_resize
                )
            # else:
            visualize_prediction(
                affordance, depth_normalized, win_name=model_names[model_id]
            )
            cv2.waitKey(1)


def get_cropped_images(
    img, img_src_local, crop_x_src, crop_y_src, crop_x_target, crop_y_target
):
    img_tmp = np.zeros(img.shape, dtype=img_src_local.dtype)
    img_tmp[
        crop_x_src[0] : crop_x_src[1], crop_y_src[0] : crop_y_src[1]
    ] = img_src_local

    img_target = img_tmp[
        crop_x_target[0] : crop_x_target[1], crop_y_target[0] : crop_y_target[1]
    ]
    return img_target


def visualize_segmentation(camera_color_img, camera_depth_img):
    # crop_x_seg = [150, 800]
    # crop_y_seg = [600, 1250]

    # cv2.imwrite("logs/tmp_color_2.jpg", camera_color_img)
    # np.save("logs/tmp_depth_2.npy", camera_depth_img)

    (
        segmentation,
        mask_corners,
        depth_transformed_small,
        seg_pred,
    ) = segment.get_segmentation(
        camera_color_img, camera_depth_img, crop_x_seg, crop_y_seg
    )
    return segmentation, mask_corners, depth_transformed_small, seg_pred


def grasp_heuristic(camera_color_img, camera_depth_img, time_average=1, select_idx=[]):
    heuristic = segment.get_heuristic(
        camera_color_img, camera_depth_img, robot.transform_l
    )
    print("heuristic.shape", heuristic.shape)
    # cv2.imshow("heuristic", heuristic)
    # cv2.waitKey(1)
    visualize_prediction(heuristic / 2, 0)

    safety_mask = get_safety_mask(camera_depth_img, crop_x_seg, crop_y_seg)

    safety_mask_resize = cv2.resize(
        safety_mask.astype(np.uint8), (heuristic.shape[0], heuristic.shape[1])
    )
    # cv2.imshow("safety_mask", safety_mask_resize * 255)
    # visualize_affordance_masked(affordance, safety_mask_resize)

    for nb_try in range(1):

        idx = policy_heuristic.grasp_policy(
            heuristic, safety_mask_resize, select_idx=select_idx
        )

        select_idx.append(idx)

        x_local, y_local = idx
        x, y = get_global_pixel_seg(x_local, y_local, crop_x_seg, crop_y_seg)

        click_point_pix = (y, x)
        target_position = get_xyz_robot(camera_depth_img, click_point_pix)
        print("target_position side_grasp", target_position)
        cv2.waitKey(1)

        # input()
        result = grasp_r(target_position)
        if result:
            return True
    return False


def get_local_pixel(x_global, y_global, crop_x, crop_y):
    x = int((x_global - crop_x[0]) / (crop_x[1] - crop_x[0]) * 224)
    y = int((y_global - crop_y[0]) / (crop_y[1] - crop_y[0]) * 224)
    return x, y


def get_global_pixel(x_local, y_local, crop_x, crop_y):
    x = int(x_local / 224.0 * (crop_x[1] - crop_x[0]) + crop_x[0])
    y = int(y_local / 224.0 * (crop_y[1] - crop_y[0]) + crop_y[0])
    return x, y


def get_global_pixel_seg(x_local, y_local, crop_x, crop_y):
    x = int(x_local + crop_x[0])
    y = int(y_local + crop_y[0])
    return x, y


def click_grasp_r(camera_color_img, camera_depth_img):
    global click_position_r, click_point_pix

    if click_position_r is not None:
        affordance = visualize_affordance(camera_color_img, camera_depth_img)

        y_global, x_global = click_point_pix
        x_local, y_local = get_local_pixel(
            x_global, y_global, crop_x_affordance, crop_y_affordance
        )
        logger.update_selected_pixel(x_local, y_local)

        result = grasp_r(click_position_r)
        click_position_r = None


def visualize_affordance_masked(affordance, depth_normalized, safety_mask):
    affordance_masked = affordance.copy()
    upper_mask = get_upper_mask(safety_mask)
    affordance_masked[upper_mask == 0] = 0
    cv2.imshow("upper_mask", upper_mask * 1.0)
    visualize_prediction(
        affordance_masked, depth_normalized, win_name="affordance_masked"
    )


def grasp_affordance(
    camera_color_img, camera_depth_img, threshold=0.4, time_average=1, select_idx=[]
):
    affordance = visualize_affordance(camera_color_img, camera_depth_img)
    if time_average > 1:
        for i in range(time_average - 1):
            camera_color_img, camera_depth_img = robot.camera.get_image()
            affordance += visualize_affordance(camera_color_img, camera_depth_img)
        affordance /= time_average
        cv2.imshow("average_affordance", affordance)
        cv2.waitKey(1)

    safety_mask = get_safety_mask(
        camera_depth_img, crop_x_affordance, crop_y_affordance
    )

    safety_mask_resize = cv2.resize(safety_mask.astype(np.uint8), (224, 224))
    # cv2.imshow("safety_mask", safety_mask_resize * 255)
    # visualize_affordance_masked(affordance, safety_mask_resize)

    for nb_try in range(1):

        idx = policy_affordance.grasp_policy(
            affordance, safety_mask_resize, threshold=threshold, select_idx=select_idx
        )

        select_idx.append(idx)

        x_local, y_local = idx
        logger.update_selected_pixel(x_local, y_local)

        x = logger.grasp_item.x
        best_pix_ind = logger.grasp_item.best_pix_ind
        image_grasp = draw_grasp_point(x, best_pix_ind)
        cv2.imshow("img_grasp", image_grasp)

        x, y = get_global_pixel(x_local, y_local, crop_x_affordance, crop_y_affordance)
        click_point_pix = (y, x)
        target_position = get_xyz_robot(camera_depth_img, click_point_pix)
        print("target_position side_grasp", target_position)
        cv2.waitKey(1)

        # input()
        result = grasp_r(target_position)
        if result:
            return True
    return False
    #
    # print("side grasp result", result)


def move_pre_sidegrasp_r():
    pose_horizontal_r = np.array([-0.16, -0.42, 0.12, 1.2, -1.2, 1.2])
    robot.urc_r.movel_wait(pose_horizontal_r)


def grasp_affordance_rotation():
    theta = 90
    for i in range(24):
        select_idx = []
        for j in range(3):
            # for j in range(1):
            open_gripper_r()
            move_pre_sidegrasp_r()

            rotate(theta)
            camera_color_img, camera_depth_img = robot.camera.get_image()

            try:
                result = grasp_affordance(
                    camera_color_img, camera_depth_img, select_idx=select_idx
                )
                if result:
                    grip_classify()
                    # c = input()
                    c = -1
                    # logger.save_log(human_label=c)
                    time.sleep(0.5)
                else:
                    break
            except Exception as e:
                print(e)
                args = e.args
                if args[0] == "no fabric in hand":
                    raise Exception("no fabric in hand")
                elif args[0] == "All affordance below threshold":
                    break
        theta = theta + 15
        # train_affordance()
    open_gripper_r()
    move_pre_sidegrasp_r()
    rotate(theta - 180)


def save_eval(model_id, attempts, c, rotation):
    data = [attempts, c, rotation]
    # model_names = ["no_sim", "sim_to_real", "finetune"]
    model_name = model_names[model_id]
    fn = f"data/evaluation/eval_{model_name}.csv"
    with open(fn, "a", newline="") as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(data)
        f_object.close()


def grasp_affordance_evaluation():
    global afford

    for model_id, fn_afford in enumerate(fn_afford_list):
        flag_grasped = False
        if fn_afford == "":
            afford = Affordance()
        elif fn_afford != "segmentation":
            afford.model.load_state_dict(
                torch.load(fn_afford, map_location=afford.device)
            )
        attempts = 0
        theta = 90
        for i in range(24):
            select_idx = []
            # for j in range(1):
            open_gripper_r()
            move_pre_sidegrasp_r()

            rotate(theta)
            camera_color_img, camera_depth_img = robot.camera.get_image()

            threshold = threshold_list[model_id]
            try:
                if fn_afford == "segmentation":
                    result = grasp_heuristic(camera_color_img, camera_depth_img)
                else:
                    result = grasp_affordance(
                        camera_color_img,
                        camera_depth_img,
                        threshold=threshold,
                        select_idx=select_idx,
                    )

                if result:
                    class_ind = grip_classify()
                    attempts += 1
                    if class_ind == 1:
                        c = input()
                        rotation = theta - 90
                        flag_grasped = True
                        save_eval(model_id, attempts, c, rotation)
                        # logger.save_log(human_label=c)
                        break
                    # time.sleep(0.5)
            except Exception as e:
                print(e)
                args = e.args
                if args[0] == "no fabric in hand":
                    raise Exception("no fabric in hand")
                elif args[0] == "All affordance below threshold":
                    pass
            theta = theta + 15
            # train_affordance()
        if flag_grasped is False:
            c = 0
            rotation = theta - 90
            save_eval(model_id, attempts, c, rotation)
        open_gripper_r()
        move_pre_sidegrasp_r()
        rotate(np.clip(theta - 180, 90, 270))
    input()


def grasp_slide():
    global afford

    # c = input()
    # if c == "c":
    #     robot.grc.set_left(0.1)
    #     time.sleep(0.2)
    #     input()
    #     robot.grc.set_left(0.6)
    #     input()
    # robot.grc.set_left(0.6)

    attempts = 0
    theta = 90
    for i in range(24):
        select_idx = []
        # for j in range(1):
        open_gripper_r()
        move_pre_sidegrasp_r()

        rotate(theta)
        camera_color_img, camera_depth_img = robot.camera.get_image()

        threshold = 0.35
        try:
            if fn_afford == "segmentation":
                result = grasp_heuristic(camera_color_img, camera_depth_img)
            else:
                result = grasp_affordance(
                    camera_color_img,
                    camera_depth_img,
                    threshold=threshold,
                    select_idx=select_idx,
                )

            if result:
                class_ind = grip_classify()
                attempts += 1
                if class_ind == 1:
                    # side_grasp_adjust()
                    follow(robot)
                    robot.grc.set_right(0.94)
                    test_fold(robot)
                    break
                # time.sleep(0.5)
        except Exception as e:
            print(e)
            args = e.args
            if args[0] == "no fabric in hand":
                raise Exception("no fabric in hand")
            elif args[0] == "All affordance below threshold":
                pass
        theta = theta + 15
        # train_affordance()

    open_gripper_r()
    move_pre_sidegrasp_r()
    rotate(np.clip(theta - 180, 90, 270))

    input()


def grip_classify():

    grip_classifier.set_gripper_r(1200)
    time.sleep(1)

    frames, frame0, frames2, frame0_2 = grip_classifier.grip_record(1000)
    frame0_raw = grip_classifier.frame0_raw

    frame0 = cv2.GaussianBlur(frame0_raw, (31, 31), 0)
    frame0 = cv2.resize(frame0, (200, 150))
    logger.update_tactile(frames, frame0, frames2, frame0_2)

    class_ind, class_probs = grip_classifier.classify(frames, frame0)
    print("classification:", grip_classifier.labels_origin[class_ind])
    logger.update_y_gt(class_ind, class_probs)
    return class_ind


def train_affordance():
    affordance_trainer.train(learning_rate=1e-3)


def main():

    pose_up = setup_robot()
    setup_click_grasp()

    seq_id = 0

    # sequence = [
    #     "open_gripper_l",
    #     "open_gripper_r",
    #     "pregrasp_right",
    #     "pick_highest_r",
    #     "close_gripper_r",
    #     "move_up_r",
    #     "open_gripper_wide_l",
    #     "grasp_corner_l",
    #     "close_gripper_l",
    #     "open_gripper_r",
    #     "pregrasp_left",
    #     "grasp_slide",
    #     # "grasp_affordance_rotation",  # "click_grasp_r",
    #     # "grip_classify",
    #     # "train",
    #     "open_gripper_r",
    #     "home_r",
    #     "open_gripper_l",
    # ]

    # sequence = ["grasp_affordance_rotation"]

    # sequence = ["grasp_affordance_evaluation"]
    # sequence = ["pregrasp_left", "grasp_slide"]
    # sequence = ["grip_classify"]
    # sequence = ["grasp_slide"]
    # sequence = ["grasp_heuristic"]
    sequence = ["pregrasp_left"]

    # sequence = ["visualize_segmentation"]

    # sequence = ["visualize_affordance"]
    # sequence = ["visualize_affordances"]

    # sequence = [
    #     # "pregrasp_left",
    #     # "close_gripper_l",
    #     "open_gripper_r",
    #     "home_r",
    #     "click_grasp_r",
    #     "grip_classify",
    # ]

    while True:
        if seq_id % len(sequence) == 0:
            theta = 180
        #     time.sleep(0.5)
        camera_color_img, camera_depth_img = robot.camera.get_image()

        if len(click_point_pix) != 0:
            camera_color_img = cv2.circle(
                camera_color_img, click_point_pix, 7, (0, 0, 255), 2
            )

        cv2.imshow("color", camera_color_img)
        # cv2.imshow("depth", camera_depth_img / 1100.0)
        cv2.waitKey(1)

        c = sequence[seq_id % len(sequence)]

        if c == "close_gripper_l":
            close_gripper_l()
        if c == "open_gripper_l":
            open_gripper_l()
        elif c == "home":
            pose_up = home(pose_up)
        elif c == "pick_highest":
            time.sleep(0.5)
            pick_highest(camera_color_img, camera_depth_img, pose_up)
        elif c == "pick_corners":
            (
                segmentation,
                mask_corners,
                depth_transformed_small,
                seg_pred,
            ) = visualize_segmentation(camera_color_img, camera_depth_img)
            pick_corners(camera_color_img, camera_depth_img, pose_up, mask_corners)
            # pick_boundary(camera_color_img, camera_depth_img, pose_up, segmentation)
        elif c == "rotate":
            pose_up = rotate(theta)
            theta = theta + 15
        elif c == "click_grasp_r":
            if click_position_r is not None:
                click_grasp_r(camera_color_img, camera_depth_img)
            else:
                seq_id -= 1
        elif c == "visualize_affordance":
            # visualize_affordance(camera_color_img, camera_depth_img)
            n_window = 1
            camera_color_img, camera_depth_img_sum = robot.camera.get_image()
            for i in range(n_window - 1):
                camera_color_img, camera_depth_img = robot.camera.get_image()
                camera_depth_img_sum += camera_depth_img
            camera_depth_img_avg = camera_depth_img_sum / n_window
            visualize_affordance(camera_color_img, camera_depth_img_avg)
        elif c == "visualize_affordances":
            camera_color_img, camera_depth_img_sum = robot.camera.get_image()
            visualize_affordances(camera_color_img, camera_depth_img)
        elif c == "visualize_segmentation":
            visualize_segmentation(camera_color_img, camera_depth_img)
        elif c == "grasp_affordance":
            try:
                result = grasp_affordance(camera_color_img, camera_depth_img)
                if result is False:
                    seq_id = -1
            except Exception as e:
                args = e.args
                open_gripper_r()
                seq_id = -1
        elif c == "grip_classify":
            grip_classify()
            # c = input()
            c = -1
            logger.save_log(human_label=c)
            time.sleep(0.5)
        elif c == "open_gripper_r":
            open_gripper_r()
        elif c == "close_gripper_r":
            close_gripper_r()
        elif c == "home_r":
            home_r()
        elif c == "pregrasp_right":
            pregrasp_right()
        elif c == "pregrasp_left":
            pregrasp_left()
        elif c == "pick_highest_r":
            pick_highest_r(camera_color_img, camera_depth_img, pose_vertical_r)
        elif c == "move_up_r":
            move_up_r()
        elif c == "grasp_corner_l":
            try:
                grasp_corner_l(camera_depth_img)
            except Exception as e:
                args = e.args
                open_gripper_r()
                seq_id = sequence.index("pick_highest_r") - 1
        elif c == "open_gripper_wide_l":
            open_gripper_wide_l()
        elif c == "grasp_affordance_evaluation":
            grasp_affordance_evaluation()
        elif c == "grasp_affordance_rotation":
            # grasp_affordance_rotation()
            try:
                grasp_affordance_rotation()
            except Exception as e:
                print(e)
                args = e.args
                open_gripper_r()
                seq_id = -1
        elif c == "input":
            c = input()
        elif c == "human_label":
            human_label()
        elif c == "train":
            train_affordance()
        elif c == "grasp_heuristic":
            grasp_heuristic(camera_color_img, camera_depth_img)
        elif c == "follow":
            follow(robot)
        elif c == "grasp_slide":
            grasp_slide()

        seq_id += 1


if __name__ == "__main__":
    main()
