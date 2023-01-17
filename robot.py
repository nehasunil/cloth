import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from controller.gripper.gripper_control_v2 import Gripper_Controller_V2
from controller.ur5.ur_controller import UR_Controller
from perception.kinect.kinect_camera import Kinect
from perception.wedge.gelsight.gelsight_driver import GelSight
from transform import Transform


def read_gelsight_csv(filename):
    rows = []
    with open(filename, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)
        for row in csvreader:
            rows.append((int(row[1]), int(row[2])))
    return rows


class Robot:
    def __init__(self, gs_disable_processing=False):
        self.gs_disable_processing = gs_disable_processing
        self.setup()

    def setup(self):
        self.setup_transform()
        # self.setup_camera()
        self.setup_ur_controller()
        self.setup_gripper_controller()
        self.setup_gelsight()

    def setup_transform(self):
        self.transform_l = Transform(log_dir_cam2robot="logs/calibration_ur5_l/")
        self.transform_r = Transform(log_dir_cam2robot="logs/calibration_ur5_r/")

    def setup_camera(self):
        self.camera = Kinect(
            self.transform_l.cam_intrinsics_origin, self.transform_l.dist
        )

    def setup_ur_controller(self):
        self.urc_l = UR_Controller(HOST="10.42.0.121")
        self.urc_r = UR_Controller(HOST="10.42.0.2")
        self.urc_l.start()
        self.urc_r.start()

        self.pose_away_l = np.array([-0.39, 0.16, 0.07, 2.22, 2.22, 0.0])
        self.pose_away_r = np.array([-0.42, -0.16, 0.05, -2.2, 2.2, 0.0])

    def setup_gripper_controller(self):
        # Start gripper
        DXL_ID_list = [1, 6]
        self.grc = Gripper_Controller_V2(DXL_ID_list)
        self.grc.follow_gripper_pos = [0.1, 0.0]
        # self.grc.follow_gripper_pos = [0.1, 0.9]
        # self.grc.follow_gripper_pos = [0.8, 0.0]
        self.grc.follow_gripper_pos = [0.4, 0.0]
        # self.grc.follow_gripper_pos = [0.6, 0.9]
        self.grc.follow_dc_pos = [-10, 10]
        # self.grc.follow_dc_pos = [0, 0]
        self.grc.gripper_helper.set_gripper_current_limit(DXL_ID_list[0], 0.3)
        self.grc.start()

    def get_gelsight(
        self,
        IP="http://rpigelsightfabric.local",
        sensor_id="fabric_0",
        gs_disable_processing=False,
        pose_enable=True,
        tracking_enable=True,
    ):

        #                   N   M  fps x0  y0  dx  dy
        tracking_setting = (10, 14, 30, 16, 41, 27, 27)

        corners = tuple(read_gelsight_csv(f"perception/wedge/config_{sensor_id}.csv"))

        gs = GelSight(
            IP=IP,
            corners=corners,
            tracking_setting=tracking_setting,
            output_sz=(400, 300),
            id="right",
            tracking_enable=tracking_enable,
            pose_enable=pose_enable,
            disable_processing=gs_disable_processing,
        )
        gs.start()
        return gs

    def setup_gelsight(self):
        sensor_id = "fabric_0"
        IP = "http://rpigelsightfabric.local"
        self.gs = self.get_gelsight(
            IP, sensor_id, self.gs_disable_processing, tracking_enable=False
        )
        print("gs 1 done")

        sensor_id2 = "fabric_1"
        IP2 = "http://rpigelsight2.local"
        self.gs2 = self.get_gelsight(IP2, sensor_id2, gs_disable_processing=True)
        print("gs 2 done")

        sensor_id3 = "fabric_2"
        IP3 = "http://rpigelsightfabric2.local"
        self.gs3 = self.get_gelsight(
            IP3, sensor_id3, gs_disable_processing=False, pose_enable=False
        )
        print("gs 3 done")

    def transform_l2r(self, L_p):
        return self.transform_l.get_other_robot_position(self.transform_r.cam_pose, L_p)

    def transform_r2l(self, R_p):
        return self.transform_r.get_other_robot_position(self.transform_l.cam_pose, R_p)

    def get_l_in_r(self):
        # get the pose of the left arm in the frame of the right arm
        pose_l = self.urc_l.getl_rt()
        L_p = pose_l[:3]
        R_p = self.transform_l2r(L_p)
        return R_p

    def get_r_in_l(self):
        # get the pose of the right arm in the frame of the left arm
        pose_r = self.urc_r.getl_rt()
        R_p = pose_r[:3]
        L_p = self.transform_r2l(R_p)
        return L_p

    def get_camera_image(self):
        camera_color_img, camera_depth_img = self.camera.get_image()
        return camera_color_img, camera_depth_img

    def __del__(self):
        self.kill_urc()
        self.kill_grc()

    def kill_urc(self):
        self.urc_l.flag_terminate = True
        self.urc_l.join()
        self.urc_r.flag_terminate = True
        self.urc_r.join()

    def kill_grc(self):
        self.grc.flag_terminate = True
        self.grc.join()
