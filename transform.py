import numpy as np


class Transform:
    def __init__(self, log_dir_cam2robot="logs/calibration_ur5_l/"):
        self.log_dir_cam2robot = log_dir_cam2robot
        self.load_config()

    def load_config(self):
        self.load_camera_matrix()
        self.load_cam2robot()

    def load_cam2robot(self):
        log_dir = self.log_dir_cam2robot
        self.cam_pose = np.loadtxt(log_dir + "real/camera_pose.txt", delimiter=" ")
        self.cam_depth_scale = np.loadtxt(
            log_dir + "real/camera_depth_scale.txt", delimiter=" "
        )

    def load_camera_matrix(self):
        self.cam_intrinsics_origin = np.loadtxt(
            "logs/cam_intrinsics_origin.txt", delimiter=" "
        )
        self.cam_intrinsics = np.loadtxt("logs/cam_intrinsics.txt", delimiter=" ")
        self.dist = np.loadtxt("logs/dist.txt", delimiter=" ")

    def get_robot_from_depth_array(self, depth, crop_x=None, crop_y=None):
        if crop_x is None:
            crop_x = [0, depth.shape[1]]
        if crop_y is None:
            crop_y = [0, depth.shape[0]]

        depth_crop = depth[crop_y[0] : crop_y[1], crop_x[0] : crop_x[1]]
        x = np.arange(depth_crop.shape[0]) + crop_y[0]
        y = np.arange(depth_crop.shape[1]) + crop_x[0]
        yy, xx = np.meshgrid(x, y)
        yy = yy.reshape([-1])
        xx = xx.reshape([-1])

        zz = (depth_crop.T).reshape([-1])

        zz = zz * self.cam_depth_scale
        zz[zz == 0] = -1000
        xx = np.multiply(xx - self.cam_intrinsics[0][2], zz / self.cam_intrinsics[0][0])
        yy = np.multiply(yy - self.cam_intrinsics[1][2], zz / self.cam_intrinsics[1][1])
        xyz_kinect = np.vstack([xx, yy, zz])
        # shape: (3, W*H)

        camera2robot = self.cam_pose
        xyz_robot = np.dot(camera2robot[0:3, 0:3], xyz_kinect) + camera2robot[0:3, 3:]

        return xyz_robot

    def get_robot_from_depth(self, x, y, d):
        # Get click point in camera coordinates
        click_z = d * self.cam_depth_scale
        click_x = np.multiply(
            x - self.cam_intrinsics[0][2], click_z / self.cam_intrinsics[0][0]
        )
        click_y = np.multiply(
            y - self.cam_intrinsics[1][2], click_z / self.cam_intrinsics[1][1]
        )

        click_point = np.asarray([click_x, click_y, click_z])
        click_point.shape = (3, 1)

        # Convert camera to robot coordinates
        # camera2robot = np.linalg.inv(robot.cam_pose)
        camera2robot = self.cam_pose
        target_position = (
            np.dot(camera2robot[0:3, 0:3], click_point) + camera2robot[0:3, 3:]
        )

        target_position = target_position[0:3, 0]
        return target_position

    def get_other_robot_position(self, R_X_C, L_p):
        # Transform left-arm position L_p to right-arm position R_p
        # R_p = R_X_C * C_X_L * L_p
        # L_p: (1, 3)
        L_p = L_p.reshape([1, 3]).T
        L_p = np.vstack([L_p, [1]])

        L_X_C = self.cam_pose
        C_X_L = np.linalg.inv(L_X_C)

        R_p = R_X_C @ C_X_L @ L_p
        R_p = R_p.T
        return R_p[0, :3]
