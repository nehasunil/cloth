import datetime
import glob
import os
import time

import cv2
import numpy as np
import torch


def draw_grasp_point(x, best_pix_ind):
    depth_img = np.dstack([x] * 3)
    w = 7
    h = 5
    x, y = best_pix_ind
    start_point = (y - w, x - h)
    end_point = (y + w, x + h)
    color = (0, 0, 255)
    thickness = 1
    image = cv2.rectangle(depth_img, start_point, end_point, color, thickness)
    return image


class GraspItem:
    def __init__(self):
        pass

    def save(self, filename):
        with open(filename, "wb") as f:
            # camera
            np.save(f, self.color)
            np.save(f, self.depth)
            np.save(f, self.x)

            # grasping point
            np.save(f, self.best_pix_ind)

            # tactile
            np.save(f, self.frames)
            np.save(f, self.frame0_blur)
            np.save(f, self.frames2)
            np.save(f, self.frame0_2_blur)

            # labels
            np.save(f, self.y_gt)
            np.save(f, self.y_class_probs)

    def load(self, filename):
        with open(filename, "rb") as f:
            # camera
            self.color = np.load(f)
            self.depth = np.load(f)
            self.x = np.load(f)

            # grasping point
            self.best_pix_ind = np.load(f)

            # tactile
            self.frames = np.load(f)
            self.frame0_blur = np.load(f)
            self.frames2 = np.load(f)
            self.frame0_2_blur = np.load(f)

            # labels
            self.y_gt = np.load(f)
            self.y_class_probs = np.load(f)

    def visualize(self):
        print(self.x.shape)
        cv2.imshow("color", self.color)
        print("best_pix_ind", self.best_pix_ind)
        image_grasp = draw_grasp_point(self.x, self.best_pix_ind)
        cv2.imshow("image_grasp", image_grasp)
        diff = (self.frames[-1] * 1.0 - self.frame0_blur) * 4 / 255.0 + 0.5
        cv2.imshow("tactile_1", diff)
        diff2 = (self.frames2[-1] * 1.0 - self.frame0_2_blur) * 4 / 255.0 + 0.5
        cv2.imshow("tactile_2", diff2)
        print("y_gt", self.y_gt)
        print("y_class_probs", self.y_class_probs)
        cv2.waitKey(0)


class Logger:
    def __init__(
        self,
        logging_directory,
        test_directory="data/affordance/20220602_test",
        iteration=0,
    ):
        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        self.iteration = iteration
        self.logs_success = []
        self.logs_fail = []

        self.base_directory = logging_directory

        self.grasp_directory = os.path.join(self.base_directory, "data")
        self.eval_directory = os.path.join(self.base_directory, "eval")
        self.visualize_directory = os.path.join(self.base_directory, "visualize")

        self.test_directory = os.path.join(test_directory, "data")
        self.test_label_directory = os.path.join(test_directory, "eval")
        self.grasp_item = GraspItem()

        if not os.path.exists(self.grasp_directory):
            os.makedirs(self.grasp_directory)
        os.makedirs(self.eval_directory, exist_ok=True)

        if iteration == -1:
            data_path = os.path.join(self.grasp_directory, "*.npy")
            self.iteration = len(glob.glob(data_path))

    def update_camera(self, color, depth, x):
        self.grasp_item.color = color
        self.grasp_item.depth = depth
        self.grasp_item.x = x

    def update_selected_pixel(self, x, y):
        self.grasp_item.best_pix_ind = [x, y]

    def update_tactile(self, frames, frame0_blur, frames2, frame0_2_blur):
        # frame0 is after blurring and resizing
        self.grasp_item.frames = frames
        self.grasp_item.frame0_blur = frame0_blur
        self.grasp_item.frames2 = frames2
        self.grasp_item.frame0_2_blur = frame0_2_blur

    def update_y_gt(self, y_gt, y_class_probs):
        self.grasp_item.y_gt = y_gt
        self.grasp_item.y_class_probs = y_class_probs

    def save_log(self, human_label=None):
        filename = os.path.join(
            self.grasp_directory, "%06d.data.npy" % (self.iteration)
        )
        self.grasp_item.save(filename)

        if human_label is not None:

            filename_eval = os.path.join(
                self.eval_directory, "%06d.data_eval.npy" % (self.iteration)
            )
            np.save(filename_eval, human_label)

        print(f"affordance data collected, iteration {self.iteration}")
        self.grasp_item = GraspItem()
        self.iteration += 1

    def get_log(self, iteration, test=False):
        if test:
            filename = os.path.join(self.test_directory, "%06d.data.npy" % (iteration))
        else:
            filename = os.path.join(self.grasp_directory, "%06d.data.npy" % (iteration))
        grasp_item = GraspItem()
        grasp_item.load(filename)
        return grasp_item


if __name__ == "__main__":
    logger = Logger(logging_directory="data/affordance/20220520", iteration=0)
    grasp_item = logger.get_log(iteration=0)
    grasp_item.visualize()
