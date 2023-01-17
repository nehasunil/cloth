import glob
import os
import random
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from logger import Logger, draw_grasp_point
from scipy.ndimage import gaussian_filter
from sklearn import metrics
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import type_of_target
from transform import Transform
from perception.segmentation.segment import Segmentation

# from model import Model
# from unet_model import UNet


class SegEvaluation:
    def __init__(self, logger, segment):
        self.logger = logger
        self.segment = segment

    def visualize_prediction(self, y_pred, depth):
        y_pred_vis = y_pred  # / 0.5
        print(y_pred.min(), y_pred.max())
        y_pred_vis = np.clip(y_pred_vis * 2, 0, 1)
        y_pred_vis = (y_pred_vis * 255).astype(np.uint8)
        im_color = cv2.applyColorMap(y_pred_vis, cv2.COLORMAP_JET)
        combined = im_color.copy() * 1.0
        for c in range(3):
            combined[:, :, c] = (depth / 2) + (im_color[:, :, c] / 255.0 / 2)
        cv2.imshow("affordance", combined)
        cv2.waitKey(0)

    def get_precision_at_k(self, y_true, y_score, k, pos_label=1):
        y_true_type = type_of_target(y_true)
        if not (y_true_type == "binary"):
            raise ValueError("y_true must be a binary column.")

        # Makes this compatible with various array types
        y_true_arr = column_or_1d(y_true)
        y_score_arr = column_or_1d(y_score)

        y_true_arr = y_true_arr == pos_label

        desc_sort_order = np.argsort(y_score_arr)[::-1]
        y_true_sorted = y_true_arr[desc_sort_order]
        y_score_sorted = y_score_arr[desc_sort_order]

        true_positives = y_true_sorted[:k].sum()

        return true_positives / k

    def get_auc(self, y_true, y_score):
        y_true = np.array([1, 1, 2, 2])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=2)
        auc = metrics.auc(fpr, tpr)
        return auc

    def get_accuracy(self, y_true, y_score, thresh):
        correct = 0
        total = 0
        for i in range(len(y_true)):
            # print("y_gt: ", y_gt, "y_pred_val: ", y_pred_val)
            if (y_true[i] > 0.9 and y_score[i] > thresh) or (
                y_true[i] < 0.9 and y_score[i] < thresh
            ):
                correct += 1
            total += 1
        return correct / total

    def get_global_pixel(self, x_local, y_local, crop_x, crop_y):
        x = int(x_local / 224.0 * (crop_x[1] - crop_x[0]) + crop_x[0])
        y = int(y_local / 224.0 * (crop_y[1] - crop_y[0]) + crop_y[0])
        return x, y

    def get_local_pixel(self, x_global, y_global, crop_x, crop_y):
        x = int(x_global - crop_x[0])
        y = int(y_global - crop_y[0])
        return x, y

    def get_seg_pixel(self, x_affordance, y_affordance):
        crop_x_seg = [150, 800]
        crop_y_seg = [600, 1250]
        crop_x_affordance = [100, 550]
        crop_y_affordance = [700, 1150]

        x_global, y_global = self.get_global_pixel(
            x_affordance, y_affordance, crop_x_affordance, crop_y_affordance
        )
        x_seg, y_seg = self.get_local_pixel(x_global, y_global, crop_x_seg, crop_y_seg)
        return x_seg, y_seg

    def get_predictions(self):
        logger = self.logger

        y_true = []
        y_score = []

        transform_l = Transform(log_dir_cam2robot="logs/calibration_ur5_l/")

        n = len(glob.glob(logger.test_directory + "/*.npy"))
        for idx in range(n):
            grasp_item = self.logger.get_log(iteration=idx, test=True)
            x = grasp_item.x
            color = grasp_item.color
            depth = grasp_item.depth

            best_pix_ind = grasp_item.best_pix_ind
            y_tac = grasp_item.y_gt

            y_pred = self.segment.get_heuristic(color, depth, transform_l)
            # y_pred = afford.get_affordance(x)

            filename = os.path.join(
                logger.test_label_directory, "%06d.data_eval.npy" % (idx)
            )
            with open(filename, "rb") as f:
                y_gt = np.load(f).astype(float)
            y_true.append(y_gt)

            pix_seg = self.get_seg_pixel(best_pix_ind[0], best_pix_ind[1])
            y_pred_val = y_pred[pix_seg[0], pix_seg[1]]
            y_score.append(y_pred_val)
        return y_true, y_score

    def evaluate(self, k=40):
        y_true, y_score = self.get_predictions()
        # auc = self.get_auc(y_true, y_score)
        # print("true: ", np.sum(y_true), "false:", len(y_true) - np.sum(y_true))
        precisionk = self.get_precision_at_k(y_true, y_score, k)
        return precisionk

    def evaluate_accuracy(self, threshold=0.9):
        y_true, y_score = self.get_predictions()
        # auc = self.get_auc(y_true, y_score)
        # print("true: ", np.sum(y_true), "false:", len(y_true) - np.sum(y_true))
        accuracy = self.get_accuracy(y_true, y_score, thresh=threshold)
        return accuracy


def eval_segmentation():
    logger = Logger(logging_directory="data/affordance/20220602", iteration=0)

    segment = Segmentation()
    seg_eval = SegEvaluation(logger, segment)
    # for k in range(5, 51, 5):
    # k = 40
    # precisionk = seg_eval.evaluate(k=k)
    # print(f"precision@{k}", precisionk)

    y_true, y_score = seg_eval.get_predictions()
    for thresh in np.arange(0, 1.01, 0.01):
        print(
            "Threshold: ",
            thresh,
            "    Accuracy: ",
            seg_eval.get_accuracy(y_true, y_score, thresh=thresh),
        )


if __name__ == "__main__":
    eval_segmentation()
