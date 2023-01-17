import random

import cv2
import numpy as np
from perception.segmentation.grasp_selector import select_grasp


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


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.array(np.unravel_index(indices, ary.shape)).T


def get_upper_mask(img, threshold=0.6):
    x, y = np.where(img[:] == 1)
    x_sort = sorted(x)
    if len(x_sort) == 0:
        # return img
        raise Exception("no fabric in hand")
    x_threshold = x_sort[int(len(x_sort) * threshold)]
    img_top = img.copy()
    img_top[x_threshold:] = 0
    return img_top


class Policy_Affordance:
    def __init__(self):
        pass

    def grasp_policy(
        self, affordance, safety_mask, threshold=0.4, num_largest=1, select_idx=[]
    ):
        # affordance.shape: (224, 224)

        affordance_masked = affordance.copy()

        cv2.imshow("safety mask", safety_mask * 1.0)
        cv2.waitKey(1)
        upper_mask = get_upper_mask(safety_mask)

        affordance_masked[upper_mask == 0] = 0

        print("affordance_masked.max()", affordance_masked.max())
        if affordance_masked.max() < threshold:
            print("threshold", threshold)
            raise Exception("All affordance below threshold")

        affordance_masked = self.remove_similar_pixels(affordance_masked, select_idx)
        # cv2.imshow("affordance_masked", affordance_masked)
        # cv2.waitKey(1)
        idx = largest_indices(affordance_masked, num_largest)

        return random.choice(idx)

    def remove_similar_pixels(self, affordance_raw, select_idx):
        # affordance.shape: (224, 224)
        padding = 8
        affordance = affordance_raw.copy()
        for idx in select_idx:
            x, y = idx
            affordance[x - padding : x + padding + 1, y - padding : y + padding + 1] = 0

        return affordance


class Policy_Heuristic:
    def __init__(self):
        pass

    def grasp_policy(
        self, affordance, safety_mask, threshold=0, num_largest=1, select_idx=[]
    ):
        # affordance.shape: (224, 224)

        affordance_masked = affordance.copy()

        cv2.imshow("safety mask", safety_mask * 1.0)
        cv2.waitKey(1)
        upper_mask = get_upper_mask(safety_mask)

        affordance_masked[upper_mask == 0] = 0

        print("affordance_masked.max()", affordance_masked.max())
        if affordance_masked.max() < threshold:
            print("threshold", threshold)
            raise Exception("All affordance below threshold")

        affordance_masked = self.remove_similar_pixels(affordance_masked, select_idx)
        # cv2.imshow("affordance_masked", affordance_masked)
        # cv2.waitKey(1)
        idx = largest_indices(affordance_masked, num_largest)

        return random.choice(idx)

    def remove_similar_pixels(self, affordance_raw, select_idx):
        # affordance.shape: (224, 224)
        padding = 8
        affordance = affordance_raw.copy()
        for idx in select_idx:
            x, y = idx
            affordance[x - padding : x + padding + 1, y - padding : y + padding + 1] = 0

        return affordance
