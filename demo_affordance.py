from perception.infer_affordance.affordance import Affordance
from perception.segmentation.segment import Segmentation
from perception.kinect.kinect_camera import Kinect
from transform import Transform
import torch
import cv2
import numpy as np


transform_l = Transform(log_dir_cam2robot="logs/calibration_ur5_l/")
camera = Kinect(transform_l.cam_intrinsics_origin, transform_l.dist)


crop_x_affordance = [100, 550]
crop_y_affordance = [700, 1150]


afford = Affordance()
fn_afford = "/home/gelsight/Code/Fabric/src/perception/infer_affordance/models/finetune_0610/best_1_0.700.pt"
# fn_afford = "/home/gelsight/Code/Fabric/src/perception/infer_affordance/models/finetune_0610/best_48_0.800.pt"
afford.model.load_state_dict(torch.load(fn_afford, map_location=afford.device))
segment = Segmentation()


fn_afford_list = [
    "/home/gelsight/Code/Fabric/src/perception/infer_affordance/models/finetune_0610/best_1_0.700.pt",
    # "/home/gelsight/Code/Fabric/src/perception/infer_affordance/models/finetune_0610/best_48_0.800.pt"
    "",
    "/home/gelsight/Code/Fabric/src/perception/infer_affordance/models/nosim_0609/best_19_0.600.pt",
]
model_names = ["finetune_lowres", "sim_to_real", "no_sim"]
threshold_list = [0.3, 0.3, 0.3]


def normalize_depth(depth, vmin, vmax):
    depth = np.clip(depth, vmin, vmax)
    depth_normalized = (depth - vmin) / (vmax - vmin) - 0.5
    return depth_normalized


def filter_cloth_mask(mask, depth_crop):
    mask = mask.reshape((depth_crop.shape[0], depth_crop.shape[1]))
    # cv2.imshow("filtered_mask", mask * 1.0)
    mask = mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = mask.reshape((-1))
    mask = mask.astype(bool)
    return mask


def get_cloth_mask(camera_depth_img, crop_x, crop_y):
    depth_crop = camera_depth_img[crop_x[0] : crop_x[1], crop_y[0] : crop_y[1]]

    xyz_robot = transform_l.get_robot_from_depth_array(camera_depth_img, crop_y, crop_x)
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

    # safety_mask = get_safety_mask(
    #     camera_depth_img, crop_x_affordance, crop_y_affordance
    # )
    # safety_mask_resize = cv2.resize(safety_mask.astype(np.uint8), (224, 224))
    # safety_mask_resize = mask_resize.copy()
    # visualize_affordance_masked(affordance, depth_normalized, safety_mask_resize)

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

        if model_names[model_id] == "no_sim":
            affordance = cv2.GaussianBlur(affordance, (21, 21), 0)

        affordance[mask_resize == 0] = 0

        print("affordance.max()", affordance.max())

        # logger.update_camera(camera_color_img, camera_depth_img, depth_masked)
        # safety_mask = get_safety_mask(
        #     camera_depth_img, crop_x_affordance, crop_y_affordance
        # )
        # safety_mask_resize = cv2.resize(safety_mask.astype(np.uint8), (224, 224))
        # visualize_affordance_masked(affordance, depth_normalized, safety_mask_resize)

        if visualize:
            # cv2.imshow("cloth_mask", cloth_mask * 1.0)
            cv2.imshow("color_crop", color_resize)
            cv2.imshow("depth_crop", depth_normalized + 0.5)
            # cv2.imshow("depth_mask", depth_normalized + 0.5)
            # cv2.imshow("depth_crop", depth_crop / 1100.0)
            # if model_id == 2:
            #     visualize_affordance_masked(
            #         affordance, depth_normalized, safety_mask_resize
            #     )
            # else:
            visualize_prediction(
                affordance, depth_normalized, win_name=model_names[model_id]
            )
            cv2.waitKey(1)
    heuristic = segment.get_heuristic(camera_color_img, camera_depth_img, transform_l)
    heuristic = cv2.resize(heuristic, (224, 224))
    visualize_prediction(heuristic / 2, 0)


def main():

    while True:
        camera_color_img, camera_depth_img = camera.get_image()
        cv2.imshow("color", camera_color_img)
        # cv2.imshow("depth", camera_depth_img / 1100.0)
        cv2.waitKey(1)
        visualize_affordances(camera_color_img, camera_depth_img)


if __name__ == "__main__":
    main()
