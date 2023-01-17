from .run import Run
from PIL import Image
import skimage.measure
import numpy as np
import torchvision.transforms as T
import cv2
import os
from .grasp_selector import select_grasp


class Segmentation:
    def __init__(self):
        self.load_model()

    def load_model(self):
        model_id = 29
        epoch = 200

        dir_path = os.path.dirname(os.path.realpath(__file__))
        pretrained_model = os.path.join(
            dir_path,
            "models/%d/chkpnts/%d_epoch%d"
            % (
                model_id,
                model_id,
                epoch,
            ),
        )
        self.model = Run(model_path=pretrained_model, n_features=3)

    def seg_output(self, depth):
        max_d = np.nanmax(depth)
        depth[np.isnan(depth)] = max_d
        # depth_min, depth_max = 400.0, 1100.0
        # depth = (depth - depth_min) / (depth_max - depth_min)
        # depth = depth.clip(0.0, 1.0)

        img_depth = Image.fromarray(depth)
        transform = T.Compose([T.ToTensor()])
        img_depth = transform(img_depth)
        img_depth = np.array(img_depth[0])

        out = self.model.evaluate(img_depth).squeeze()
        seg_pred = out[:, :, :3]

        #         prob_pred *= mask
        # seg_pred_th = deepcopy(seg_pred)
        # seg_pred_th[seg_pred_th < 0.8] = 0.0

        return seg_pred

    def get_segmentation(self, color, depth_transformed, crop_x, crop_y):
        # Crop images
        color_small = cv2.resize(color, (0, 0), fx=1, fy=1)
        color_small = color_small[crop_x[0] : crop_x[1], crop_y[0] : crop_y[1]]

        depth_transformed_small = cv2.resize(depth_transformed, (0, 0), fx=1, fy=1)
        depth_transformed_small = depth_transformed_small[
            crop_x[0] : crop_x[1], crop_y[0] : crop_y[1]
        ]

        # depth_min = 400.0
        # depth_max = 1100.0

        depth_min = 500.0
        depth_max = 1200.0
        depth_transformed_small_img = (depth_transformed_small - depth_min) / (
            depth_max - depth_min
        )
        depth_transformed_small_img = depth_transformed_small_img.clip(0, 1)

        # cv2.imshow("seg_color_crop", color_small)
        # cv2.imshow("seg_depth_crop", depth_transformed_small_img)

        depth_transformed_100x100 = skimage.measure.block_reduce(
            depth_transformed_small_img, (4, 4), np.mean
        )
        seg_pred = self.seg_output(depth_transformed_100x100)
        mask = seg_pred.copy()
        mask_corners = mask[:, :, 0] > 0.85

        H, W = mask.shape[0], mask.shape[1]
        mask = (
            mask.reshape((H * W, 3))
            @ np.array([[0, 0, 1.0], [0, 1.0, 1.0], [0, 1.0, 0]])
        ).reshape((H, W, 3))
        # mask = 1.0 / (1 + np.exp(-5 * (mask - 0.8)))

        mask = cv2.resize(
            mask, (depth_transformed_small.shape[0], depth_transformed_small.shape[1])
        )
        mask_corners = cv2.resize(
            mask_corners * 1.0,
            (depth_transformed_small.shape[0], depth_transformed_small.shape[1]),
        )

        mask_resize = cv2.resize(mask, (224, 224))
        cv2.imshow("seg_prediction", mask_resize)
        # cv2.imshow("mask_corners", mask_corners * 1.0)
        cv2.waitKey(1)

        return mask, mask_corners, depth_transformed_small, seg_pred

    def get_heuristic(self, camera_color_img, camera_depth_img, transform):
        crop_x_seg = [150, 800]
        crop_y_seg = [600, 1250]
        (
            segmentation,
            mask_corners,
            depth_transformed_small,
            seg_pred,
        ) = self.get_segmentation(
            camera_color_img, camera_depth_img, crop_x_seg, crop_y_seg
        )

        surface_normal_small = get_surface_normal(
            transform,
            camera_color_img,
            camera_depth_img,
            crop_x_normal=crop_x_seg,
            crop_y_normal=crop_y_seg,
        )

        # print("surface_normal_small.shape", surface_normal_small.shape)
        # cv2.imshow("surface_normal", surface_normal_small)

        surface_normal_100x100 = cv2.resize(
            surface_normal_small, (seg_pred.shape[1], seg_pred.shape[0])
        )

        feeding_mask = np.ones(seg_pred.shape[:2])
        depth_transformed_reduced = skimage.measure.block_reduce(
            depth_transformed_small, (4, 4), np.mean
        )
        feeding_mask[depth_transformed_reduced == 0.0] = 0.0
        # cv2.imshow("feeding_mask_nonzeros", feeding_mask)

        heuristic = select_grasp(
            seg_pred, feeding_mask, surface_normal_100x100, num_neighbour=100
        )
        # outer_pt_x, outer_pt_y, angle, inner_pt_x, inner_pt_y = select_grasp(
        #     seg_pred, feeding_mask, surface_normal_100x100, num_neighbour=100
        # )
        # print(outer_pt_x, outer_pt_y, angle, inner_pt_x, inner_pt_y)
        # img_x, img_y = inner_pt_x * 4 + topleft[1], inner_pt_y * 4 + topleft[0]

        heuristic = cv2.resize(
            heuristic, (crop_x_seg[1] - crop_x_seg[0], crop_y_seg[1] - crop_y_seg[0])
        )

        return heuristic


def get_surface_normal(
    transform,
    camera_color_img,
    camera_depth_img,
    crop_x_normal=[300, 950],
    crop_y_normal=[55, 705],
):

    camera_depth_img_crop = camera_depth_img[
        crop_x_normal[0] : crop_x_normal[1], crop_y_normal[0] : crop_y_normal[1]
    ]
    camera_color_img_crop = camera_color_img[
        crop_x_normal[0] : crop_x_normal[1], crop_y_normal[0] : crop_y_normal[1]
    ]

    xyz_robot = transform.get_robot_from_depth_array(
        camera_depth_img, crop_y_normal, crop_x_normal
    )

    # get gradient
    W, H = camera_depth_img_crop.shape[0], camera_depth_img_crop.shape[1]
    xyz_robot_unflatten = xyz_robot.reshape(3, W, H)
    xx_robot = xyz_robot_unflatten[0]
    yy_robot = xyz_robot_unflatten[1]
    zz_robot = xyz_robot_unflatten[2]

    gx_x, gy_x = np.gradient(xx_robot)
    gx_y, gy_y = np.gradient(yy_robot)
    gx_z, gy_z = np.gradient(zz_robot)
    # gx_x shape: (W, H)

    gx_x_flatten, gy_x_flatten = gx_x.reshape([-1]), gy_x.reshape([-1])
    gx_y_flatten, gy_y_flatten = gx_y.reshape([-1]), gy_y.reshape([-1])
    gx_z_flatten, gy_z_flatten = gx_z.reshape([-1]), gy_z.reshape([-1])
    # gx_x_flatten shape: (W*H)

    gx = np.vstack([gx_x_flatten, gx_y_flatten, gx_z_flatten]).T
    gy = np.vstack([gy_x_flatten, gy_y_flatten, gy_z_flatten]).T
    # gx shape: (W*H, 3)

    normal = np.cross(gx, gy)
    # normal shape: (W*H, 3)

    normal = normal / (np.sum(normal**2, axis=-1, keepdims=True) ** 0.5 + 1e-12)
    normal = normal.reshape(W, H, 3)

    for c in range(3):
        normal[:, :, c] = np.fliplr(np.rot90(normal[:, :, c], k=3))

    normal = cv2.GaussianBlur(normal, (21, 21), 0)

    # sign = (normal[:,:,2] > 0) * 2 - 1
    # for c in range(3):
    #     normal[:,:,c] *= sign

    # cv2.imshow("normal", normal / 2 + 0.5)
    return normal
