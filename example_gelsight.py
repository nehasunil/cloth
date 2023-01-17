import time

import cv2
import numpy as np

# from classification_examples import grip_classify
from perception.wedge.gelsight.pose import Classification
from robot import Robot

robot = Robot()


# def test_classification():
#     gs = robot.gs
#
#     labels = ["no fabric", "edge", "all fabric", "corner", "fold"]
#     print(
#         labels[
#             grip_classify(
#                 rx150, 1600, 405, 90, end_ang, rx_roll * 2048 / np.pi, corners
#             )
#         ]
#     )


def test_no_processing():
    gs = robot.get_gelsight(gs_disable_processing=True)

    while True:
        img = gs.stream.image
        unwarped_img = gs.get_unwarped_img(img)

        if unwarped_img is not None:
            cv2.imshow("unwarped_img", unwarped_img)

        c = cv2.waitKey(1) & 0xFF
        if c == ord("q"):
            break


def test_combined():
    # gs = robot.gs
    gs = robot.gs3
    # gs = robot.get_gelsight(gs_disable_processing=True)

    while True:
        img = gs.stream.image
        unwarped_img = gs.get_unwarped_img(img)

        # get pose image
        pose_img = gs.pc.pose_img
        # pose_img = gs.pc.frame_large

        # get tracking image
        tracking_img = gs.tc.tracking_img

        pose = gs.pc.pose

        if img is None:
            continue

        # cv2.imshow("img", img)

        if unwarped_img is not None:
            cv2.imshow("unwarped_img", unwarped_img)

        if pose_img is not None:
            cv2.imshow("pose", pose_img)

        if tracking_img is not None:
            cv2.imshow("marker", tracking_img)

        # cv2.imshow("mask", gs.tc.mask * 1.0)

        if gs.tc.diff_raw is not None:
            cv2.imshow("diff", gs.tc.diff_raw / 255)
        # cv2.imshow("depth", gs.pc.depth / 8.)

        c = cv2.waitKey(1) & 0xFF
        if c == ord("q"):
            break


def test_dual_gelsight():
    gs = robot.gs
    gs2 = robot.gs2
    # gs = robot.get_gelsight(gs_disable_processing=True)

    while True:
        img = gs.stream.image
        unwarped_img = gs.get_unwarped_img(img)
        img2 = gs2.stream.image
        unwarped_img2 = gs2.get_unwarped_img(img2)

        if unwarped_img is not None:
            cv2.imshow("raw", unwarped_img)

        if unwarped_img2 is not None:
            cv2.imshow("raw2", unwarped_img2)

        c = cv2.waitKey(1) & 0xFF
        if c == ord("q"):
            break


def test_three_gelsights():
    gs = robot.gs
    gs2 = robot.gs2
    gs3 = robot.gs3
    # gs = robot.get_gelsight(gs_disable_processing=True)

    while True:
        img = gs.stream.image
        unwarped_img = gs.get_unwarped_img(img)
        img2 = gs2.stream.image
        unwarped_img2 = gs2.get_unwarped_img(img2)
        img3 = gs3.stream.image
        unwarped_img3 = gs3.get_unwarped_img(img3)

        if unwarped_img is not None:
            cv2.imshow("raw", unwarped_img)

        if unwarped_img2 is not None:
            cv2.imshow("raw2", unwarped_img2)

        if unwarped_img3 is not None:
            cv2.imshow("raw3", unwarped_img3)

        c = cv2.waitKey(1) & 0xFF
        if c == ord("q"):
            break


if __name__ == "__main__":
    # test_dual_gelsight()
    # test_three_gelsights()
    # test_no_processing()
    test_combined()
