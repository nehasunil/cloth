import glob
import os
import time

import cv2
import numpy as np
from perception.wedge.gelsight.pose import Classification
from perception.wedge.gelsight.util.reconstruction import Class_3D, demark
from robot import Robot


class Grip_Classifier:
    def __init__(self, robot):
        self.frame0 = None
        self.robot = robot
        if robot is not None:
            self.gs = robot.get_gelsight(gs_disable_processing=True)
            # self.gs = robot.gs
            self.gs2 = robot.gs2
        self.init_3D_reconstruction()

        self.classifier = Classification()
        self.labels_origin = ["no fabric", "edge", "all fabric", "corner", "fold"]
        self.labels = ["no fabric", "edge", "non-edge", "non-edge", "non-edge"]

        self.n, self.m = 150, 200

    def init_3D_reconstruction(self):
        dir_abs = os.path.dirname(os.path.realpath(__file__))
        sensor_id = "Fabric0"
        model_id = "RGB"
        model_fn = os.path.join(
            dir_abs,
            "perception/wedge/gelsight/models/",
            f"LUT_{sensor_id}_{model_id}.pth",
        )
        self.c3d = Class_3D(model_fn=model_fn, features_type=model_id)

    def open_gripper_r(self):
        self.robot.grc.set_right(0.0)
        time.sleep(0.5)
        print("Opening gripper " * 20)

    def get_frame0(self):
        self.set_gripper_r(1200)

        while True:
            img = self.gs.stream.image
            img2 = self.gs2.stream.image
            if img == "" or img2 == "":
                print("waiting for streaming")
                continue
            unwarped_img = self.gs.get_unwarped_img(img)
            unwarped_img2 = self.gs2.get_unwarped_img(img2)

            self.frame0_raw = unwarped_img.copy()
            self.frame0_raw2 = unwarped_img2.copy()
            self.frame0 = cv2.GaussianBlur(self.frame0_raw, (31, 31), 0)
            self.frame0_2 = cv2.GaussianBlur(self.frame0_raw2, (31, 31), 0)
            break

        return self.frame0, self.frame0_2

    def normalize_width(self, w, min_w, max_w):
        return (w - min_w) / (max_w - min_w)

    def set_gripper_r(self, width):
        gripper_id = 1
        min_w = self.robot.grc.min_position_list[gripper_id]
        max_w = self.robot.grc.max_position_list[gripper_id]
        width_cmd = self.normalize_width(width, min_w, max_w)
        width_cmd = np.clip(width_cmd, 0, 1)
        self.robot.grc.set_right(width_cmd)

    def add_label_text(self, img, class_ind=None):
        img_label = np.ones((40, img.shape[1], img.shape[2]), dtype=np.float32)
        if class_ind == 1:
            img_label[1] = 1
        if class_ind is not None:
            K = 3
            img_label = cv2.resize(img_label, (0, 0), fx=K, fy=K)
            img_label = cv2.putText(
                img_label,
                self.labels_origin[class_ind],
                (0, 25 * K),
                # cv2.FONT_HERSHEY_SIMPLEX,
                cv2.FONT_HERSHEY_DUPLEX,
                1 * K,
                (0, 0, 0),
                1 * K,
                cv2.LINE_AA,
                False,
            )
            img_label = cv2.GaussianBlur(img_label, (3, 3), 0)
            img_label = cv2.resize(img_label, (0, 0), fx=1 / K, fy=1 / K)
        img = np.vstack([img, img_label])
        return img

    def visualize_stacked_image(
        self, frame_list, frame0, name="frames_stacked", class_ind=None
    ):
        frames = np.array(frame_list)
        # frame0_blur = cv2.GaussianBlur(frames[0], (31, 31), 0)
        frame0_blur = cv2.GaussianBlur(frame0, (31, 31), 0)
        frames = (frames * 1.0 - frame0_blur) / 255.0 * 4 + 0.5
        # frames_selected = frames[5::5]

        # frames_selected = frames[::5]
        # frames_selected = frames_selected[-5:]

        # select 5 frames: N-20, N-15, N-10, N-5, N
        frames_selected = frames[::-1][::5][::-1][-5:]

        # frames_stacked = np.stack(np.array(frames_selected), axis=0)
        frames_stacked = np.hstack(np.array(frames_selected))
        # if class_ind is not None:
        #     frames_stacked = self.add_label_text(frames_stacked, class_ind)
        cv2.imshow(name, frames_stacked)
        cv2.waitKey(1)

    def get_selected_depth(self, frame0, select, sensor_id="1"):
        depth_imgs = []
        for frame in select:
            diff = (frame * 1.0 - frame0) / 255.0 + 0.5
            depth, gx, gy = self.c3d.infer(diff * 255.0, demark=demark, display=False)
            #             theta, dx, dy = 0
            #             shifted, mask, M = transform(depth, theta, dx, dy, depth_max=depth_max)
            depth3 = np.dstack([depth] * 3)
            # depth_small = cv2.resize(depth3 / 10.0 + 0.03, (120, 90))
            # depth_small = cv2.resize(depth3 / 10.0 + 0.02, (120, 90))
            depth_small = cv2.resize(depth3 / 10.0, (120, 90))

            depth_imgs.append(depth_small)
            # cv2.imshow("depth_small" + sensor_id, depth_small)

        depth_imgs = np.array(depth_imgs)
        return depth_imgs

    def adjust_frame_rate(self, n):
        dx = n / 45 * 5
        ind = np.linspace(n - dx * 4 - 1, n - 1, 5)
        ind_int = np.round(ind).astype(int)
        return ind_int

    def classify(self, frames, frame0):
        # save data
        # select = frames[::-1][::5][::-1][-5:]
        n = len(frames)
        select_ind = self.adjust_frame_rate(n)
        frames = np.array(frames)
        select = frames[select_ind]

        if len(select) < 5:
            print("not enough frames")
            return -1

        # self.visualize_stacked_image(frames, frame0)

        depth_imgs = self.get_selected_depth(frame0, select, sensor_id="1")

        depth_imgs_stacked = np.concatenate(
            (
                depth_imgs[0],
                depth_imgs[1],
                depth_imgs[2],
                depth_imgs[3],
                depth_imgs[4],
            ),
            axis=2,
        )

        class_ind, class_prob = self.classifier.get_class([depth_imgs_stacked])

        self.visualize_stacked_image(frames, frame0, class_ind=class_ind)

        diff = ((frames[-1] * 1.0 - frame0)) / 255.0 + 0.5
        img_warped = cv2.resize((diff - 0.5) * 4 + 0.5, (self.m, self.n))
        img_warped_text = self.add_label_text(img_warped, class_ind)
        cv2.imshow("warped", img_warped_text)
        cv2.waitKey(1)

        return class_ind, class_prob

    def classify_2finger(self, frames, frame0, frames2, frame0_2):
        # save data
        # select = frames[::5]
        # select = select[-5:]
        select = frames[::-1][::5][::-1][-5:]
        select2 = frames2[::-1][::5][::-1][-5:]

        if len(select) < 5:
            print("not enough frames")
            return -1

        self.visualize_stacked_image(frames, frame0, name="visualize_stacked_image_1")
        self.visualize_stacked_image(
            frames2, frame0_2, name="visualize_stacked_image_2"
        )

        depth_imgs = self.get_selected_depth(frame0, select, sensor_id="1")
        depth_imgs2 = self.get_selected_depth(frame0_2, select2, sensor_id="2")

        depth_imgs_stacked = np.concatenate(
            (
                depth_imgs[0],
                depth_imgs[1],
                depth_imgs[2],
                depth_imgs[3],
                depth_imgs[4],
                depth_imgs2[0],
                depth_imgs2[1],
                depth_imgs2[2],
                depth_imgs2[3],
                depth_imgs2[4],
            ),
            axis=2,
        )

        class_ind, class_probs = self.classifier.get_class([depth_imgs_stacked])
        return class_ind, class_probs

    def record(self, im_raw, gripper_width):
        im_small = cv2.resize(im_raw, (0, 0), fx=0.5, fy=0.5)
        self.frames.append(im_small)

    def record2(self, im_raw, gripper_width):
        im_small = cv2.resize(im_raw, (0, 0), fx=0.5, fy=0.5)
        self.frames2.append(im_small)

    def check_diff(self, last_frame, im_raw):
        return (last_frame is not None) and (np.sum(last_frame - im_raw) != 0)

    def clear_record(self):
        self.frames = []
        self.frames2 = []

    def grip_record(self, gripper_width):
        frame0, frame0_2 = self.get_frame0()
        frame0 = cv2.resize(frame0, (self.m, self.n))
        frame0_2 = cv2.resize(frame0_2, (self.m, self.n))

        self.clear_record()

        # set gripper to initial width and wait for consistency
        self.set_gripper_r(gripper_width)
        time.sleep(0.2)

        last_frame = None
        last_frames2 = None
        # gripper_width = 1000

        gripper_width_inc = -10

        flag_recording = True

        while True:
            frame = self.gs.stream.image
            frame2 = self.gs2.stream.image

            im_raw = self.gs.get_unwarped_img(frame)
            im_raw2 = self.gs2.get_unwarped_img(frame2)
            im = cv2.resize(im_raw, (self.m, self.n))
            im2 = cv2.resize(im_raw2, (self.m, self.n))

            diff = ((im * 1.0 - frame0)) / 255.0 + 0.5
            diff2 = ((im2 * 1.0 - frame0_2)) / 255.0 + 0.5

            img_warped = cv2.resize((diff - 0.5) * 4 + 0.5, (self.m, self.n))
            img_warped = self.add_label_text(img_warped, class_ind=None)
            cv2.imshow("warped", img_warped)
            # cv2.imshow("warped2", cv2.resize((diff2 - 0.5) * 4 + 0.5, (self.m, self.n)))
            c = cv2.waitKey(1)

            if flag_recording:
                self.set_gripper_r(gripper_width)

            # time.sleep(0.014)
            time.sleep(0.02)

            if self.check_diff(last_frame, im_raw) and flag_recording:
                self.record(im_raw, gripper_width)

            if self.check_diff(last_frames2, im_raw2) and flag_recording:
                self.record2(im_raw2, gripper_width)

            gripper_width += gripper_width_inc

            if gripper_width < 450 and flag_recording:
                print("len(self.frames)", len(self.frames))
                return self.frames, frame0, self.frames2, frame0_2

            last_frame = im_raw.copy()
            last_frames2 = im_raw2.copy()

    def grip_classify(self, gripper_width):
        frames, frame0, frames2, frame0_2 = self.grip_record(gripper_width)
        class_ind = self.classify(frames, frame0)
        return class_ind


class Logger:
    def __init__(self, category="no_fabric", sensor_id=1):
        self.dir_data = f"data/touch/20220519/{category}/{sensor_id}"
        os.makedirs(self.dir_data, exist_ok=True)

    def save_video(self, frame_list, frame0, video_id):
        # TODO: double-check .mov vs .npy for quality loss
        fn_frame0 = os.path.join(self.dir_data, "frame0_{:03d}.jpg".format(video_id))
        cv2.imwrite(fn_frame0, frame0)

        fn_npy = os.path.join(self.dir_data, "F{:03d}.npy".format(video_id))
        np.save(fn_npy, frame_list)

        fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        fn_video = os.path.join(self.dir_data, "F{:03d}.mov".format(video_id))
        print("fn_video", fn_video)

        col = 200
        row = 150
        out = cv2.VideoWriter(
            fn_video, fourcc, 20.0, (col * 1, row * 1)
        )  # The fps depends on CPU
        for frame in frame_list:
            out.write(frame)
        out.release()


def record_data():
    robot = Robot(gs_disable_processing=True)
    grip_classifier = Grip_Classifier(robot)
    category = "no_fabric"
    logger = Logger(category=category, sensor_id=1)
    logger2 = Logger(category=category, sensor_id=2)
    # logger = Logger(category="corners")

    video_id = 110
    total_num = 110
    pauses = []
    # pauses = [total_num // 4, total_num * 2 // 4, total_num * 3 // 4]
    # pauses = [
    #     total_num * 3 // 10,
    #     total_num * 4 // 10,
    #     total_num * 5 // 10,
    #     total_num * 6 // 10,
    #     total_num * 7 // 10,
    #     total_num * 8 // 10,
    #     total_num * 9 // 10,
    # ]
    print(pauses)

    # time.sleep(3)

    for i in range(total_num):
        print("collected", i)
        frames, frame0, frames2, frame0_2 = grip_classifier.grip_record(1000)
        frame0_raw = grip_classifier.frame0_raw
        frame0_raw2 = grip_classifier.frame0_raw2
        logger.save_video(frames, frame0_raw, video_id)
        logger2.save_video(frames2, frame0_raw2, video_id)
        video_id += 1

        grip_classifier.set_gripper_r(1200)
        # time.sleep(0.5)
        time.sleep(1)
        if i in pauses:
            input()


def load_video(fn):
    cap = cv2.VideoCapture(fn)

    frame_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # diff = (frame * 1.0 - frame0) / 255.0 + 0.5
        frame_list.append(frame)

    cap.release()
    return frame_list


def load_npy(fn):
    frame_list = np.load(fn)
    return frame_list


def load_frame0(data_dir, category, sensor_id, video_id):
    # frame0 = cv2.imread(os.path.join(data_dir, "frame0.jpg"))
    frame0 = cv2.imread(
        os.path.join(
            data_dir, "{}/{}/frame0_{:03d}.jpg".format(category, sensor_id, video_id)
        )
    )
    frame0 = cv2.GaussianBlur(frame0, (31, 31), 0)
    frame0 = cv2.resize(frame0, (200, 150))
    return frame0


def get_video_id(fn):
    fn_last = fn.split("/")[-1]
    elem = fn_last.split("_")[-1]
    elem = elem.split(".")[0]
    elem = elem[1:]
    elem = elem.lstrip("0")
    if elem == "":
        return 0
    return int(elem)


def evaluate(finger2=False):
    robot = None
    grip_classifier = Grip_Classifier(robot)

    # category = "no_fabric"
    # category = "edge"
    # category = "all_fabric"
    # category = "corners"
    category = "fold"
    data_dir = f"data/touch/20220518"
    video_id = 0

    # fn_list = glob.glob(os.path.join(data_dir, category, "*.mov"))
    fn_list = glob.glob(os.path.join(data_dir, category, "1", "*.npy"))
    fn_list = sorted(fn_list)
    fn_list2 = glob.glob(os.path.join(data_dir, category, "2", "*.npy"))
    fn_list2 = sorted(fn_list2)
    # fn_list = fn_list[110:143]
    fn_list = fn_list[110:]
    # fn_list = fn_list[:27] + fn_list[55:]
    # fn_list = fn_list[27:55]

    true_id = 4
    correct = 0
    nonedge = 0
    total = 0

    for fn, fn2 in zip(fn_list, fn_list2):
        video_id = get_video_id(fn)
        frame0 = load_frame0(data_dir, category, "1", video_id)
        # frame_list = load_video(fn)
        frame_list = load_npy(fn)

        if finger2:
            frame0_2 = load_frame0(data_dir, category, "2", video_id)
            frame_list2 = load_npy(fn2)
            class_ind, class_probs = grip_classifier.classify_2finger(
                frame_list, frame0, frame_list2, frame0_2
            )
        else:
            class_ind, class_probs = grip_classifier.classify(frame_list, frame0)

        print(grip_classifier.labels[class_ind], fn)
        print("class_probs", class_probs)
        if class_ind != 1:
            nonedge += 1
        if class_ind == true_id:
            correct += 1
        total += 1

    print("non-edge accuracy", nonedge / total)
    print("total accuracy", correct / total)


def infer_from_video():
    robot = Robot(gs_disable_processing=True)
    grip_classifier = Grip_Classifier(robot)
    data_dir = f"data/touch/20220516"
    category = "no_fabric"
    logger = Logger(category=category)

    video_id = 0
    while True:
        frames, frame0, frames2, frame0_2 = grip_classifier.grip_record(1000)
        frame0_raw = grip_classifier.frame0_raw
        logger.save_video(frame_list, frame0_raw, video_id)

        # frame0 = load_frame0(data_dir, category, video_id)
        fn_video = os.path.join(data_dir, category, "F{:03d}.mov".format(video_id))
        frame_list = load_video(fn_video)

        print("LOAD len(frame_list)", len(frame_list))

        # frame_list = load_npy(fn)
        class_ind, class_probs = grip_classifier.classify(frame_list, frame0)
        print(grip_classifier.labels[class_ind])

        grip_classifier.set_gripper_r(1200)
        # time.sleep(0.5)
        time.sleep(1)


def infer():
    robot = Robot(gs_disable_processing=True)
    grip_classifier = Grip_Classifier(robot)

    while True:
        frames, frame0, frames2, frame0_2 = grip_classifier.grip_record(1000)
        class_ind, class_probs = grip_classifier.classify(frames, frame0)
        print(grip_classifier.labels[class_ind])
        # print(grip_classifier.labels[grip_classifier.grip_classify(1000)])
        grip_classifier.set_gripper_r(1200)
        # grip_classifier.set_gripper_r(1150)
        time.sleep(1)
        # time.sleep(2)


if __name__ == "__main__":
    # record_data()
    # evaluate()
    infer()
    # infer_from_video()
