from threading import Thread
import numpy as np
import socket
import time
import urx
from scipy.spatial.transform import Rotation as R


def clip_speed(v, vmax):
    v_norm = np.sum(v[:3] ** 2) ** 0.5
    if v_norm > vmax:
        v[:3] = v[:3] / v_norm * vmax
    return v


class UR_Controller(Thread):
    def __init__(self, HOST="10.42.0.121", PORT=30003):
        Thread.__init__(self)

        self.rob = urx.Robot(HOST, use_rt=True)

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((HOST, PORT))

        self.flag_terminate = False
        self.pose_following = None

        # Check whether the robot has reach the target pose
        self.last_p = []
        self.len_p = 5
        self.total_error = 0

    def getl_rt(self):
        # get current pose with urx urrtmon with 125 Hz
        return self.rob.rtmon.getTCF(True)

    def send(self, cmd):
        cmd = str.encode(cmd)
        self.s.send(cmd)

    def speedl(self, v, a=0.5, t=0.05):
        # send speedl command in socket
        cmd = "speedl({}, a={}, t={})\n".format(str(list(v)), a, t)
        self.send(cmd)

    def movej_pose(self, pose, a=1, v=0.4, wait=True):
        # joint move in tool space and wait
        s = self.s

        cmd = "movej(p{}, a={}, v={})\n".format(str(list(pose)), a, v)
        cmd = str.encode(cmd)
        s.send(cmd)
        time.sleep(0.2)

        if wait:
            self.wait_following()

    def movej_joints(self, joints, a=1, v=0.4, wait=True):
        # joint move in tool space and wait
        s = self.s

        cmd = "movej({}, a={}, v={})\n".format(str(list(joints)), a, v)
        cmd = str.encode(cmd)
        s.send(cmd)
        time.sleep(0.2)

        if wait:
            self.wait_following()

    def movel_wait(self, pose, a=2.5, v=0.25):
        # linear move in tool space and wait
        s = self.s

        cmd = "movel(p{}, a={}, v={})\n".format(str(list(pose)), a, v)
        cmd = str.encode(cmd)
        s.send(cmd)
        time.sleep(0.2)

        self.wait_following()

    def follow(self, vmax=0.2):
        if self.pose_following is None:
            return
        cur_pose = self.getl_rt()
        error = self.pose_following - cur_pose
        kp = 5
        ki = 0.1
        self.total_error = self.total_error * 0.9 + error
        v = error * kp + self.total_error * ki
        v = clip_speed(v, vmax)
        cmd = np.zeros(6)
        cmd[:3] = v[:3]
        # print("cur", cur_pose, "goal", self.pose_following)
        self.speedl(cmd, a=1, t=0.05)

    def wait_following(self, len_p=20, threshold=1e-6):
        last_p = []
        # len_p = 20

        while True:
            p = self.getl_rt()

            last_p.append(p.copy())
            if len(last_p) > len_p:
                last_p = last_p[-len_p:]

            diff = np.sum((p - np.mean(last_p, axis=0)) ** 2)

            if len(last_p) == len_p and diff < threshold:
                break
            time.sleep(0.02)

    def movel_nowait(self, pose, a=1, v=0.04):
        # linear move in tool space and wait
        s = self.s

        cmd = "movel(p{}, a={}, v={})\n".format(str(list(pose)), a, v)
        cmd = str.encode(cmd)
        s.send(cmd)

    def clear_history(self):
        self.last_p = []

    def check_stopped(self):
        p = self.getl_rt()

        self.last_p.append(p.copy())
        if len(self.last_p) > self.len_p:
            self.last_p = self.last_p[-self.len_p :]

        diff = np.sum((p - np.mean(self.last_p, axis=0)) ** 2)

        if len(self.last_p) == self.len_p and diff < 1e-12:
            return True

        return False

    def run(self):
        while not self.flag_terminate:
            self.follow()
            time.sleep(0.01)
        self.rob.close()


def main():
    urc = UR_Controller()
    urc.start()

    pose0 = np.array([-0.431, 0.05, 0.21, -2.23, -2.194, -0.019])
    urc.movel_wait(pose0)
    time.sleep(2)
    pose = pose0.copy()
    for i in range(120):
        # print(urc.getl_rt())
        pose[2] = pose0[2] + np.sin(i / 40 * np.pi) * 0.03
        urc.pose_following = pose
        time.sleep(0.05)
    urc.flag_terminate = True
    urc.join()


def test_thread():
    urc_l = UR_Controller(HOST="10.42.0.121")
    urc_r = UR_Controller(HOST="10.42.0.2")
    urc_l.start()
    urc_r.start()

    print(urc_r.getl_rt())

    pose0_l = np.array([-0.41, -0.13, 0.15, 0.0, 3.14, 0.0])
    urc_l.movel_wait(pose0_l)

    # pose0_r = np.array([0.07, -0.41, 0.15, 0.0, 3.14, 0.0])
    pose0_r = np.array([-0.02, -0.41, 0.15, -2.2, 2.2, 0.0])
    urc_r.movel_wait(pose0_r)

    pose_l = pose0_l.copy()
    pose_r = pose0_r.copy()

    for i in range(480):
        pose_l[2] = pose_l[2] + np.sin(i / 120 * np.pi) * 0.001
        pose_r[2] = pose_r[2] + np.sin(i / 120 * np.pi) * 0.001
        urc_l.pose_following = pose_l
        urc_r.pose_following = pose_r
        time.sleep(0.01)

    urc_l.flag_terminate = True
    urc_l.join()
    urc_r.flag_terminate = True
    urc_r.join()


if __name__ == "__main__":
    test_thread()
