import os
import sys
import termios
import time
import tty
from threading import Thread

import cv2
import numpy as np
from dynamixel_sdk import *
from simple_pid import PID

from .rg_serial import SBMotor


def scale_width(pos, min_position, max_position):
    return int(min_position + pos * (max_position - min_position))


class Gripper_Controller_V2(Thread):
    def __init__(
        self,
        DXL_ID_list=[1, 6],
        min_position_list=[938, 1600],  # delibrately swap 1600 and 750 for consistency
        max_position_list=[2700, 550],
    ):
        Thread.__init__(self)
        self.DXL_ID_list = DXL_ID_list
        self.min_position_list = min_position_list
        self.max_position_list = max_position_list

        self.follow_gripper_pos = [0.0, 0.0]
        self.gripper_pos_last = [None, None]
        self.follow_dc_pos = [0, 0]
        self.dc_pos_last = [None, None]
        self.flag_terminate = False

        self.gripper_helper = GripperHelper(DXL_ID_list)

    def follow_dxl(self):
        # Set the position to self.follow_gripper_pos
        for gripper_id in range(len(self.DXL_ID_list)):
            # skip for the same command
            if self.gripper_pos_last[gripper_id] == self.follow_gripper_pos[gripper_id]:
                continue

            self.gripper_pos_last[gripper_id] = self.follow_gripper_pos[gripper_id]
            gripper_width = scale_width(
                self.follow_gripper_pos[gripper_id],
                self.min_position_list[gripper_id],
                self.max_position_list[gripper_id],
            )

            self.gripper_helper.set_gripper_pos(
                self.DXL_ID_list[gripper_id], gripper_width
            )

    def set_left(self, val):
        self.follow_gripper_pos[0] = val

    def set_right(self, val):
        self.follow_gripper_pos[1] = val

    def get_left_dxl(self):
        DXL_ID = self.DXL_ID_list[0]
        return self.gripper_helper.get_dxl_position(DXL_ID)

    def get_right_dxl(self):
        DXL_ID = self.DXL_ID_list[1]
        return self.gripper_helper.get_dxl_position(DXL_ID)

    def follow_dc(self):
        if self.follow_dc_pos == self.dc_pos_last:
            return

        self.gripper_helper.set_dc_pos(self.follow_dc_pos)

    def follow(self):
        self.follow_dxl()
        self.follow_dc()

    def run(self):
        while not self.flag_terminate:
            self.follow()
            time.sleep(0.01)


class GripperHelper(object):
    def __init__(self, DXL_ID_list, DEVICENAME="/dev/ttyUSB0"):
        self.DXL_ID_list = DXL_ID_list
        self.DEVICENAME = DEVICENAME

        self.init()

    def set_dc_pos(self, pos):
        # e.g. pos: [0, 0]
        self.dc_motors.set_all_velocity(pos)

    def set_gripper_pos(self, DXL_ID, pos):
        # Set gripper position, 0-1
        ADDR_MX_GOAL_POSITION = 116
        CurrentPosition = int(pos)
        dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(
            self.portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, CurrentPosition
        )

    def set_gripper_current_limit(self, DXL_ID, current_limit):
        # Set gripper current limit, 0-1
        ADDR_GOAL_CURRENT = 102
        CURRENT_LIMIT_UPBOUND = 1193
        CurrentTorque = int(CURRENT_LIMIT_UPBOUND * current_limit)
        dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(
            self.portHandler, DXL_ID, ADDR_GOAL_CURRENT, CurrentTorque
        )

    def get_dxl_position(self, DXL_ID):
        ADDR_PRO_PRESENT_POSITION = 132

        (
            dxl_present_position,
            dxl_comm_result,
            dxl_error,
        ) = self.packetHandler.read4ByteTxRx(
            self.portHandler, DXL_ID, ADDR_PRO_PRESENT_POSITION
        )
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        return dxl_present_position

    def init(self):

        ################################################################################################################
        # setup dc motor

        motor_cpr = 3040.7596
        com_baud = 1000000

        print("Establishing Serial port...")
        self.dc_motors = SBMotor("/dev/cu.usbmodem123", com_baud)
        print(self.dc_motors.ser)
        # dc_motors.init_single_motor(4, motor_cpr, 15.0, 0.2, 100.0, ctrl_mode=1) # velocity control
        self.dc_motors.init_single_motor(
            4, motor_cpr, 14.0, 0.0, 0.3, ctrl_mode=0
        )  # position control
        # dc_motors.init_single_motor(5, motor_cpr, 15.0, 0.2, 100.0, ctrl_mode=1)
        self.dc_motors.init_single_motor(5, motor_cpr, 14.0, 0.0, 0.3, ctrl_mode=0)

        ################################################################################################################
        # setup for the motor
        ADDR_MX_TORQUE_ENABLE = (
            64  # Control table address is different in Dynamixel model
        )
        ADDR_MX_PRESENT_POSITION = 132

        # Protocol version
        PROTOCOL_VERSION = 2.0  # See which protocol version is used in the Dynamixel

        # Default setting

        BAUDRATE = 57600  # Dynamixel default baudrate : 57600
        # DEVICENAME                  = '/dev/tty.usbserial-FT2N061F'    # Check which port is being used on your controller
        # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"
        DEVICENAME = (
            self.DEVICENAME
        )  # Check which port is being used on your controller
        # ex)

        TORQUE_ENABLE = 1  # Value for enabling the torque
        TORQUE_DISABLE = 0  # Value for disabling the torque
        DXL_MOVING_STATUS_THRESHOLD = 5

        # Initialize PortHandler instance
        # Set the port path
        # Get methods and members of PortHandlerLinux or PortHandlerWindows
        portHandler = PortHandler(DEVICENAME)
        self.portHandler = portHandler

        # Initialize PacketHandler instance
        # Set the protocol version
        # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
        packetHandler = PacketHandler(PROTOCOL_VERSION)
        self.packetHandler = packetHandler

        # Open port
        if portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")

        # Set port baudrate
        if portHandler.setBaudRate(BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")

        # Enable Dynamixel Torque
        for DXL_ID in self.DXL_ID_list:
            dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
                portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE
            )
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))
            else:
                print("Dynamixel has been successfully connected")

            # Changing operating mode
            ADDR_OPERATING_MODE = 11
            OP_MODE_POSITION = 5
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(
                portHandler, DXL_ID, ADDR_OPERATING_MODE, OP_MODE_POSITION
            )

            # set the current limit
            ADDR_CURRENT_LIMIT = 38
            CURRENT_LIMIT_UPBOUND = 1193
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(
                portHandler, DXL_ID, ADDR_CURRENT_LIMIT, CURRENT_LIMIT_UPBOUND
            )

            # SET THE VELOCITY LIMIT
            ADDR_VELOCITY_LIMIT = 44
            VELOCITY_LIMIT_UPBOUND = 1023
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(
                portHandler, DXL_ID, ADDR_VELOCITY_LIMIT, VELOCITY_LIMIT_UPBOUND
            )

            # SET THE GOAL VELOCITY
            ADDR_GOAL_VELOCITY = 104
            GOAL_VELOCITY_MAXPOSITION = 1023
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(
                portHandler, DXL_ID, ADDR_GOAL_VELOCITY, GOAL_VELOCITY_MAXPOSITION
            )

            ADDR_ACCELERATION_PROFILE = 108
            ACCELERATION_ADDRESS_POSITION = 0
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(
                portHandler,
                DXL_ID,
                ADDR_ACCELERATION_PROFILE,
                ACCELERATION_ADDRESS_POSITION,
            )

            ADDR_VELOCITY_PROFILE = 112
            VELOCITY_ADDRESS_POSITION = 0
            dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(
                portHandler, DXL_ID, ADDR_VELOCITY_PROFILE, VELOCITY_ADDRESS_POSITION
            )

            # Enable Dynamixel Torque
            dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
                portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE
            )
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % packetHandler.getRxPacketError(dxl_error))
            else:
                print("Dynamixel has been successfully connected")
