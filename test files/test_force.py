#!/usr/bin/env python3
import sys
sys.path.append("FoundationPose")
from estimater import *
from datareader import *
from dino_mask import get_mask_from_GD   
from create_camera import CreateRealsense
import cv2
import numpy as np
# import open3d as o3d
import pyrealsense2 as rs
# import torch
import time, os, sys
import json
import threading
from datetime import datetime
import gc
import torch
# from ultralytics.models.sam import Predictor as SAMPredictor
from simple_api import SimpleApi, ForceMonitor, ErrorMonitor
from dobot_gripper import DobotGripper
from transforms3d.euler import euler2mat, mat2euler
from scipy.spatial.transform import Rotation as R
import queue
from spatialmath import SE3, SO3
from grasp_utils import normalize_angle, extract_euler_zyx, print_pose_info
from calculate_grasp_pose_from_object_pose import execute_grasp_from_object_pose, detect_dent_orientation
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge





# ---------- 机械臂 ----------
def init_robot():
    dobot = SimpleApi("192.168.5.1", 29999)
    dobot.clear_error()
    dobot.enable_robot()
    dobot.stop()
    # 启动力传感器
    dobot.enable_ft_sensor(1)
    time.sleep(1)
    # 力传感器置零(以当前受力状态为基准)
    dobot.six_force_home()
    time.sleep(1)
    # 力监控线程
    # force_monitor = ForceMonitor(dobot)
    # force_monitor.start_monitoring()
    # error_monitor = ErrorMonitor(dobot)
    # error_monitor.start_monitoring()
    gripper = DobotGripper(dobot)
    gripper.connect(init=True)
    return dobot, gripper



if __name__ == "__main__":
    # 初始化ROS节点（使用rospy.Rate前必须初始化）
    rospy.init_node('test_force_monitor', anonymous=True)

    # 初始化机械臂
    dobot, gripper = init_robot()


    gripper.control(20, 100, 80)
    wait_grasp = rospy.Rate(1/5)
    wait_grasp.sleep()


    #垂直桌面向下移动玻璃棒，检测是否触碰到桌面
    print("\n开始监测玻璃棒与桌面接触...")

    sample_interval = 0.03  # 秒
    max_force_samples = 15
    force_threshold = 1.6  # N，触碰判定阈值
    consecutive_hits_required = 2

    contact_detected = False
    contact_force = 0.0

    print("开始持续监测力传感器数据")
    
    try:
        while True:
            wait = rospy.Rate(33)
            wait.sleep()

            consecutive_hits = 0
            for _ in range(max_force_samples):
                short_wait = rospy.Rate(1/sample_interval)
                short_wait.sleep()
                force_values = dobot.get_force()
                if not force_values:
                    continue

                # print("force_values: ", force_values)
                

                max_force_component = max(abs(value) for value in force_values)
                if max_force_component >= force_threshold:
                    consecutive_hits += 1
                    contact_force = max_force_component
                    if consecutive_hits >= consecutive_hits_required:
                        contact_detected = True
                        break
                else:
                    consecutive_hits = 0

            if contact_detected:
                print(f"\n检测到受力变化！力值≈{contact_force:.2f}N")
                # break
            else:
                pass
                # print("未检测到受力变化")

    except KeyboardInterrupt:
        print("\n用户中断，停止监测")

    print("力传感器监测完成\n")

        
    # 可选：返回home位置（根据需要取消注释）
    # dobot.move_to_pose(435.4503, 281.809, 348.9125, -179.789, -0.8424, 14.4524, speed=9)

    #移动到目标位置
    pose_now = dobot.get_pose()
    x_target, y_target, z_target= 450, -150, 12
    rx_target, ry_target, rz_target= pose_now[3], pose_now[4], pose_now[5]
    # dobot.move_to_pose(x_target, y_target, z_target, rx_target, ry_target, rz_target, speed=9)


