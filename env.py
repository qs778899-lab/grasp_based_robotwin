"""
Author: zhangcongshe

Date: 2025/11/1

Version: 1.0
"""


import time
import json
from spatialmath import SE3, SO3
import numpy as np



from simple_api import SimpleApi
from dobot_gripper import DobotGripper
from create_camera import CreateRealsense




class create_env:
    def __init__(self,config_pth):
        with open(config_pth, 'r') as file:
            self.config = json.load(file)

        self.robot1, self.gripper = self.init_robot1()
        self.camera1 = self.init_camera()



    def init_camera1(self,):
        self.camera1 = CreateRealsense(self.config['camera_id'])
        self.camera1_rt = SE3.Rt(np.array(self.config['camera_1']['T_ee_cam']['rotation_matrix']),
                                 self.config['camera_1']['T_ee_cam']['translation_vector'], 
                                 check=False)
        self.camera1_main = {
            "cam": self.camera1,
            "Rt": self.camera1_rt,
            "cam_k": self.config['camera_1']['cam_k']
        }
        return self.camera1_main
                    

    def init_robot1(self):
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
        robot_main = {
            "robot": dobot,
            "tcp2ee": self.config['robot1']['tcp2ee']
        }
        return robot_main,gripper

