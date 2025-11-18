from dobot_api import DobotApiDashboard
import time
import numpy as np
import rospy
class SimpleApi(DobotApiDashboard):
    def __init__(self, ip, port):
        """
        :param ip: 机械臂的IP地址
        :param port: 机械臂的端口号
        """
        super().__init__(ip, port)
#运动控制相关
    def move_to_pose(self, x, y, z, rx, ry, rz, speed=15, acceleration=15):
        """
        移动到指定的位姿
        :param x: X轴位置 (mm)
        :param y: Y轴位置 (mm)
        :param z: Z轴位置 (mm)
        :param rx: Rx轴角度 (度)
        :param ry: Ry轴角度 (度)
        :param rz: Rz轴角度 (度)
        :param speed: 运动速度比例 (默认5)
        :param acceleration: 运动加速度比例 (默认5)
        """
        return self.MovL(x, y, z, rx, ry, rz, coordinateMode=0, v=speed, a=acceleration)

    def move_to_joint(self, j1, j2, j3, j4, j5, j6, speed=5, acceleration=5):
        """
        移动到指定的关节位置
        :param j1: J1轴角度 (度)
        :param j2: J2轴角度 (度)
        :param j3: J3轴角度 (度)
        :param j4: J4轴角度 (度)
        :param j5: J5轴角度 (度)
        :param j6: J6轴角度 (度)
        :param speed: 运动速度比例 
        :param acceleration: 运动加速度比例 
        """
        return self.MovJ(j1, j2, j3, j4, j5, j6, coordinateMode=1, v=speed, a=acceleration)
    
    def move_line(self, x, y, z, rx, ry, rz, speed=20, acceleration=20):
        """
        直线运动
        :param x: X轴位置 (mm)
        :param y: Y轴位置 (mm)
        :param z: Z轴位置 (mm)
        :param rx: Rx轴角度 (度)
        :param ry: Ry轴角度 (度)
        :param rz: Rz轴角度 (度)
        """
        return self.MovL(x, y, z, rx, ry, rz, coordinateMode=0, v=speed, a=acceleration)

    def move_relative(self, offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, speed=5, acceleration=5):
        """
        相对当前位置进行移动
        :param offset_x: X轴偏移量 (mm)正数向前，负数向后
        :param offset_y: Y轴偏移量 (mm)正数向右，负数向左
        :param offset_z: Z轴偏移量 (mm)正数向下，负数向上
        :param offset_rx: Rx轴偏移量 (度)
        :param offset_ry: Ry轴偏移量 (度)
        :param offset_rz: Rz轴偏移量 (度)
        :param speed: 运动速度比例 (默认5)
        :param acceleration: 运动加速度比例 (默认5)
        """
        return self.RelMovLTool(offset_x, offset_y, offset_z, offset_rx, offset_ry, offset_rz, v=speed, a=acceleration)
    
    def replay_trajectory(self, trajectory, speed=15, acceleration=15):
        """
        复现轨迹
        :param trajectory: 轨迹点列表，每个点是一个包含6个元素的列表，表示末端位姿 [x, y, z, rx, ry, rz]
        :param speed: 运动速度比例 (默认15)
        :param acceleration: 运动加速度比例 (默认15)
        """
        for point in trajectory:
            x, y, z, rx, ry, rz = point
            self.move_to_pose(x, y, z, rx, ry, rz, speed=speed, acceleration=acceleration)
            # self.check_pose(x, y, z)
    
    def move_to_pose_servo(self, x, y, z, rx, ry, rz, 
                            t, 
                            aheadtime, 
                            gain,
                            tolerance, 
                            max_duration,
                            verbose=False):
        """
        使用 ServoP 实时伺服控制移动到目标位姿
        
        适用场景：
        - 目标点固定不变
        - 短距离移动（< 50mm）
        - 需要实时响应外部传感器反馈
        
        :param x: X轴位置 (mm)
        :param y: Y轴位置 (mm)
        :param z: Z轴位置 (mm)
        :param rx: Rx轴角度 (度)
        :param ry: Ry轴角度 (度)
        :param rz: Rz轴角度 (度)
        :param t: ServoP控制周期 (秒)，建议范围 [0.02, 0.1]，对应 50Hz~10Hz
        :param aheadtime: 类似PID的D项，范围 [20.0, 100.0]，默认50.0
        :param gain: 类似PID的P项，范围 [200.0, 1000.0]，默认500.0
        :param tolerance: 到达判定阈值 (mm)，默认1.0mm
        :param max_duration: 最大运动时间 (秒)，防止死循环
        :param verbose: 是否打印详细信息
        :return: 是否成功到达目标位置
    
        """
        start_time = time.time()
        target_pos = np.array([x, y, z])
        rate = rospy.Rate(1/t)
        
        # if verbose:
        #     print(f"[ServoP] 开始移动到目标: [{x:.1f}, {y:.1f}, {z:.1f}]")
        
        while True:
            elapsed = time.time() - start_time
            
            # 超时检查
            if elapsed > max_duration:
                if verbose:
                    print(f"[ServoP警告] 运动超时({max_duration}秒)")
                return False
            
            # 发送ServoP指令（持续发送固定目标点）
            self.ServoP(x, y, z, rx, ry, rz, t=t, aheadtime=aheadtime, gain=gain)
            
            # 获取当前位置
            try:
                current_pose = self.get_pose()
                print("current_pose: ", current_pose)
                current_pos = np.array(current_pose[:3])
            except Exception as e:
                if verbose:
                    print(f"[ServoP警告] 无法获取当前位置: {e}")
                time.sleep(t)
                continue
            
            # 计算距离
            distance = np.linalg.norm(current_pos - target_pos)
            print("target_pos: ", target_pos)
            print("distance: ", distance)
            
            # # 定期打印进度
            # if verbose and int(elapsed * 10) % 10 == 0:  # 每秒打印一次
            #     print(f"[ServoP] 距离目标: {distance:.2f}mm, 已用时: {elapsed:.1f}s")
            
            # 到达判定
            if distance < tolerance:
                if verbose:
                    print(f"[ServoP] 到达目标，最终误差: {distance:.2f}mm")
                # time.sleep(0.1)  # 稳定一下
                return True
            
            # 控制发送频率
            # time.sleep(t)
            rate.sleep() #和time.sleep(t)不一样

    def move_to_pose_servo_smooth(self, x, y, z, rx, ry, rz, 
                                    t=0.05, 
                                    aheadtime=50.0, 
                                    gain=500.0,
                                    max_speed=50.0,
                                    tolerance=1.0, 
                                    max_duration=30.0,
                                    verbose=False):
        """
        使用 ServoP 进行平滑运动（带轨迹插值版本）- 推荐使用
        
        特点：
        - 自动在起点和终点之间生成平滑轨迹
        - 使用S曲线插值，无加速度突变
        - 速度可控，运动更平滑
        
        适用场景：
        - 中长距离移动（> 50mm）
        - 需要平滑无顿挫的运动
        - 替代MovL实现更好的运动品质
        
        :param x: X轴目标位置 (mm)
        :param y: Y轴目标位置 (mm)
        :param z: Z轴目标位置 (mm)
        :param rx: Rx轴目标角度 (度)
        :param ry: Ry轴目标角度 (度)
        :param rz: Rz轴目标角度 (度)
        :param t: ServoP控制周期 (秒)，建议 [0.02, 0.1]
        :param aheadtime: 类似PID的D项，范围 [20.0, 100.0]
        :param gain: 类似PID的P项，范围 [200.0, 1000.0]
        :param max_speed: 最大运动速度 (mm/s)，用于计算运动时间
        :param tolerance: 到达判定阈值 (mm)
        :param max_duration: 最大运动时间 (秒)
        :param verbose: 是否打印详细信息
        :return: 是否成功到达目标位置
        
        示例：
            # 替代MovL进行平滑运动
            success = dobot.move_to_pose_servo_smooth(
                300, 0, 200, 0, 0, 0, 
                max_speed=30.0,  # 控制速度
                verbose=True
            )
        """
        # 获取起始位置和姿态
        try:
            start_pose = self.get_pose()
            start_pos = np.array(start_pose[:3])
            start_rot = np.array(start_pose[3:6])
        except Exception as e:
            if verbose:
                print(f"[ServoP平滑错误] 无法获取当前位置: {e}")
            return False
        
        # 目标位置和姿态
        target_pos = np.array([x, y, z])
        target_rot = np.array([rx, ry, rz])
        
        # 计算总距离和预计时间
        distance = np.linalg.norm(target_pos - start_pos)
        
        # 如果距离太小，直接返回
        if distance < tolerance:
            if verbose:
                print(f"[ServoP平滑] 已在目标位置，无需移动")
            return True
        
        # 计算运动时间（距离 / 速度）
        duration = distance / max_speed
        # 最小运动时间0.5秒，避免太快
        duration = max(duration, 0.5)
        
        if verbose:
            print(f"[ServoP平滑] 起点: [{start_pos[0]:.1f}, {start_pos[1]:.1f}, {start_pos[2]:.1f}]")
            print(f"[ServoP平滑] 终点: [{x:.1f}, {y:.1f}, {z:.1f}]")
            print(f"[ServoP平滑] 距离: {distance:.1f}mm, 预计时间: {duration:.2f}s, 速度: {max_speed:.1f}mm/s")
        
        start_time = time.time()
        last_print_time = start_time
        
        while True:
            elapsed = time.time() - start_time
            
            # 超时检查
            if elapsed > max_duration:
                if verbose:
                    print(f"[ServoP平滑警告] 运动超时({max_duration}秒)")
                return False
            
            # 计算插值系数（0 → 1）
            alpha = min(elapsed / duration, 1.0)
            
            # S曲线插值（平滑加减速）
            # 公式: f(t) = 3t^2 - 2t^3，保证速度和加速度连续
            alpha_smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            
            # 位置插值
            current_target_pos = start_pos + alpha_smooth * (target_pos - start_pos)
            
            # 姿态插值（简单线性插值，可以改进为四元数插值）
            current_target_rot = start_rot + alpha_smooth * (target_rot - start_rot)
            
            # 发送ServoP指令（目标点随时间变化）
            self.ServoP(
                current_target_pos[0], 
                current_target_pos[1], 
                current_target_pos[2],
                current_target_rot[0], 
                current_target_rot[1], 
                current_target_rot[2],
                t=t,
                aheadtime=aheadtime,
                gain=gain
            )
            
            # 获取实际位置
            try:
                actual_pose = self.get_pose()
                actual_pos = np.array(actual_pose[:3])
            except:
                time.sleep(t)
                continue
            
            # 计算到目标的距离
            distance_to_target = np.linalg.norm(actual_pos - target_pos)
            
            # 定期打印进度（每0.5秒）
            if verbose and (time.time() - last_print_time) > 0.5:
                progress = alpha * 100
                print(f"[ServoP平滑] 进度: {progress:.0f}%, 距终点: {distance_to_target:.1f}mm, 已用时: {elapsed:.1f}s")
                last_print_time = time.time()
            
            # 到达判定：时间进度达到100% 且 位置误差小于阈值
            if alpha >= 1.0 and distance_to_target < tolerance:
                if verbose:
                    print(f"[ServoP平滑] 到达目标，最终误差: {distance_to_target:.2f}mm，总用时: {elapsed:.2f}s")
                time.sleep(0.1)  # 稳定一下
                return True
            
            # 控制发送频率
            time.sleep(t)

    def get_robot_mode(self):
        MODE_MAP = {
            1:  "ROBOT_MODE_INIT 初始化状态",
            2:  "ROBOT_MODE_BRAKE_OPEN 有任意关节的抱闸松开",
            3:  "ROBOT_MODE_POWEROFF 机械臂下电状态",
            4:  "ROBOT_MODE_DISABLED 未使能（无抱闸松开）",
            5:  "ROBOT_MODE_ENABLE 使能且空闲",
            6:  "ROBOT_MODE_BACKDRIVE 拖拽模式",
            7:  "ROBOT_MODE_RUNNING 运行状态(工程，TCP队列运动等)",
            8:  "ROBOT_MODE_SINGLE_MOVE 单次运动状态（点动、RunTo等）",
            9:  "ROBOT_MODE_ERROR 有未清除的报警",
            10: "ROBOT_MODE_PAUSE 工程暂停状态",
            11: "ROBOT_MODE_COLLISION 碰撞检测触发状态"
        }

        raw_output = self.RobotMode()

        # 1. 查找花括号
        start_index = raw_output.find("{") + 1
        end_index   = raw_output.find("}")
        real_output = [int(value) for value in raw_output[start_index:end_index].split(",")]
        return real_output
        

    def positive_kinematics(self, j1, j2, j3, j4, j5, j6, user=-1, tool=-1):
        """
        正解运算：给定关节角度，计算末端位姿
        :param j1: J1轴角度 (度)
        :param j2: J2轴角度 (度)
        :param j3: J3轴角度 (度)
        :param j4: J4轴角度 (度)
        :param j5: J5轴角度 (度)
        :param j6: J6轴角度 (度)
        :param user: 用户坐标系索引 (默认-1，使用全局用户坐标系)
        :param tool: 工具坐标系索引 (默认-1，使用全局工具坐标系)
        """
        raw_output = self.PositiveKin(j1, j2, j3, j4, j5, j6, user=user, tool=tool)
        start_index = raw_output.find("{") + 1
        end_index = raw_output.find("}")

        # 提取数值字符串并分割为列表
        real_output = [float(value) for value in raw_output[start_index:end_index].split(",")]
        return real_output

    def inverse_kinematics(self, x, y, z, rx, ry, rz, user=-1, tool=-1, use_joint_near=-1, joint_near=''):
        """
        逆解运算：给定位姿，计算关节角度
        :param x: X轴位置 (mm)
        :param y: Y轴位置 (mm)
        :param z: Z轴位置 (mm)
        :param rx: Rx轴角度 (度)
        :param ry: Ry轴角度 (度)
        :param rz: Rz轴角度 (度)
        :param user: 用户坐标系索引 (默认-1，使用全局用户坐标系)
        :param tool: 工具坐标系索引 (默认-1，使用全局工具坐标系)
        :param use_joint_near: 是否使用最近关节角度 (默认-1，不使用)
        :param joint_near: 最近关节角度 (默认为空)
        """
        raw_output = self.InverseKin(x, y, z, rx, ry, rz, user=user, tool=tool, useJointNear=use_joint_near, JointNear=joint_near)

        start_index = raw_output.find("{") + 1
        end_index = raw_output.find("}")

        real_output = [float(value) for value in raw_output[start_index:end_index].split(",")]
        return real_output
    def get_angle(self):
        """
        获取当前关节角度
        """
        raw_output = self.GetAngle()

        start_index = raw_output.find("{") + 1
        end_index = raw_output.find("}")

        real_output = [float(value) for value in raw_output[start_index:end_index].split(",")]
        return real_output

    # def get_pose(self, user=-1, tool=-1):
    #     """
    #     获取当前位姿
    #     :param user: 用户坐标系索引 (默认-1，使用全局用户坐标系)
    #     :param tool: 工具坐标系索引 (默认-1，使用全局工具坐标系)
    #     """
    #     raw_output = self.GetPose(user=user, tool=tool)

    #     start_index = raw_output.find("{") + 1
    #     end_index = raw_output.find("}")

    #     real_output = [float(value) for value in raw_output[start_index:end_index].split(",")]
    #     if not real_output or len(real_output) != 6:
    #         raise ValueError("获取的位姿数据无效")
    #     return real_output
    
    def get_pose(self, user=-1, tool=-1):
        """
        获取当前位姿
        :param user: 用户坐标系索引 (默认-1，使用全局用户坐标系)
        :param tool: 工具坐标系索引 (默认-1，使用全局工具坐标系)
        """
        raw = self.GetPose(user=user, tool=tool)
        return self._safe_parse_list(raw, 6, [0.0]*6)
    
    def check_pose(self, x, y, z):
        """
        判断是否到达预定位置，主要配合夹爪使用
        """
        while True:
            # 获取当前机械臂的位置
            current_pose = self.get_pose()
            current_position = np.array(current_pose[:3]) 

            # 目标位置
            target_position = np.array([x, y, z])

            # 判断当前位置是否等于目标位置（允许一定的误差范围）
            if np.linalg.norm(current_position - target_position) < 1.0:  
                break
        time.sleep(0.2)
        return True

#机械臂状态相关   
    def get_errorid(self):
        raw_output = self.GetErrorID()

        start_index = raw_output.find("{") + 1
        end_index = raw_output.find("}")

        inner = raw_output[start_index:end_index].strip()

        if inner == "[]":
            return []

        inner = inner.replace("[", "").replace("]", "")

        try:
            real_output = [int(value.strip()) for value in inner.split(",") if value.strip()]
        except ValueError:
            return []

        return real_output
    def enable_robot(self):
        """
        使能机械臂
        """
        self.EnableRobot()

    def disable_robot(self):
        """
        下使能机械臂
        """
        self.DisableRobot()
    
    def clear_error(self):
        """
        清除机器⼈报警。清除报警后，⽤⼾可以根据RobotMode来判断机器⼈是否还处于报警状态。部
        分报警需要解决报警原因或者重启控制柜后才能清除。
        """
        self.ClearError()
    
    def transto_tcp(self):
        """
        使⽤TCP模式。

        """
        self.RequestControl()
    
    def stop(self):
        """
        解决连续绿色灯，⽽不进⼊运动模式(运动非常慢，不正常）。

        """
        self.Stop()
#夹爪相关函数   
    def setToolPower(self, status):
        """
        设置末端⼯具供电状态
        """
        self.SetToolPower(status, identify=-1)
        time.sleep(2)
    
    def setTool485(self, port, baudrate, parity):
        """
        设置末端⼯具的RS485接⼝对应的数据格式。
        """
        self.SetTool485(port, baudrate, parity)
        time.sleep(2)
    
    def modbusCreate(self,ip, port, slave_id, isRTU):
        """
        创建Modbus主站，并和从站建⽴连接。
        """
        return self.ModbusCreate(ip, port, slave_id, isRTU)
    
    def modbusClose(self, index):
        """
        和Modbus从站断开连接，释放主站。
        """
        self.ModbusClose(index)
    
    def setHoldRegs(self, index, addr, count, valTab):
        """
        将指定的值以指定的数据类型写⼊Modbus从站保持寄存器指定的地址。
        """
        return self.SetHoldRegs(index, addr, count, valTab)
    
    def getHoldRegs(self, index, addr, count):
        """
        按照指定的数据类型，读取Modbus从站保持寄存器地址的值。
        """
        return self.GetHoldRegs(index, addr, count)
#力控模式相关函数  
    def enable_ft_sensor(self, enable):
        """
        启用或禁用六维力传感器
        :param enable: 1-启用, 0-禁用
        """
        self.EnableFTSensor(enable)

    def six_force_home(self):
        """
        六维力传感器置零(以当前状态为基准)
        """
        self.SixForceHome()
    def get_force(self):
        raw = self.GetForce()
        return self._safe_parse_list(raw, 6, [0.0]*6)

    def fc_force_mode(self,x, y, z, rx, ry, rz, fx,fy,fz,frx,fry,frz):
        """
        以用户指定的配置参数开启力控。
        {x,y,z,rx,ry,rz} 
            开启/关闭笛卡尔空间某个方向的力控调节。
            0表示关闭该方向的力控。
            1表示开启该方向的力控。
        {fx,fy,fz,frx,fry,frz} 
            目标力：是工具末端与作用对象之间接触力的目标值，是一种模拟力，可以由用户自行设定；目标力方向分别对应笛卡尔空间的{x,y,z,rx,ry,rz}方向。
            位移方向的目标力范围[-200,200]，单位N；姿态方向的目标力范围[-12,12]，单位N/m。
        """
        return self.FCForceMode(x, y, z, rx, ry, rz, fx,fy,fz,frx,fry,frz)

    def fc_set_deviation(self, dx, dy, dz, drx, dry, drz, controltype=1):
        """
        设置力控偏差允许范围
        :param dx: X轴允许偏差 (mm)
        :param dy: Y轴允许偏差 (mm)
        :param dz: Z轴允许偏差 (mm)
        :param drx: Rx轴允许偏差 (度)
        :param dry: Ry轴允许偏差 (度)
        :param drz: Rz轴允许偏差 (度)
        :param controltype: 控制类型 (1-位置控制)
        """
        return self.FCSetDeviation(dx, dy, dz, drx, dry, drz, controltype)
    def fc_set_drive_mode(self,  x, y, z, rx, ry, rz):
        """
        指定可拖拽的方向并进入力控拖拽模式。
        {x,y,z,rx,ry,rz} string
        用于指定可拖拽的方向。
        0代表该方向不能拖拽，1代表该方向可以拖拽。
        例：
        {1,1,1,1,1,1}表示机械臂可在各轴方向上自由拖动
        {1,1,1,0,0,0}表示机械臂仅可在XYZ轴方向上拖动
        {0,0,0,1,1,1}表示机械臂仅可在RxRyRz轴方向上旋转
        """
        self.ForceDriveMode( x, y, z, rx, ry, rz)

    def fc_set_force_limit(self, fx, fy, fz, mx, my, mz):
        """
        设置力控最大力限制
        :param fx: X轴最大力 (N)
        :param fy: Y轴最大力 (N)
        :param fz: Z轴最大力 (N)
        :param mx: X轴最大力矩 (Nm)
        :param my: Y轴最大力矩 (Nm)
        :param mz: Z轴最大力矩 (Nm)
        """
        return self.FCSetForceLimit(fx, fy, fz, mx, my, mz)

    def fc_set_stiffness(self, kx, ky, kz, krx, kry, krz):
        """
        设置力控刚度系数
        :param kx: X轴刚度系数
        :param ky: Y轴刚度系数
        :param kz: Z轴刚度系数
        :param krx: Rx轴刚度系数
        :param kry: Ry轴刚度系数
        :param krz: Rz轴刚度系数
        """
        return self.FCSetStiffness(kx, ky, kz, krx, kry, krz)

    def fc_set_damping(self, dx, dy, dz, drx, dry, drz):
        """
        设置力控阻尼系数
        :param dx: X轴阻尼系数
        :param dy: Y轴阻尼系数
        :param dz: Z轴阻尼系数
        :param drx: Rx轴阻尼系数
        :param dry: Ry轴阻尼系数
        :param drz: Rz轴阻尼系数
        """
        return self.FCSetDamping(dx, dy, dz, drx, dry, drz)

    def fc_off(self):
        """
        关闭力控模式
        """
        self.FCOff()

    def _safe_parse_list(self, raw_output, expected_len=None, default=None):
        """
        把 {a,b,c,...} 形式的字符串解析成 list[float]。
        空串 / 格式错误时返回 default。
        """
        if not raw_output:
            return default
        start = raw_output.find("{") + 1
        end   = raw_output.find("}")
        if start <= 0 or end <= 0:
            return default
        parts = raw_output[start:end].split(",")
        try:
            values = [float(p.strip()) for p in parts if p.strip() != ""]
            if expected_len and len(values) != expected_len:
                return default
            return values
        except ValueError:
            return default
    
    def __del__(self):
        """
        析构函数：对象销毁时自动断开机械臂连接
        """
        try:
            self.close()        
        except Exception:
            pass  


import threading
import time

class ForceMonitor:
    def __init__(self, dobot):
        """
        初始化 ForceMonitor 类。
        """
        self.robot = dobot
        self.stop_event = threading.Event()
        self.force_thread = None 
        self.force_exceeded = threading.Event() 


    def monitor_force(self):
        FORCE_THRESHOLD = 10.0
        CHECK_INTERVAL = 0.1
        while not self.stop_event.is_set():
            try:
                raw = self.robot.get_force()
                if not raw:                # 空或 None
                    time.sleep(CHECK_INTERVAL)
                    continue
                force = [float(t) if t else 0.0 for t in raw]
                current_force_2 = abs(force[2])
                if current_force_2 > FORCE_THRESHOLD:
                    print(f"力超过阈值 ({FORCE_THRESHOLD} N)，机械臂停止...")
                    self.robot.disable_robot()
                    self.force_exceeded.set()
                    break
            except Exception as e:
                print(f"[ForceMonitor] 解析力数据失败: {e}")
            time.sleep(CHECK_INTERVAL)

    def start_monitoring(self):
        """
        启动力监测线程。
        """
        self.stop_event.clear()
        self.force_thread = threading.Thread(target=self.monitor_force)
        self.force_thread.start()
        print("力监测线程已启动。")

    def stop_monitoring(self):
        """
        停止力监测线程。
        """
        self.stop_event.set()
        self.force_thread.join()
        print("力监测线程已停止。")

# 主程序
if __name__ == "__main__":
    try:
        # 初始化 ForceMonitor
        dobot = SimpleApi("192.168.5.1", 29999)
        dobot.clear_error()
        dobot.enable_robot()
        dobot.stop()
        dobot.enable_ft_sensor(1)
        time.sleep(1)
        dobot.six_force_home()
        time.sleep(1)
        force_monitor = ForceMonitor(dobot)
        force_monitor.start_monitoring()

        # 主线程可以执行其他任务，或者等待用户输入来停止程序
        input("按回车键停止程序...\n")

    except Exception as e:
        print(f"程序异常: {str(e)}")
    finally:
        # 停止力监测线程
        force_monitor.stop_monitoring()
        print("程序结束")


class WorkspaceMonitor:
    def __init__(self, dobot):
        self.robot = dobot
        self.stop_event = threading.Event()
        self.monitor_thread = None
        self.out_of_range_event = threading.Event()
        self.flag = True

        # 工作空间限制
        self.limits = {
            'x': (400, 750),
            'y': (-200, 450),
            'z': (-250, 400)
        }

    def is_in_workspace(self, pose):
        x, y, z = pose[0], pose[1], pose[2]
        return (
            self.limits['x'][0] <= x <= self.limits['x'][1] and
            self.limits['y'][0] <= y <= self.limits['y'][1] and
            self.limits['z'][0] <= z <= self.limits['z'][1]
        )

    def monitor_workspace(self):
        CHECK_INTERVAL = 0.1
        while not self.stop_event.is_set():
            try:
                pose = self.robot.get_pose()
                if not pose:               # 空或 None
                    time.sleep(CHECK_INTERVAL)
                    continue
                if not self.is_in_workspace(pose):
                    print(f"[WorkspaceMonitor] 超出安全区: {pose}")
                    self.robot.disable_robot()
                    self.out_of_range_event.set()
                    break
            except Exception as e:
                print(f"[WorkspaceMonitor] 获取位姿失败: {e}")
            time.sleep(CHECK_INTERVAL)

    def start_monitoring(self):
        self.stop_event.clear()
        self.out_of_range_event.clear()
        self.monitor_thread = threading.Thread(target=self.monitor_workspace)
        self.monitor_thread.start()
        print("工作空间监控线程已启动。")

    def stop_monitoring(self):
        self.stop_event.set()
        self.monitor_thread.join()
        print("工作空间监控线程已停止。")


# ========== 主程序测试 ==========
if __name__ == "__main__":
    try:
        dobot = SimpleApi("192.168.5.1", 29999)
        dobot.clear_error()
        dobot.enable_robot()
        dobot.stop()

        pose_monitor = WorkspaceMonitor(dobot)
        pose_monitor.start_monitoring()

        # 主线程继续运行其他任务（或等待用户干预）
        input("按回车键停止程序...\n")

    except Exception as e:
        print(f"程序异常: {e}")
    finally:
        pose_monitor.stop_monitoring()
        print("程序结束")



class ErrorMonitor:
    def __init__(self, dobot):
        self.robot = dobot
        self.stop_event = threading.Event()
        self.monitor_thread = None
        self.error_detected = threading.Event()

    def monitor_error(self):
        CHECK_INTERVAL = 0.5
        while not self.stop_event.is_set():
            try:
                errors = self.robot.get_errorid()
                mode = self.robot.get_robot_mode()
                if errors is None or mode is None:
                    time.sleep(CHECK_INTERVAL)
                    continue
                if (errors and len(errors)) or mode[0] not in [5, 6, 7, 8]:
                    print(f"[ErrorMonitor] 错误码: {errors}  模式: {mode}")
                    self.error_detected.set()
                    break
            except Exception as e:
                print(f"[ErrorMonitor] 获取错误码失败: {e}")
            time.sleep(CHECK_INTERVAL)

    def start_monitoring(self):
        self.stop_event.clear()
        self.error_detected.clear()
        self.monitor_thread = threading.Thread(target=self.monitor_error)
        self.monitor_thread.start()
        print("错误码监控线程已启动。")

    def stop_monitoring(self):
        self.stop_event.set()
        self.monitor_thread.join()
        print("错误码监控线程已停止。")


# ========== 主程序测试 ==========
if __name__ == "__main__":
    try:
        dobot = SimpleApi("192.168.5.1", 29999)
        dobot.clear_error()
        dobot.enable_robot()
        dobot.stop()

        error_monitor = ErrorMonitor(dobot)
        error_monitor.start_monitoring()

        # 模拟主线程运行其他任务
        while not error_monitor.error_detected.is_set():
            time.sleep(0.1)

        print("检测到错误，程序结束。")

    except Exception as e:
        print(f"程序异常: {e}")
    finally:
        error_monitor.stop_monitoring()
        print("程序结束")