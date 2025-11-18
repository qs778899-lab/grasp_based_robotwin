import time
from simple_api import SimpleApi

class DobotGripper:
    # 夹爪寄存器地址定义
    REGISTER_INIT = 256          # 初始化寄存器
    REGISTER_FORCE = 257         # 力值设置寄存器
    REGISTER_POSITION = 259      # 位置设置寄存器
    REGISTER_SPEED = 260         # 状态读取寄存器

    CURRENT_POSITION = 514       # 当前位置
    CURRENT_FORCE = 513          # 当前力值
    CURRENT_SPEED = 512          # 当前速度
    CURRENT_INIT_STATE = 512     # 当前初始化状态
    CURRENT_GRIP_STATE = 513     # 当前夹爪状态
    CURRENT_TARGET_POSITION = 259  # 当前目标位置
    CURRENT_TARGET_FORCE = 257    # 当前目标力值

    def __init__(self, dobot):
        self.api = dobot
        self.modbus_index = None  # Modbus主站索引
        self.is_connected = False

    def connect(self, init=False, init_position=30):
        try:
            if self.is_connected:
                print("已经连接，无需重复连接")
                return True
            # 1. 开启末端工具电源
            self.api.setToolPower(1)
            time.sleep(0.5)
            self.api.setTool485(115200, "N", 1)

            # 2. 创建Modbus主站
            result = self.api.modbusCreate("192.168.5.1", 60000, 1, 1)

            # 3. 解析返回值（格式示例："0,{3},ModbusRTUCreate(5,115200,N);"）
            if isinstance(result, str):
                parts = result.split(',')
                if len(parts) >= 1:
                    error_code = int(parts[0])  # 错误码（0 表示成功）
                    if error_code != 0:
                        print(f"Modbus主站创建失败，错误码: {error_code}")
                        return False

                    if len(parts) > 1 and parts[1].startswith('{') and parts[1].endswith('}'):
                        self.modbus_index = int(parts[1][1:-1])  # 去掉大括号
                        print(f"Modbus主站索引解析成功: {self.modbus_index}")
                    else:
                        print("无法解析Modbus主站索引")
                        return False
            else:
                print("ModbusRTUCreate 返回值格式不符合预期")
                return False

            # 4. 初始化夹爪
            if init:
                self._initialize_gripper()
                self.set_position(init_position)
                self.control(position=init_position, force=80, speed=50)
                time.sleep(1.0)

            self.is_connected = True
            return True

        except Exception as e:
            print(f"[ERROR] 连接失败: {e}")
            return False

    def disconnect(self):
        if self.modbus_index is None:
            print("Modbus 连接已关闭，无需重复关闭")
            return
        try:
            print(f"尝试关闭 Modbus 连接，索引: {self.modbus_index}")
            result = self.api.modbusClose(self.modbus_index)
        except Exception as e:
            print(f"Error while closing Modbus connection: {e}")
        finally:
            self.modbus_index = None
            self.is_connected = False
            print("机械臂连接已断开")

    def _initialize_gripper(self):
        """夹爪初始化"""
        # 写入初始化寄存器
        result = self.api.setHoldRegs(self.modbus_index, self.REGISTER_INIT, 1, str({1}))
        if result[0] != '0':
            print(f"initialize_gripper get error: {result}")
        time.sleep(0.5)

    def set_force(self, force_percent):
        """
        设置夹爪力值

        Args:
            force_percent (int): 力值百分比 (0-100)
        """
        if not self.is_connected:
            print("请先连接机械臂")
            return False
        if not (0 <= force_percent <= 100):
            print("力值百分比必须在0-100之间")
            return False
        # 写入力值寄存器
        result = self.api.setHoldRegs(
            index=self.modbus_index, 
            addr=self.REGISTER_FORCE, 
            count=1, 
            valTab=str({force_percent})
        )
        if result[0] != '0':
            print(f"set_force get error: {result}")

    def set_speed(self, speed_percent):
        """
        设置夹爪速度

        Args:
            speed_percent (int): 速度百分比 (0-100)
        """
        if not self.is_connected:
            print("请先连接机械臂")
            return False
        if not (0 <= speed_percent <= 100):
            print("速度百分比必须在0-100之间")
            return False
        # 写入速度寄存器
        result = self.api.setHoldRegs(
            index=self.modbus_index, 
            addr=self.REGISTER_SPEED, 
            count=1, 
            valTab=str({speed_percent})
        )
        if result[0] != '0':
            print(f"set_speed get error: {result}")

    def set_position(self, position):
        """
        设置夹爪位置

        Args:
            position (int): 位置值 (0-1000, 0为完全闭合，1000为完全张开)
        """
        if not self.is_connected:
            print("请先连接机械臂")
            return False
        if not (0 <= position <= 1000):
            print("位置值必须在0-1000之间")
            return False
        # 写入位置寄存器
        result = self.api.setHoldRegs(
            index=self.modbus_index, 
            addr=self.REGISTER_POSITION, 
            count=1, 
            valTab=str({position})
        )
        if result[0] != '0':
            print(f"set_position get error: {result}")

    def control(self, position, force, speed):
        """
        控制夹爪
        """
        if not self.is_connected:
            print("请先连接机械臂")
            return False
        if not (0 <= speed <= 100):
            print("速度百分比必须在0-100之间")
            return False
        if not (0 <= force <= 100):
            print("力值百分比必须在0-100之间")
            return False
        if not (0 <= position <= 1000):
            print("位置值必须在0-1000之间")
            return False
        self.set_force(int(force))
        self.set_speed(int(speed))
        self.set_position(int(position))

    def read_target_position(self):
        """
        读取夹爪目标位置
        """
        if not self.is_connected:
            print("请先连接机械臂")
            return None
        # 读取目标位置寄存器
        raw_output = self.api.getHoldRegs(
            index=self.modbus_index, 
            addr=self.CURRENT_TARGET_POSITION, 
            count=1
        )
        start_index = raw_output.find("{") + 1
        end_index = raw_output.find("}")
        real_output = [float(value) for value in raw_output[start_index:end_index].split(",")]
        return real_output

    def read_current_position(self):
        """
        读取夹爪当前位置
        """
        if not self.is_connected:
            print("请先连接机械臂")
            return None
        # 读取当前位置寄存器
        raw_output = self.api.getHoldRegs(
            index=self.modbus_index, 
            addr=self.CURRENT_POSITION, 
            count=1
        )
        start_index = raw_output.find("{") + 1
        end_index = raw_output.find("}")
        real_output = [float(value) for value in raw_output[start_index:end_index].split(",")]
        return real_output

    def read_target_force(self):
        """
        读取夹爪目标力值
        """
        if not self.is_connected:
            print("请先连接机械臂")
            return None
        # 读取目标力值寄存器
        raw_output = self.api.getHoldRegs(
            index=self.modbus_index, 
            addr=self.CURRENT_TARGET_FORCE, 
            count=1
        )
        start_index = raw_output.find("{") + 1
        end_index = raw_output.find("}")
        real_output = [float(value) for value in raw_output[start_index:end_index].split(",")]
        return real_output

    def read_target_speed(self):
        """
        读取夹爪目标速度
        """
        if not self.is_connected:
            print("请先连接机械臂")
            return None
        # 读取目标速度寄存器
        raw_output = self.api.getHoldRegs(
            index=self.modbus_index, 
            addr=self.REGISTER_SPEED, 
            count=1 
        )
        start_index = raw_output.find("{") + 1
        end_index = raw_output.find("}")
        real_output = [float(value) for value in raw_output[start_index:end_index].split(",")]
        return real_output

    def read_init_state(self):
        """
        读取夹爪初始化状态
        """
        if not self.is_connected:
            print("请先连接机械臂")
            return None
        # 读取初始化状态寄存器
        raw_output = self.api.getHoldRegs(
            index=self.modbus_index, 
            addr=self.CURRENT_INIT_STATE, 
            count=1
        )
        start_index = raw_output.find("{") + 1
        end_index = raw_output.find("}")
        real_output = [float(value) for value in raw_output[start_index:end_index].split(",")]
        return real_output

    def read_grip_state(self):
        """
        读取夹爪抓取状态
        """
        if not self.is_connected:
            print("请先连接机械臂")
            return None
        # 读取抓取状态寄存器
        raw_output = self.api.getHoldRegs(
            index=self.modbus_index, 
            addr=self.CURRENT_GRIP_STATE, 
            count=1
        )
        start_index = raw_output.find("{") + 1
        end_index = raw_output.find("}")
        real_output = [float(value) for value in raw_output[start_index:end_index].split(",")]
        return real_output

    def get_all_states(self):
        """
        读取所有夹爪状态信息
        Returns:
            dict: 包含所有状态信息的字典
        """
        if not self.is_connected:
            print("请先连接机械臂")
            return None
        states = {
            "init_state": self.read_init_state(),
            "grip_state": self.read_grip_state(),
            "current_position": self.read_current_position(),
            "target_position": self.read_target_position(),
            "target_force": self.read_target_force(),
            "target_speed": self.read_target_speed()
        }
        return states

    def pause_movement(self, force=10, speed=10):
        # Get current position from state
        current_pos = self.get_all_states()['current_position']
        print(f"Pausing movement at position: {current_pos}")
        # Send command to maintain current position
        # Using lower speed and force for safety
        self.control(
            position=current_pos[0],
            force=force,  # Lower force to prevent excessive pressure
            speed=speed   # Lower speed for smoother stop
        )
        return current_pos

    def __del__(self):
        """析构函数，确保对象销毁时断开连接"""
        self.disconnect()

    def __enter__(self):
        if not self.connect(init=True):
            raise Exception("无法连接到机械臂")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

# 使用示例
if __name__ == "__main__":
    # 创建DobotGripper实例
    dobot = SimpleApi("192.168.5.1", 29999) 
    dobot.clear_error()
    dobot.enable_robot()
    dobot.stop()

    with DobotGripper(dobot) as gripper:
        # 连接并初始化夹爪
        if gripper.connect(init=True):
            # 控制夹爪
            states = gripper.get_all_states()
            print("夹爪状态:", states)
            gripper.control(position=100, force=100, speed=50)
            time.sleep(1)
            states = gripper.get_all_states()
            print("夹爪状态:", states)
            gripper.control(position=300, force=100, speed=50)
            time.sleep(1)
            states = gripper.get_all_states()
            print("夹爪状态:", states)
            gripper.control(position=600, force=100, speed=50)
            time.sleep(1)
            states = gripper.get_all_states()
            print("夹爪状态:", states)
            gripper.control(position=900, force=100, speed=50)
            time.sleep(1)
            states = gripper.get_all_states()
            print("夹爪状态:", states)