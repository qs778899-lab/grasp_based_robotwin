# 基于物体位姿的抓取系统使用说明

## 项目结构

```
labgrasp/
├── hand_eye_calibration.json          # 手眼标定配置文件
├── grasp_utils.py                     # 通用工具函数
├── calculate_grasp_pose_from_object_pose.py  # 核心抓取算法
├── grasp_main.py                      # 主程序（示例）
├── mesh/                              # 物体网格模型
└── FoundationPose/                    # FoundationPose模块
```

## 核心模块说明

### 1. hand_eye_calibration.json - 手眼标定配置

存储相机到末端执行器的变换矩阵：

```json
{
  "T_ee_cam": {
    "description": "从相机坐标系到末端执行器坐标系的变换矩阵",
    "rotation_matrix": [[...], [...], [...]],
    "translation_vector": [x, y, z]
  }
}
```

### 2. grasp_utils.py - 工具函数库

#### extract_euler_zyx()

从旋转矩阵提取ZYX欧拉角：

```python
rx, ry, rz = extract_euler_zyx(rotation_matrix)
```

#### print_pose_info()

打印变换矩阵的位姿信息：

```python
print_pose_info(T_matrix, description="物体姿态")
```

#### normalize_angle()

将角度规范化到[-180, 180]范围。

### 3. calculate_grasp_pose_from_object_pose.py - 核心抓取算法

## 核心函数概述

`execute_grasp_from_object_pose()` 是一个封装好的抓取执行函数，可以从任何物体姿态估计模型（FoundationPose、其他6D Pose模型等）获取的物体位姿计算并执行机械臂抓取。

## 函数

```python
def execute_grasp_from_object_pose(
        center_pose_array: 物体中心在相机坐标系中的位姿 (4x4 numpy array)
        dobot: Dobot机械臂对象
        gripper: 夹爪对象
        T_ee_cam: 相机到末端执行器的变换矩阵 (SE3对象)
        z_xoy_angle: 物体绕z轴旋转角度，用于调整抓取接近方向 (度)
        vertical_euler: 垂直向下抓取的grasp姿态的的欧拉角 [rx, ry, rz] (度)，默认[-180, 0, -90]
        grasp_tilt_angle: 倾斜抓取角度 (度)，叠加在vertical_euler[0]上, 由垂直向下抓取旋转为斜着向下抓取的grasp姿态的旋转角度.
        angle_threshold: z轴对齐的角度阈值 (度)
        T_tcp_ee_z: TCP到末端执行器的z轴偏移 (米)
        T_safe_distance: 安全距离，防止抓取时与物体碰撞 (米)
        z_safe_distance: 最终移动时z方向的额外安全距离,也是为了抓取物体靠上的部分。 (毫米)
        verbose: 是否打印详细信息
):
    """
    Returns:
        success (bool): 是否成功执行抓取
        T_base_ee_ideal (SE3): 计算得到的理想末端执行器位姿
    """
```

## 核心功能

该函数自动完成以下步骤：

1. **坐标系转换**：将物体在相机坐标系中的位姿转换到机器人基坐标系
2. **物体姿态调整**：
   - 将物体Z轴对齐到垂直向上方向
   - 将物体X/Y轴对齐到基坐标系
   - 绕Z轴旋转指定角度以优化抓取接近方向
3. **抓取姿态计算**：根据配置的欧拉角参数计算理想抓取姿态
4. **执行抓取**：控制机械臂移动并执行抓取动作

## 坐标系说明

- **相机坐标系 cam**: 物体姿态估计模型的输出坐标系
- **末端执行器坐标系 ee**: 机械臂末端法兰坐标系
- **TCP坐标系 tcp/grasp**: 工具中心点坐标系（夹爪中心）
- **机器人基坐标系**: 机器人底座坐标系

## 参数的物理含义说明

### 必需参数

- **center_pose_array** (numpy.ndarray, 4x4):
  - 物体中心在相机坐标系中的位姿矩阵
  - 格式：齐次变换矩阵
  - 单位：米（平移部分）

- **dobot** (SimpleApi):
  - Dobot机械臂对象

- **gripper** (DobotGripper):
  - 夹爪控制对象

- **T_ee_cam** (SE3):
  - 相机到末端执行器的手眼标定变换矩阵

### 姿态调整参数

- **z_xoy_angle** (float, 默认30):
  - 物体绕Z轴旋转的角度（度）
  - 用于优化抓取接近方向


- **vertical_euler** (list, 默认[-180, 0, -90]):
  - 垂直抓取姿态的欧拉角 [rx, ry, rz]（度）
  - 定义了"垂直向下"的基准姿态

- **grasp_tilt_angle** (float, 默认55):
  - 在vertical_euler基础上的倾斜角度（度）， 叠加在vertical_euler[0]上
  - 由垂直向下抓取旋转为斜着向下抓取的grasp姿态的旋转角度，用于从斜向角度接近物体

- **angle_threshold** (float, 默认10.0):
  - Z轴对齐的角度阈值（度）
  - 当物体Z轴与垂直方向偏差超过此值时进行对齐

### 几何参数

- **T_tcp_ee_z** (float, 默认-0.17):
  - TCP到末端执行器的Z轴偏移（米）
  - 负值表示TCP在末端执行器下方

- **T_safe_distance** (float, 默认0.003):
  - 沿夹爪轴向的安全距离，有利于抓取物体靠上的部分，也防止抓取时与物体碰撞 （米）
 

- **z_safe_distance** (float, 默认30):
  - z方向的一个安全距离，防止抓取时与物体碰撞，有利于抓取物体靠上的部分（毫米）

### 运动参数（硬编码于函数内部）

- **move_speed** (float, 默认15):
  - 机械臂移动速度

- **gripper_open_pos** (int, 默认1000):
  - 夹爪张开位置

- **gripper_close_pos** (int, 默认80):
  - 夹爪闭合位置

- **gripper_force** (int, 默认30):
  - 夹爪力度

- **gripper_speed** (int, 默认30):
  - 夹爪速度

### 其他参数

- **verbose** (bool, 默认True):
  - 是否打印详细的调试信息
  - True: 打印所有中间计算结果
  - False: 仅打印关键信息

## 返回值

函数返回一个元组 `(success, T_base_ee_ideal)`:

- **success** (bool): 
  - True: 抓取执行成功
  - False: 抓取执行失败（未能到达目标位置）

- **T_base_ee_ideal** (SE3):
  - 计算得到的理想末端执行器位姿
  - 可用于后续分析或调试

### 4. grasp_main.py - 主程序示例

包含完整的抓取流程示例，集成了：
- FoundationPose物体姿态估计
- 手眼标定加载
- 机械臂初始化
- 实时抓取执行

## 使用示例

### 示例1：使用FoundationPose

```python
import sys
sys.path.append("FoundationPose")

from estimater import *
from create_camera import CreateRealsense
from simple_api import SimpleApi
from dobot_gripper import DobotGripper
from spatialmath import SE3, SO3
import numpy as np
from calculate_grasp_pose_from_object_pose import execute_grasp_from_object_pose

# 加载手眼标定
def load_hand_eye_calibration(json_path="hand_eye_calibration.json"):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    calibration = data['T_ee_cam']
    rotation_matrix = np.array(calibration['rotation_matrix'])
    translation_vector = calibration['translation_vector']
    return SE3.Rt(rotation_matrix, translation_vector, check=False)

T_ee_cam = load_hand_eye_calibration()

# 初始化机器人和相机
dobot, gripper = init_robot()
camera = CreateRealsense("231522072272")

# 使用FoundationPose获取物体位姿
# ... (FoundationPose初始化代码)
color = camera.get_frames()['color']
depth = camera.get_frames()['depth'] / 1000
mask = get_mask_from_GD(color, "red cylinder")
pose = est.register(K=cam_k, rgb=color, depth=depth, ob_mask=mask, iteration=50)
center_pose_array = np.array(pose @ np.linalg.inv(to_origin), dtype=float)

# 执行抓取
success, T_ee = execute_grasp_from_object_pose(
    center_pose_array=center_pose_array,
    dobot=dobot,
    gripper=gripper,
    T_ee_cam=T_ee_cam,
    z_xoy_angle=30,  # 根据物体方向调整
    grasp_tilt_angle=55,  # 55度倾斜抓取
    verbose=True
)

if success:
    print("抓取成功！")
```

## 常见问题

### Q1: 如何调整抓取角度？

修改 `vertical_euler` 和 `grasp_tilt_angle` 参数：

```python
# 完全垂直
vertical_euler = [-180, 0, -90]
grasp_tilt_angle = 0

# 45度倾斜
vertical_euler = [-180, 0, -90]
grasp_tilt_angle = 45

# 侧面抓取
vertical_euler = [-90, 0, -90]
grasp_tilt_angle = 0
```

### Q2: 如何调整接近方向？

修改 `z_xoy_angle` 参数控制从哪个方向接近物体：

```python
z_xoy_angle = 0    # 从物体X轴正方向接近
z_xoy_angle = 90   # 从物体Y轴正方向接近
z_xoy_angle = -90  # 从物体Y轴负方向接近
```

## 调试技巧

1. **使用verbose模式**：
   ```python
   success, T_ee = execute_grasp_from_object_pose(..., verbose=True)
   ```

2. **检查中间结果**：
   ```python
   # 使用辅助函数检查姿态
   from grasp_utils import print_pose_info
   print_pose_info(center_pose_array, "物体姿态")
   ```



