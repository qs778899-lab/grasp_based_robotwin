import numpy as np
from spatialmath import SE3


def normalize_angle(angle):
    """将角度规范化到[-180, 180]范围"""
    angle = angle % 360  # 先转换为[0, 360)范围
    if angle > 180:
        return angle - 360
    return angle

def extract_euler_zyx(rotation_matrix):
    """
    从旋转矩阵提取ZYX欧拉角（外旋）
    
    Args:
        rotation_matrix: 3x3旋转矩阵
    
    Returns:
        rx, ry, rz: 欧拉角（弧度）
    """
    sy = np.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)
    singular = sy < 1e-6
    
    if not singular:
        rx = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
        ry = np.arctan2(-rotation_matrix[2,0], sy)
        rz = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
    else:
        rx = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
        ry = np.arctan2(-rotation_matrix[2,0], sy)
        rz = 0
    
    return rx, ry, rz

def print_pose_info(T_matrix, description="姿态"):
    """
    打印变换矩阵的位姿信息
    
    Args:
        T_matrix: 4x4变换矩阵（numpy array或SE3对象）
        description: 描述信息
    """
    if isinstance(T_matrix, SE3):
        T_array = np.array(T_matrix, dtype=float)
    else:
        T_array = T_matrix
    
    translation = T_array[:3, 3]
    rotation_matrix = T_array[:3, :3]
    
    rx, ry, rz = extract_euler_zyx(rotation_matrix)
    rx_deg, ry_deg, rz_deg = np.degrees([rx, ry, rz])
    
    print(f"{description}:")
    print(f"  平移: x={translation[0]:.4f}, y={translation[1]:.4f}, z={translation[2]:.4f} m")
    print(f"  旋转: rx={rx_deg:.2f}°, ry={ry_deg:.2f}°, rz={rz_deg:.2f}°")

