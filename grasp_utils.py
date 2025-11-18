import math
import os
import time
from typing import Generator, Iterable, Optional, Tuple

import cv2
import numpy as np
from spatialmath import SE3


def normalize_angle(angle: float) -> float:
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


def detect_dent_orientation(img: np.ndarray, save_dir: Optional[str] = None) -> Tuple[Iterable[float], float]:
    """检测玻璃棒的方向角度."""
    import matplotlib

    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(img, 52, 160)

    if save_dir:
        edges_path = os.path.join(save_dir, 'edges_detected.png')
        cv2.imwrite(edges_path, edges)
    else:
        cv2.imwrite('edges_detected.png', edges)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=62)
    angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta)

            if angle > 90:
                angle -= 180
            angles.append(angle)

            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if save_dir:
        lines_path = os.path.join(save_dir, 'lines_detected.png')
        cv2.imwrite(lines_path, img)

    if angles:
        avg_angle = float(np.mean(angles))
        print(f"✅ 平均朝向角度: {avg_angle:.2f}度 (检测到 {len(angles)} 条直线)")
    else:
        avg_angle = 0.0
        print("❌ 未检测到直线")

    return angles, avg_angle


def measure_max_force(dobot, samples: int = 5, interval: float = 0.02) -> float:
    """采样若干次力传感器数据并返回绝对值最大的分量."""
    max_force = 0.0
    for _ in range(max(samples, 1)):
        force_values = dobot.get_force()
        if force_values:
            max_component = max(abs(value) for value in force_values)
            if max_component > max_force:
                max_force = max_component
        if interval > 0:
            time.sleep(interval)
    return max_force


def generate_spiral_offsets(
    step: float,
    max_radius: float,
    angle_increment_deg: float = 20.0,
    skip_count: int = 0,
) -> Generator[Tuple[float, float], None, None]:
    """生成阿基米德螺旋线上的偏移点 (mm)."""
    if step <= 0 or max_radius <= 0:
        return

    angle_increment = math.radians(angle_increment_deg if angle_increment_deg > 0 else 10.0)
    b = step / (2.0 * math.pi)
    angle = angle_increment
    point_index = 0

    while True:
        radius = b * angle
        if radius > max_radius:
            break

        if point_index >= skip_count:
            yield radius * math.cos(angle), radius * math.sin(angle)

        point_index += 1
        angle += angle_increment

