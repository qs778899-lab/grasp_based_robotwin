"""
Author: zhangcongshe

Date: 2025/11/1

Version: 1.0
"""

import numpy as np
import cv2
from spatialmath import SE3, SO3

from calculate_grasp_pose_from_object_pose import calculate_grasp_pose_from_object_pose as choose_grasp_pose
from Utils import *





'''
例如grasp, lift, approach, 
twist, push, align, release, pull, nudge,等


'''
 # 核心控制函数 



def detect_object_pose_using_foundation_pose(target:str,mesh_path,cam:dict[str, Any]):
    '''
    使用foundation pose来检测物体位姿
    先找到物体分割图像（grounding + sam），然后使用foundation pose来检测物体位姿

    Args:
        target: 要检测的物体
        mesh_path: 物体的mesh路径
        str: 物体的str
        cam: env.camera_main
    Returns:
        center_pose: 物体位姿在相机坐标
    '''

    debug = 0
    debug_dir = "debug"
    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(mesh_path)
    mesh.vertices /= 1000 #! 单位转换除以1000
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    # 初始化评分器和姿态优化器
    scorer = ScorePredictor() 
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    # 创建FoundationPose估计器
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    cam_k = cam["cam_k"]
    camera = cam["cam"]
    color = camera.get_frames()['color']  #get_frames获取当前帧的所有数据（RGB、深度、红外等）
    depth = camera.get_frames()['depth']/1000
    ir1 = camera.get_frames()['ir1']
    ir2 = camera.get_frames()['ir2']
    cv2.imwrite("ir1.png", ir1)
    cv2.imwrite("ir2.png", ir2)
    mask = get_mask_from_GD(color, target)
    pose = est.register(K=cam_k, rgb=color, depth=depth, ob_mask=mask, iteration=50)
    print(f"第{frame_count}帧检测完成，pose: {pose}")
    center_pose = pose@np.linalg.inv(to_origin) #! 这个才是物体中心点的Pose
    vis = draw_posed_3d_box(cam_k, img=color, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_k, thickness=3, transparency=0, is_input_rgb=True)
    cv2.imshow('1', vis[...,::-1])

    return center_pose




def choose_grasp_pose(target,center_pose, cam_main:dict[str, Any],robot_main:dict[str, Any]):

    '''
    args:
        center_pose: 物体中心在相机坐标系中的位姿 (4x4 numpy array)
        cam: env.camera_main
        robot: env.robot_main
    Returns:
        grasp_pose: 抓取姿态 (4x4 numpy array)
    '''
    import json
    grasp_library = json.load(open('GraspLibrary.json', 'r'))
    grasp_params = grasp_library[target]

    
    # load grasp params
    z_xoy_angle = grasp_params["z_xoy_angle"]
    vertical_euler = grasp_params["vertical_euler"]
    grasp_tilt_angle = grasp_params["grasp_tilt_angle"]
    angle_threshold = grasp_params["angle_threshold"]
    T_safe_distance = grasp_params["T_safe_distance"]
    z_safe_distance = grasp_params["z_safe_distance"]
    T_ee_cam = cam_main["T_ee_cam"]
    T_tcp_ee = robot_main["tcp2ee"] 

    # ------计算在机器人基系中的object pose------
    T_cam_object = SE3(center_pose, check=False)
    pose_now = robot_main["robot"].get_pose()  # 获取当前末端执行器位姿
    x_e, y_e, z_e, rx_e, ry_e, rz_e = pose_now
    
    
    # 从当前机器人位姿构造变换矩阵 T_base_ee
    T_base_ee = SE3.Rt(
        SO3.RPY([rx_e, ry_e, rz_e], unit='deg', order='zyx'),
        np.array([x_e, y_e, z_e]) / 1000.0,  # 毫米转米
        check=False
    )
    
    # 坐标变换链: T_base_cam = T_base_ee * T_ee_cam
    T_base_cam = T_base_ee * T_ee_cam
    T_base_obj = T_base_cam * T_cam_object
    
    # ------object pose 调整------
    T_base_obj_array = np.array(T_base_obj, dtype=float)
    
    # 1. 将object pose的z轴调整为垂直桌面朝上
    current_rotation_matrix = T_base_obj_array[:3, :3]
    current_z_axis = current_rotation_matrix[:3, 2]  # 提取当前z轴方向
    target_z_axis = np.array([0, 0, 1])  # 目标z轴方向（垂直向上）
    # 计算当前z轴与目标z轴的夹角
    z_angle_error = np.degrees(np.arccos(np.clip(np.dot(current_z_axis, target_z_axis), -1.0, 1.0)))
    

    if z_angle_error > angle_threshold:
        
        # 计算旋转轴（两向量叉乘）
        rotation_axis = np.cross(current_z_axis, target_z_axis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-6:  # 两轴几乎平行
            rotation_matrix_new = current_rotation_matrix
        else:
            rotation_axis = rotation_axis / rotation_axis_norm  # 单位化旋转轴
            rotation_angle = np.arccos(np.clip(np.dot(current_z_axis, target_z_axis), -1.0, 1.0))
            # 构造反对称矩阵K（用于Rodrigues旋转公式）
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            # Rodrigues旋转公式: R = I + sin(θ)K + (1-cos(θ))K²
            R_z_align = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)
            rotation_matrix_new = np.dot(R_z_align, current_rotation_matrix)
        
        T_base_obj_aligned = np.eye(4)
        T_base_obj_aligned[:3, :3] = rotation_matrix_new
        T_base_obj_aligned[:3, 3] = T_base_obj_array[:3, 3]
        T_base_obj_final = SE3(T_base_obj_aligned, check=False)
    else:
        T_base_obj_final = T_base_obj
    
    # 2. 将object pose的x,y轴对齐到机器人基坐标系的x,y轴
    rotation_matrix_after_z = np.array(T_base_obj_final.R)
    current_x_axis = rotation_matrix_after_z[:3, 0]  # 提取当前x轴方向
    # 将x轴投影到水平面（xy平面）
    x_projected = np.array([current_x_axis[0], current_x_axis[1], 0])
    x_projected_norm = np.linalg.norm(x_projected)
    
    if x_projected_norm > 1e-6:
        x_projected = x_projected / x_projected_norm  # 单位化投影向量
        # 计算投影与基坐标系x轴的夹角
        x_angle = np.arctan2(x_projected[1], x_projected[0])
        # 构造绕z轴旋转矩阵（消除该夹角）
        R_z_align_xy = np.array([
            [np.cos(-x_angle), -np.sin(-x_angle), 0],
            [np.sin(-x_angle), np.cos(-x_angle), 0],
            [0, 0, 1]
        ])
        rotation_matrix_final = np.dot(R_z_align_xy, rotation_matrix_after_z)
        T_base_obj_final_aligned = np.eye(4)
        T_base_obj_final_aligned[:3, :3] = rotation_matrix_final
        T_base_obj_final_aligned[:3, 3] = T_base_obj_array[:3, 3]
        T_base_obj_final = SE3(T_base_obj_final_aligned, check=False)

    

    T_base_obj_array = T_base_obj_final.A
    current_rotation = T_base_obj_array[:3, :3]
    current_translation = T_base_obj_array[:3, 3]
    
    # 构造绕z轴旋转的旋转矩阵
    theta = np.radians(z_xoy_angle)
    R_z = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    new_rotation = np.dot(R_z, current_rotation)  # 左乘以在基坐标系中旋转
    T_base_obj_rotated = np.eye(4)
    T_base_obj_rotated[:3, :3] = new_rotation
    T_base_obj_rotated[:3, 3] = current_translation
    T_base_obj_final = SE3(T_base_obj_rotated, check=False)
    

    
    # ------调整抓取姿态------
    # 在垂直抓取基础上叠加倾斜角度
    tilted_euler = [vertical_euler[0] + grasp_tilt_angle, vertical_euler[1], vertical_euler[2]]
    

    # 从欧拉角构造抓取姿态（相对于物体坐标系）
    R_target_xyz = R.from_euler('xyz', tilted_euler, degrees=True)
    T_object_grasp_ideal = SE3.Rt(
        SO3(R_target_xyz.as_matrix()),
        [0, 0, 0],  # 抓取点在物体中心
        check=False
    )
    

    
    # ------计算在机器人基系中，夹爪grasp即tcp的抓取姿态------
    # 坐标变换链: T_base_grasp = T_base_obj * T_obj_grasp
    T_base_grasp_ideal = T_base_obj_final * T_object_grasp_ideal
    

    
    # ------计算在机器人基系中，末端执行器ee的抓取姿态------
    # TCP到末端执行器的偏移（z方向）
    T_tcp_ee = SE3(0, 0, T_tcp_ee)
    T_safe_distance = SE3(0, 0, T_safe_distance)  # 额外安全距离
    # 变换链: T_base_ee = T_base_grasp * T_grasp_tcp * T_tcp_ee * T_safe
    T_base_ee_ideal = T_base_grasp_ideal * T_tcp_ee * T_safe_distance
    
    # ------执行抓取动作------
    pos_mm = T_base_ee_ideal.t * 1000  # 转换为毫米
    # 提取ZYX欧拉角（机械臂使用的旋转顺序）
    rx, ry, rz = T_base_ee_ideal.rpy(unit='deg', order='zyx')

    def normalize_angle(angle):
        """将角度规范化到[-180, 180]范围"""
        angle = angle % 360  # 先转换为[0, 360)范围
        if angle > 180:
            return angle - 360
        return angle

    rz = normalize_angle(rz)  # 规范化到[-180, 180]度
    
    pre_grasp_pose = [pos_mm[0], pos_mm[1], pos_mm[2]+z_safe_distance, rx, ry, rz]
    grasp_pose = [pos_mm[0], pos_mm[1], pos_mm[2], rx, ry, rz]


    return pre_grasp_pose, grasp_pose



    
  




"""
从物体位姿计算并执行抓取动作

Args:
    center_pose_array: 物体中心在相机坐标系中的位姿 (4x4 numpy array)
    dobot: Dobot机械臂对象 初始对象
    gripper: 夹爪对象 初始对象
    T_ee_cam: 相机到末端执行器的变换矩阵 (SE3对象) 数据库
    z_xoy_angle: 物体绕z轴旋转角度，用于调整抓取接近方向 (度)  动态调整
    vertical_euler: 垂直向下抓取的grasp姿态的的欧拉角 [rx, ry, rz] (度)，默认[-180, 0, -90] 先验
    grasp_tilt_angle: 倾斜抓取角度 (度)，叠加在vertical_euler[0]上, 由垂直向下抓取旋转为斜着向下抓取的grasp姿态的旋转角度. 先验
    angle_threshold: z轴对齐的角度阈值 (度) 定数
    T_tcp_ee_z: TCP到末端执行器的z轴偏移 (米) 数据库
    T_safe_distance: 安全距离，防止抓取时与物体碰撞 (米)  先验
    # z_safe_distance: 最终移动时z方向的额外安全距离,也是为了抓取物体靠上的部分。 (毫米)
    gripper_close_pos: 夹爪闭合位置 (0-1000)，默认80 先验
    enable_gripper: 是否执行夹爪抓取动作，默认True 
    verbose: 是否打印详细信息

Returns:
    success: 是否成功执行抓取
    T_base_ee_ideal: 计算得到的理想末端执行器位姿 (SE3对象)
"""
    





def grasp(
    object, #要抓取的物体
    arm, #机械臂
    pre_grasp_dist, #抓取前距离
    grap_dis, #抓取具体位置，抓取位置的z方向
    gripper_pose#夹爪张开闭合的尺度
):
    # 计算抓取姿态
    pre_grasp_pose, grasp_pose = choose_grasp_pose()

    action.move(pre_grasp_pose_dist)

    action.move(grasp_pose)
    pass

