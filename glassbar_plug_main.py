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
from calculate_grasp_pose_from_object_pose import (
    execute_grasp_from_object_pose, 
    detect_dent_orientation,
    adjust_to_vertical_and_lift,
    descend_with_force_feedback,
    calculate_grasppose_from_objectpose_withoutmove,
    force_guided_spiral_insertion,
)
import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ros_utils import ROSSubscriberTest, DummySubscriber



# ---------- 手眼标定 ----------
def load_hand_eye_calibration(json_path="hand_eye_calibration.json"):
    """从JSON文件加载手眼标定矩阵"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    calibration = data['T_ee_cam']
    rotation_matrix = np.array(calibration['rotation_matrix'])
    translation_vector = calibration['translation_vector']
    return SE3.Rt(rotation_matrix, translation_vector, check=False)

# 从相机坐标系到末端执行器坐标系的变换矩阵
T_ee_cam = load_hand_eye_calibration()

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
    # gripper = DobotGripper(dobot) #notice
    # gripper.connect(init=True)
    return dobot
    # return dobot, gripper #notice


# ---------- ROS节点 -------------------------------------------------------
# ROSSubscriberTest类已移至ros_utils.py模块中
# 使用方法: from ros_utils import ROSSubscriberTest, DummySubscriber


if __name__ == "__main__":
    # 初始化ROS节点（使用anonymous=True避免节点名冲突）
    rospy.init_node('glassbar_plug_main', anonymous=True)
    
    # 创建带时间戳的保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("record_images_during_grasp", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    # print(f"图像将保存到: {save_dir}")
    
    camera = CreateRealsense("231522072272") 
    # mesh_file = "mesh/cube.obj"
    mesh_file = "mesh/thin_cube.obj"
    debug = 0
    debug_dir = "debug"
    set_logging_format()
    set_seed(0)
    mesh = trimesh.load(mesh_file)
    #? openscad的单位是mm， 但是转为obj文件后单位又变成m，所以还是需要转换！
    mesh.vertices /= 1000 #! 单位转换除以1000
    # mesh.vertices /= 3
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
    

    # 初始化机械臂
    # dobot, gripper = init_robot() #notice
    dobot = init_robot()
    #notice
    # #? 初始化ROS订阅者（在后台daemon线程运行，不会阻塞main程序）
    # try:
    #     ros_subscriber = ROSSubscriberTest()
    #     ros_thread = threading.Thread(target=ros_subscriber.run, daemon=True)
    #     ros_thread.start()
    #     print("✅ ROS订阅者已在后台启动（非阻塞模式）")
    #     time.sleep(1)  # 短暂等待ROS节点启动
    # except Exception as e:
    #     print(f"⚠️  ROS订阅者启动失败: {e}")
    #     # 使用ros_utils中的DummySubscriber作为占位对象，防止后续代码出错
    #     ros_subscriber = DummySubscriber()

    # 初始化评分器和姿态优化器
    # scorer = ScorePredictor() 
    # refiner = PoseRefinePredictor()
    # glctx = dr.RasterizeCudaContext()
    # # 创建FoundationPose估计器
    # est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    # logging.info("estimator initialization done")
    # # 获取相机外参
    # cam_k = np.loadtxt(f'cam_K.txt').reshape(3,3)

    # ------交互式选择目标点坐标-------------------------------------------------------
    #? 相机内参或者调用函数计算转换不对
    # print("\n开始交互式坐标选择...")
    # point_info = camera.get_point_coordinate(window_name="select_target_point")
    
    # pixel_coord = None
    # camera_coord = None
    # depth_value = None
    # target_point = None
    # target_point_in_camera = None
    
    # if point_info is not None:
    #     pixel_coord = point_info['pixel']
    #     camera_coord = point_info['camera_coord']
    #     depth_value = point_info['depth']
        
    #     print(f"\n获取到的目标点信息:")
    #     # print(f"  像素坐标: {pixel_coord}")
    #     print(f"  相机坐标系 (X, Y, Z): ({camera_coord[0]:.4f}, {camera_coord[1]:.4f}, {camera_coord[2]:.4f}) m") # (X, Y, Z): (0.0582, -0.1203, 0.5940) m
    #     print(f"  深度值: {depth_value:.4f} m\n")
    #     # 转换为numpy数组
    #     target_point_in_camera = np.array([camera_coord[0], camera_coord[1], camera_coord[2]])  # 相机坐标系下的3D坐标
        
    # else:
    #     print("未选择目标点，跳过该步骤继续...\n")

    # input("break000")

    #错误: 0.0582, -0.1203, 0.5940
    target_point_in_camera = np.array([0.11175, -0.14139, 0.59187]) # 相机坐标系下的物体的3D坐标. eg2: x = 0.11175, y = -0.14139, z = 0.59187.  eg3: 

    #-----计算抓取姿态但不抓取-------------------------------------------------------
    #转换为机器人基坐标系
    # 配置抓取参数
    # z_xoy_angle = 0 # 物体绕z轴旋转角度
    # vertical_euler = [-180, 0, -90]  # 垂直向下抓取的grasp姿态的rx, ry, rz
    # grasp_tilt_angle = 30  #  由垂直向下抓取旋转为斜着向下抓取的grasp姿态的旋转角度： 加了30度会朝外旋转
    # z_safe_distance= 30  #z方向的一个安全距离，也是为了抓取物体靠上的部分，可灵活调整
    # T_base_ee_ideal, target_pos_mm, rx, ry, rz = calculate_grasppose_from_objectpose_withoutmove(
    #         dobot=dobot,
    #         gripper=gripper,
    #         T_ee_cam=T_ee_cam,
    #         z_xoy_angle=z_xoy_angle,
    #         vertical_euler=vertical_euler,
    #         grasp_tilt_angle=grasp_tilt_angle,
    #         angle_threshold=10.0,
    #         T_tcp_ee_z= -0.16, 
    #         T_safe_distance= 0.00, 
    #         z_safe_distance=z_safe_distance,
    #         verbose=True,
    #         target_point_camera=target_point_in_camera,)
    # print("target_pos_mm: ", target_pos_mm) #错误eg: [     759.45     -234.16        -391]
    #eg2: 

    
    # ------直接移动到玻璃棒上方抓取，直接给坐标值-------------------------------------------------------
    # dobot.move_to_pose(585, -220, 72, rx, ry, rz, speed=13, acceleration=1) 
    # wait_move = rospy.Rate(1/2)
    # wait_move.sleep()
    # gripper.control(position=13, force=12, speed=27)
    # wait_grasp = rospy.Rate(1/5)
    # wait_grasp.sleep()

    # pose_now = dobot.get_pose()
    # x_adjustment = 42
    # z_adjustment = 60
    # dobot.move_to_pose(pose_now[0]+x_adjustment, pose_now[1], pose_now[2]+z_adjustment, pose_now[3], pose_now[4], pose_now[5], speed=9)
    


    frame_count = 0
    last_valid_pose = None  # 保存上一次有效的pose
    last_valid_angle = None  # 保存上一次有效的ROS角度
    last_seen_ts = None  # 上一次使用的ROS时间戳（tracking_data）
    last_seen_img_ts = None  # 上一次使用的图像时间戳
    last_valid_detected_angles = None  # 保存上一次检测到的角度列表
    last_valid_avg_angle = 0.0  # 保存上一次检测到的平均角度
        
    # try:
    #     frame_count = 0
    #     last_valid_pose = None  # 保存上一次有效的pose
    #     last_valid_angle = None  # 保存上一次有效的ROS角度
    #     last_seen_ts = None  # 上一次使用的ROS时间戳（tracking_data）
    #     last_seen_img_ts = None  # 上一次使用的图像时间戳
    #     last_valid_detected_angles = None  # 保存上一次检测到的角度列表
    #     last_valid_avg_angle = 0.0  # 保存上一次检测到的平均角度
        
    #     while True:
    #         # 获取当前帧
    #         # color = camera.get_frames()['color']  #get_frames获取当前帧的所有数据（RGB、深度、红外等）
    #         # depth = camera.get_frames()['depth']/1000
    #         # ir1 = camera.get_frames()['ir1']
    #         # ir2 = camera.get_frames()['ir2']
    #         frames = camera.get_frames()
    #         if frames is None:
    #             continue
    #         color = frames['color']  #get_frames获取当前帧的所有数据（RGB、深度、红外等）
    #         depth = frames['depth']/1000
    #         ir1 = frames['ir1']
    #         ir2 = frames['ir2']

    #         color_path = os.path.join(save_dir, f"color_frame_{frame_count:06d}.png")
    #         print("befor foundation pose, color_shape: ", color.shape)
    #         cv2.imwrite(color_path, color)
            
            
    #         # 每隔30帧进行一次FoundationPose检测
    #         if frame_count % 15 == 0:
    #             #使用GroundingDINO进行语义理解找到物体的粗略位置，SAM获取物体的相对精确掩码
    #             mask = get_mask_from_GD(color, "red stirring rod")
    #             # mask = get_mask_from_GD(color, "Plastic dropper") 
    #             # mask = get_mask_from_GD(color, "long yellow bar")
    #             # mask = get_mask_from_GD(color, "long red bar")
    #             # print("mask_shape: ", mask.shape)
            
    #             cv2.imshow("mask", mask)
    #             cv2.imshow("color", color)
    #             pose = est.register(K=cam_k, rgb=color, depth=depth, ob_mask=mask, iteration=50)
    #             print(f"第{frame_count}帧检测完成，pose: {pose}")
    #             center_pose = pose@np.linalg.inv(to_origin) 
    #             vis = draw_posed_3d_box(cam_k, img=color, ob_in_cam=center_pose, bbox=bbox)
    #             vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_k, thickness=3, transparency=0, is_input_rgb=True)
    #             cv2.imshow('1', vis[...,::-1])
    
    #             mask_path = os.path.join(save_dir, f"mask_frame_{frame_count:06d}.png")
    #             vis_path = os.path.join(save_dir, f"vis_frame_{frame_count:06d}.png")
    #             cv2.imwrite(mask_path, mask)
    #             cv2.imwrite(vis_path, vis[...,::-1])                

    #             # cv2.waitKey(0) #waitKey(0) 是一种阻塞
    #             # input("break001") #input也是一种阻塞
    #             # print("break001")
    
    #             torch.cuda.empty_cache()
    #             gc.collect()
 
    #             last_valid_pose = center_pose  # 保存这次检测的结果
    #         else:
    #             # 使用上一次检测的结果
    #             center_pose = last_valid_pose
    #             # print(f"第{frame_count}帧使用上次检测结果")

    #         print("center_pose_object: ", center_pose) 
    #         frame_count += 1
    #         if center_pose is not None:
    #             break

    # except KeyboardInterrupt:
    #     print("\n[用户中断] 收到终止信号")
    # finally:
    #     cv2.destroyAllWindows()
    #     # dobot.disable_robot()


    key = cv2.waitKey(1)
    # if key == ord('q'):  # 按q退出
    #     break
    # elif key == ord('a'):  # 按a执行抓取
    #
    # init_position = 10
    # gripper.control(position=init_position, force=80, speed=10)



#     # 将center_pose转换为numpy数组
#     center_pose_array = np.array(center_pose, dtype=float)
    
# # -------执行抓取-------------------------------------------------------
#     # 配置抓取参数
#     z_xoy_angle = 0 # 物体绕z轴旋转角度
#     vertical_euler = [-180, 0, -90]  # 垂直向下抓取的grasp姿态的rx, ry, rz
#     grasp_tilt_angle = 30  #  由垂直向下抓取旋转为斜着向下抓取的grasp姿态的旋转角度： 加了30度会朝外旋转
#     z_safe_distance= 39  #z方向的一个安全距离，也是为了抓取物体靠上的部分，可灵活调整
    
#     # 调用封装函数执行抓取
#     success, T_base_ee_ideal = execute_grasp_from_object_pose(
#         center_pose_array=center_pose_array,
#         dobot=dobot,
#         gripper=gripper,
#         T_ee_cam=T_ee_cam,
#         z_xoy_angle=z_xoy_angle,
#         vertical_euler=vertical_euler,
#         grasp_tilt_angle=grasp_tilt_angle,
#         angle_threshold=10.0,
#         T_tcp_ee_z= -0.16, 
#         T_safe_distance= 0.00, #可灵活调整
#         z_safe_distance=z_safe_distance,
#         gripper_close_pos=15,
#         verbose=True
#     )
    
#     pose_now = dobot.get_pose()
#     x_adjustment = 115
#     z_adjustment = 180
#     dobot.move_to_pose(pose_now[0]+x_adjustment, pose_now[1], pose_now[2]+z_adjustment, pose_now[3], pose_now[4], pose_now[5], speed=7, acceleration=1) 







#     wait1 = rospy.Rate(1.0 / 5.0)
#     wait1.sleep()
# #-------检测玻璃棒方向-------------------------------------------------------
#     #mark: 循环获取ROS原始图像并检测方向，直到检测成功
#     print("开始检测玻璃棒方向...")

#     detected_angles = None
#     avg_angle = 0.0
#     detection_attempts = 0
    
#     while True:
#         detection_attempts += 1
#         # 获取ROS原始图像数据
#         raw_image, img_timestamp = ros_subscriber.get_latest_raw_image()
#         has_new_image = raw_image is not None
#         if has_new_image:
#             # 收到新图像，进行方向检测
#             print(f"\n📷 第{detection_attempts}次尝试: 检测新原始图像方向 (时间戳: {img_timestamp:.2f})")
#             detected_angles, avg_angle = detect_dent_orientation(raw_image, save_dir=save_dir)
            
#             if detected_angles:
#                 last_valid_detected_angles = detected_angles
#                 last_valid_avg_angle = avg_angle
#                 last_seen_img_ts = img_timestamp
#                 print(f"成功检测到物体朝向角度: {detected_angles}, 平均: {avg_angle:.2f}°")
#                 print("="*60)
#                 break  
#             else:
#                 print("当前图像未检测到明显方向特征，继续等待...")
#                 time.sleep(0.1)  
#         else:
#             print(f"第{detection_attempts}次尝试: 等待图像数据...")
#             time.sleep(0.1)  
        
#         # 可选：最大尝试次数限制
#         if detection_attempts >= 100:
#             print(" 警告: 达到最大尝试次数(100次)，使用默认角度")
#             detected_angles = []
#             avg_angle = 0.0
#             break



#-----------开始调整玻璃棒姿态-------------------------------------------------------
    # # 调用封装函数：调整姿态至垂直并抬升
    # adjust_result = adjust_to_vertical_and_lift(
    #     dobot=dobot,
    #     avg_angle=avg_angle, # 检测到的玻璃棒当前倾斜角度（度）
    #     grasp_tilt_angle=grasp_tilt_angle,
    #     x_adjustment = 0,
    #     z_adjustment = 0,
    #     verbose=True
    # )

    # wait_rate = rospy.Rate(1.0 / 5.0)  
    # wait_rate.sleep()
    


# --------移动到目标hole正上方的位置-------------------------------------------------------

    #移动到目标位置
    pose_now = dobot.get_pose()
    x_safe_adjustment = -52
    y_safe_adjustment = 30
    z_safe_adjustment = -50
    pose_now = [586, -22, 68, -133.37, 0, -90]
    #target_pos_mm
    dobot.move_to_pose(pose_now[0], pose_now[1], pose_now[2], pose_now[3], pose_now[4], pose_now[5], speed=9)

    wait_nearby = rospy.Rate(1.0 / 3.5)  
    wait_nearby.sleep()


#-------通过力控下降并在XY平面执行螺旋微调，尝试插入hole-------------------------------------------------------
    insertion_result = force_guided_spiral_insertion(
        dobot=dobot,
        verbose=True,
    )

    print("螺旋插入结果:", insertion_result)

    # # 调用封装函数：垂直下降并检测力反馈
    # descend_result = descend_with_force_feedback(
    #     dobot=dobot,
    #     move_step=1,
    #     max_steps=700,
    #     force_threshold=1.5,
    #     verbose=True
    # )


#if __name__ == "__main__":
#!  main() 封装进去