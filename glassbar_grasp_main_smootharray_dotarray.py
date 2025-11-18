#!/usr/bin/env python3
import sys
import signal
import atexit
sys.path.append("FoundationPose")
from estimater import *
from datareader import *
from dino_mask import get_mask_from_GD 
from qwen_mask import get_mask_from_qwen
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
from camera_reader import CameraReader
import rospy
from std_msgs.msg import Float64MultiArray


camera = None
angle_camera = None
contact_camera = None
dobot = None
gripper = None
preview_running = None


def _cleanup_resources():
    """释放相机、机械臂和窗口等资源"""
    global camera, angle_camera, contact_camera, dobot, preview_running
    
    # 停止相机预览线程
    try:
        if preview_running:
            preview_running.clear()
            print("[清理] 相机预览线程已停止")
    except Exception:
        pass
    
    try:
        if angle_camera and getattr(angle_camera, "cap", None):
            angle_camera.cap.release()
    except Exception:
        pass
    try:
        if contact_camera and getattr(contact_camera, "cap", None):
            contact_camera.cap.release()
    except Exception:
        pass
    try:
        if camera:
            camera.release()
    except Exception:
        pass
    try:
        if dobot:
            dobot.stop()
            dobot.disable_robot()
    except Exception:
        pass
    cv2.destroyAllWindows()


def _signal_handler(signum, frame):
    print("\n[中断] 用户终止程序")
    _cleanup_resources()
    try:
        rospy.signal_shutdown("User interrupt")
    except Exception:
        pass
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)
atexit.register(_cleanup_resources)



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
    gripper = DobotGripper(dobot)
    gripper.connect(init=True)
    return dobot, gripper




if __name__ == "__main__":
    rospy.init_node('ros_test', anonymous=True)
    # 创建带时间戳的保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("record_images_during_grasp", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    # print(f"图像将保存到: {save_dir}")
    
    # 创建角度数据记录文件
    angle_log_path = os.path.join(save_dir, "angle_log.csv")
    with open(angle_log_path, 'w') as f:
        f.write("frame,timestamp,angle_z_deg,detected_angles,avg_angle\n")
    # print(f"角度数据将保存到: {angle_log_path}")
    
    camera = CreateRealsense("231522072272")                     
    # #? 怎么检查没有反？
    # angle_camera = CameraReader(camera_id=11, init_camera=True)   #! 用于角度检测的USB相机 (id=11, 是后加的)
    # contact_camera = CameraReader(camera_id=10, init_camera=True) #! 用于触碰检测的USB相机 （id=10, 是原来的）
    
    # # 启动相机预览线程
    # preview_running = threading.Event()
    # preview_running.set()
    # def _camera_preview_thread():
    #     """后台线程：实时显示两个相机画面"""
    #     # 在线程内部创建窗口
    #     cv2.namedWindow("Angle Camera", cv2.WINDOW_NORMAL)
    #     cv2.namedWindow("Contact Camera", cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow("Angle Camera", 640, 480)
    #     cv2.resizeWindow("Contact Camera", 640, 480)
        
    #     while preview_running.is_set():
    #         # 获取角度相机画面
    #         angle_frame = angle_camera.get_current_frame()
    #         if angle_frame is not None:
    #             cv2.imshow("Angle Camera", angle_frame)
            
    #         # 获取接触相机画面
    #         contact_frame = contact_camera.get_current_frame()
    #         if contact_frame is not None:
    #             cv2.imshow("Contact Camera", contact_frame)
            
    #         # 必须调用waitKey让窗口响应
    #         key = cv2.waitKey(30)  # 30ms = 约33fps
    #         if key == ord('q'):
    #             print("用户按'q'关闭相机预览")
    #             preview_running.clear()
    #             break
    # preview_thread = threading.Thread(target=_camera_preview_thread, daemon=True)
    # preview_thread.start()
    # time.sleep(0.5)  # 等待窗口创建
    # print("📹 相机实时预览已启动 (按'q'可关闭预览窗口)")
    
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
    dobot, gripper = init_robot()

    
    # 初始化评分器和姿态优化器
    scorer = ScorePredictor() 
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    # 创建FoundationPose估计器
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")
    # 获取相机内参
    cam_k = np.loadtxt(f'cam_K.txt').reshape(3,3)

    
    try:
        frame_count = 0
        last_valid_pose = None  # 保存上一次有效的pose
        last_valid_angle = None  # 保存上一次有效的ROS角度
        last_seen_ts = None  # 上一次使用的ROS时间戳（tracking_data）
        last_seen_img_ts = None  # 上一次使用的图像时间戳
        last_valid_detected_angles = None  # 保存上一次检测到的角度列表
        last_valid_avg_angle = 0.0  # 保存上一次检测到的平均角度
        
        while True:
            # 获取当前帧
            # color = camera.get_frames()['color']  #get_frames获取当前帧的所有数据（RGB、深度、红外等）
            # depth = camera.get_frames()['depth']/1000
            # ir1 = camera.get_frames()['ir1']
            # ir2 = camera.get_frames()['ir2']
            frames = camera.get_frames()
            if frames is None:
                continue
            color = frames['color']  #get_frames获取当前帧的所有数据（RGB、深度、红外等）
            depth = frames['depth']/1000
            ir1 = frames['ir1']
            ir2 = frames['ir2']

            color_path = os.path.join(save_dir, f"color_frame_{frame_count:06d}.png")
            print("befor foundation pose, color_shape: ", color.shape)
            cv2.imwrite(color_path, color)
            
            
            # 每隔30帧进行一次FoundationPose检测
            if frame_count % 15 == 0:
                #使用GroundingDINO进行语义理解找到物体的粗略位置，SAM获取物体的相对精确掩码
                mask = get_mask_from_qwen(color, "red stirring rod", model_path="/home/erlin/work/labgrasp/Qwen3-VL/Qwen3-VL-4B-Thinking", bbox_vis_path=os.path.join(save_dir, f"qwen_bbox_frame_{frame_count:06d}.png"))
                # mask = get_mask_from_GD(color, "red stirring rod")
                # mask = get_mask_from_GD(color, "Plastic dropper") 
                # mask = get_mask_from_GD(color, "long yellow bar")
                # mask = get_mask_from_GD(color, "long red bar")
                # print("mask_shape: ", mask.shape)
            
                cv2.imshow("mask", mask)
                cv2.imshow("color", color)
                pose = est.register(K=cam_k, rgb=color, depth=depth, ob_mask=mask, iteration=50)
                print(f"第{frame_count}帧检测完成，pose: {pose}")
                center_pose = pose@np.linalg.inv(to_origin) #! 这个才是物体中心点的Pose
                vis = draw_posed_3d_box(cam_k, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=cam_k, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imshow('object 6D pose', vis[...,::-1])
    
                mask_path = os.path.join(save_dir, f"mask_frame_{frame_count:06d}.png")
                vis_path = os.path.join(save_dir, f"vis_frame_{frame_count:06d}.png")
                cv2.imwrite(mask_path, mask)
                cv2.imwrite(vis_path, vis[...,::-1])                

                cv2.waitKey(0) #waitKey(0) 是一种阻塞
                input("break001") #input也是一种阻塞
                print("break001")
                
                #? 清理内存 (这个有用吗？)
                torch.cuda.empty_cache()
                gc.collect()
 
                last_valid_pose = center_pose  # 保存这次检测的结果
            else:
                # 使用上一次检测的结果
                center_pose = last_valid_pose
                # print(f"第{frame_count}帧使用上次检测结果")
            

            print("center_pose_object: ", center_pose) 
            
            frame_count += 1

            if center_pose is not None:
                break

    except KeyboardInterrupt:
        print("\n[用户中断] 收到终止信号")
    finally:
        cv2.destroyAllWindows()
        # dobot.disable_robot()


    key = cv2.waitKey(1)
    # if key == ord('q'):  # 按q退出
    #     break
    # elif key == ord('a'):  # 按a执行抓取
    #
    # init_position = 10
    # gripper.control(position=init_position, force=80, speed=10)


    #? 怎么检查没有反？
    angle_camera = CameraReader(camera_id=11, init_camera=True)   #! 用于角度检测的USB相机 (id=11, 是后加的)
    contact_camera = CameraReader(camera_id=10, init_camera=True) #! 用于触碰检测的USB相机 （id=10, 是原来的）


    # 将center_pose转换为numpy数组
    center_pose_array = np.array(center_pose, dtype=float)
    
    # ------使用封装函数执行抓取------
    # 配置抓取参数
    z_xoy_angle = 0 # 物体绕z轴旋转角度
    vertical_euler = [-180, 0, -90]  # 垂直向下抓取的grasp姿态的rx, ry, rz
    grasp_tilt_angle = 30  #  由垂直向下抓取旋转为斜着向下抓取的grasp姿态的旋转角度： 加了30度会朝外旋转
    z_safe_distance= 46  #z方向的一个安全距离，也是为了抓取物体靠上的部分，可灵活调整
    
    # 调用封装函数执行抓取
    success, T_base_ee_ideal = execute_grasp_from_object_pose(
        center_pose_array=center_pose_array,
        dobot=dobot,
        gripper=gripper,
        T_ee_cam=T_ee_cam,
        z_xoy_angle=z_xoy_angle,
        vertical_euler=vertical_euler,
        grasp_tilt_angle=grasp_tilt_angle,
        angle_threshold=10.0,
        T_tcp_ee_z= -0.16, 
        T_safe_distance= 0.00, #可灵活调整
        z_safe_distance=z_safe_distance,
        gripper_close_pos=15,
        verbose=True
    )
    
    pose_now = dobot.get_pose()
    x_adjustment = 10
    z_adjustment = 50
    dobot.move_to_pose(pose_now[0]+x_adjustment, pose_now[1], pose_now[2]+z_adjustment, pose_now[3], pose_now[4], pose_now[5], speed=7, acceleration=1) 


#-----------开始检测玻璃棒方向-------------------------------------------------------
    print("\n" + "="*60)
    print("🔍 开始检测玻璃棒方向...")
    print("="*60)
    
    detected_angles = None
    avg_angle = 0.0
    detection_attempts = 0
    
    while True:
        detection_attempts += 1

        raw_image = angle_camera.get_current_frame()
        if raw_image is None:
            print(f"第{detection_attempts}次尝试: 等待相机数据...")
            time.sleep(0.1)
            continue
        img_timestamp = time.time()

        print(f"\n📷 第{detection_attempts}次尝试: 检测新原始图像方向 (时间戳: {img_timestamp:.2f})")
        detected_angles, avg_angle = detect_dent_orientation(raw_image, save_dir=save_dir)

        if detected_angles:
            last_valid_detected_angles = detected_angles
            last_valid_avg_angle = avg_angle
            last_seen_img_ts = img_timestamp
            print(f"成功检测到物体朝向角度: {detected_angles}, 平均: {avg_angle:.2f}°, 绝对值: {abs(avg_angle):.2f}°")
            print("="*60)
            break
        else:
            print("当前图像未检测到明显方向特征，继续等待...")
            time.sleep(0.1)

        # 可选：最大尝试次数限制
        if detection_attempts >= 100:
            print(" 警告: 达到最大尝试次数(100次)，使用默认角度")
            detected_angles = []
            avg_angle = 0.0
            break


#-----------开始调整玻璃棒姿态-------------------------------------------------------

    print("开始调整玻璃棒姿态至垂直桌面向下")
    pose_now = dobot.get_pose()
    delta_ee = abs(avg_angle) - grasp_tilt_angle
    #需要让tcp朝外旋转； grasp_tilt_angle为正值时，tcp会朝外旋转。
    pose_target = [pose_now[0]+15, pose_now[1], pose_now[2], pose_now[3]+delta_ee, pose_now[4], pose_now[5]]
    dobot.move_to_pose(pose_target[0], pose_target[1], pose_target[2], pose_target[3], pose_target[4], pose_target[5], speed=12, acceleration=1)
    

    wait_rate = rospy.Rate(1.0 / 12.0)  
    wait_rate.sleep()
    
    # 验证是否到达目标位置
    pose_after_adjust = dobot.get_pose()
    print(f"检查姿态调整是否完成: Rx={pose_after_adjust[3]:.2f}° (目标: {pose_target[3]:.2f}°)")



#-----------开始检测玻璃棒是否触碰到桌面-------------------------------------------------------
    print("\n开始监测玻璃棒与桌面接触...")

    gray_debug_dir = os.path.join(save_dir, "gray_images_debug")
    os.makedirs(gray_debug_dir, exist_ok=True)
    print(f"灰度图将保存到: {gray_debug_dir}")

    sample_interval = 0.1  # 秒
    move_step = 3          # mm
    max_steps = 700
    change_threshold = 3 #0.06% 变化灵敏度 

    rate = rospy.Rate(1.0 / sample_interval)
    rate.sleep()
    # rospy.sleep(sample_interval)
    frame_before = None
    while frame_before is None:
        initial_frame = contact_camera.get_current_frame()
        if initial_frame is not None:
            frame_before = initial_frame
        else:
            print("等待初始图像...")
            rospy.sleep(sample_interval)

    print("已获取初始图像")
    pose_current = dobot.get_pose()

    for step in range(max_steps):
        wait = rospy.Rate(33)  
        wait.sleep()
        # 动作前帧
        frame_data_before = contact_camera.get_current_frame()
        if frame_data_before is None:
            print(f"  步骤 {step+1}: 等待动作前图像...")
            rospy.sleep(sample_interval)
            continue
        frame_before = frame_data_before

        # 向下移动一小步
        pose_current[2] -= move_step
        dobot.move_to_pose(
            pose_current[0], pose_current[1], pose_current[2],
            pose_current[3], pose_current[4], pose_current[5],
            speed=5, acceleration=1
        )

        # 等待并抓取动作后的新帧
        frame_after = None
        has_change = False
        #连续高频采样检测
        for _ in range(20): #0.1*20 = 2s
            rate.sleep()
            candidate_frame = contact_camera.get_current_frame()
            if candidate_frame is not None:
                frame_after = candidate_frame

                has_change = contact_camera.has_significant_change(
                    frame_before, frame_after,
                    change_threshold=change_threshold,
                    pixel_threshold=2,
                    min_area=2,
                    save_dir=gray_debug_dir,
                    step_num=step
                )

                if has_change:
                    break
            
                # break

        if frame_after is None:
            print(f"  步骤 {step+1}: 未收到新图像，继续等待...")
            continue


        if has_change:
            print(f"检测到显著变化！玻璃棒可能已接触桌面 (步数: {step+1}, 下降: {(step+1)*move_step}mm)")
            break

        print(f"  步骤 {step+1}/{max_steps}: 未检测到接触，继续下降...")
    else:
        print("达到垂直向下最大移动距离，未检测到明显变化")

    print("玻璃棒下降检测完成\n")

        
    # 可选：返回home位置（根据需要取消注释）
    # dobot.move_to_pose(435.4503, 281.809, 348.9125, -179.789, -0.8424, 14.4524, speed=9)


