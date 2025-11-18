# coding=utf-8
import os
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import threading

###适用于RealSense D435i相机

def create_folder_with_date():
    """创建时间戳文件夹"""
    folder_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

class CreateRealsense:
    def __init__(self, device_id):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(device_id)
        
        # 配置流
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        self.config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
        
        # 启动管道
        self.profile = self.pipeline.start(self.config)
        # 对齐对象：深度→彩色
        self.align = rs.align(rs.stream.color)
        
        # 配置传感器
        self.depth_scale = None
        self._setup_sensor()
        
        # 预热相机
        for _ in range(30):
            self.pipeline.wait_for_frames()
        print("相机初始化完成")
        
        # 线程控制
        self.running = False
        self.frame_lock = threading.Lock()
        self.frames = {'color': None, 'depth': None, 'ir1': None, 'ir2': None}
        self.save_counter = 1

    def _setup_sensor(self):
        """配置传感器参数"""
        sensor = self.profile.get_device().query_sensors()[0]
        sensor.set_option(rs.option.emitter_enabled, 0)
        
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        self.depth_scale = float(depth_scale)
        print(f"深度缩放因子: {self.depth_scale}")

    def get_frames(self):
        """获取所有帧数据"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        ir1_frame = aligned_frames.get_infrared_frame(1)
        ir2_frame = aligned_frames.get_infrared_frame(2)
        
        if not (color_frame and depth_frame and ir1_frame and ir2_frame):
            return None
            
        return {
            'color': np.asanyarray(color_frame.get_data()),
            'depth': np.asanyarray(depth_frame.get_data()),
            'ir1': np.asanyarray(ir1_frame.get_data()),
            'ir2': np.asanyarray(ir2_frame.get_data())
        }

    def _update_frames(self):
        """后台更新帧数据"""
        while self.running:
            frame_data = self.get_frames()
            if frame_data:
                with self.frame_lock:
                    self.frames = frame_data

    def _save_frames(self, save_dir):
        """保存当前帧"""
        with self.frame_lock:
            frames = self.frames.copy()
            
        if not all(frames.values()):
            return
            
        base_name = os.path.join(save_dir, f"{self.save_counter:04d}")
        
        # 保存文件
        cv2.imwrite(f"{base_name}_rgb.jpg", frames['color'])
        cv2.imwrite(f"{base_name}_depth.png", frames['depth'])
        cv2.imwrite(f"{base_name}_ir1.png", frames['ir1'])
        cv2.imwrite(f"{base_name}_ir2.png", frames['ir2'])
        
        # 保存深度彩色图
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(frames['depth'], alpha=0.03), 
            cv2.COLORMAP_JET
        )
        cv2.imwrite(f"{base_name}_depthmap.png", depth_colormap)
        
        print(f"已保存第 {self.save_counter} 组图像")
        self.save_counter += 1

    def show_frame(self):
        """显示实时画面"""
        self.running = True
        save_dir = None
        
        # 启动帧更新线程
        update_thread = threading.Thread(target=self._update_frames, daemon=True)
        update_thread.start()
        
        try:
            while self.running:
                with self.frame_lock:
                    frames = self.frames.copy()
                
                if not all(frames.values()):
                    time.sleep(0.01)
                    continue
                
                # 创建显示图像
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(frames['depth'], alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                display1 = np.hstack((frames['color'], depth_colormap))
                display2 = np.hstack((frames['ir1'], frames['ir2']))
                
                cv2.imshow("RGB & Depth", display1)
                cv2.imshow("IR1 & IR2", display2)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or cv2.getWindowProperty("RGB & Depth", cv2.WND_PROP_VISIBLE) < 1:
                    break
                elif key == ord('s'):
                    if save_dir is None:
                        save_dir = create_folder_with_date()
                        print(f"保存目录: {save_dir}")
                    self._save_frames(save_dir)
                    
        finally:
            self.running = False
            cv2.destroyAllWindows()

    def release(self):
        """释放资源"""
        self.running = False
        self.pipeline.stop()

    def get_point_coordinate(self, window_name="select_point"):
        """
        显示实时画面，允许用户点击获取该点在相机坐标系中的3D坐标
        
        Returns:
            dict: 包含点击点的像素坐标和3D坐标
                {
                    'pixel': (x, y),           # 像素坐标
                    'camera_coord': (X, Y, Z), # 相机坐标系中的3D坐标 (米)
                    'depth': depth_value       # 深度值 (米)
                }
            如果用户取消或出错，返回 None
        """
        click_data = {'clicked': False, 'x': 0, 'y': 0}
        selected_point = None
        
        def mouse_callback(event, x, y, flags, param):
            """鼠标回调函数"""
            if event == cv2.EVENT_LBUTTONDOWN:
                click_data['clicked'] = True
                click_data['x'] = x
                click_data['y'] = y
        
        # 获取相机内参
        profile = self.pipeline.get_active_profile()
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        print("=" * 60)
        print("🖱️  交互式坐标选择模式")
        print("=" * 60)
        print("操作说明:")
        print("  - 鼠标左键点击: 选择目标点")
        print("  - ESC键: 取消并退出")
        print("=" * 60)
        
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            # 显示一帧空白图像，确保窗口创建成功
            dummy_frame = np.zeros((50, 50, 3), dtype=np.uint8)
            cv2.imshow(window_name, dummy_frame)
            cv2.waitKey(1)
        except cv2.error as e:
            print(f"❌ 无法创建窗口: {e}")
            return None

        try:
            cv2.setMouseCallback(window_name, mouse_callback)
        except cv2.error as e:
            print(f"❌ 无法设置鼠标回调: {e}")
            cv2.destroyWindow(window_name)
            return None

        try:
            while True:
                # 获取当前帧
                frame_data = self.get_frames()
                if frame_data is None:
                    time.sleep(0.01)
                    continue
                
                color = frame_data['color'].copy()
                depth = frame_data['depth']
                
                # 绘制十字准星和提示信息
                h, w = color.shape[:2]
                # cv2.line(color, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 0), 1)
                # cv2.line(color, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 0), 1)
                # cv2.putText(color, "Click to select point | ESC to cancel", 
                #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 显示画面
                cv2.imshow(window_name, color)

                # 检查是否点击
                if click_data['clicked']:
                    px, py = click_data['x'], click_data['y']
                    
                    # 获取点击点的深度值（原始单位）
                    depth_value_raw = float(depth[py, px])
                    depth_scale = self.depth_scale if self.depth_scale else 0.001
                    depth_value_m = depth_value_raw * depth_scale
                    
                    if depth_value_raw == 0 or depth_value_m <= 0:
                        print(f"⚠️  警告: 点击点 ({px}, {py}) 的深度值为0，请重新选择")
                        click_data['clicked'] = False
                        continue
                    
                    # 使用相机内参将像素坐标转换为相机坐标系下的3D坐标
                    # rs2_deproject_pixel_to_point 函数执行反投影
                    camera_coord = rs.rs2_deproject_pixel_to_point(
                        intrinsics, [px, py], depth_value_m
                    )
                    
                    selected_point = {
                        'pixel': (px, py),
                        'camera_coord': tuple(camera_coord),
                        'depth': depth_value_m
                    }
                    
                    print("=" * 60)
                    print("✅ 已选择点:")
                    print(f"  像素坐标: ({px}, {py})")
                    print(f"  深度值: {depth_value_m:.4f} m")
                    print(f"  相机坐标系 (X, Y, Z): ({camera_coord[0]:.4f}, {camera_coord[1]:.4f}, {camera_coord[2]:.4f}) m")
                    print("=" * 60)
                    
                    # 在图像上标记选中的点
                    cv2.circle(color, (px, py), 5, (0, 0, 255), -1)
                    cv2.putText(color, f"Selected: ({px},{py})", 
                               (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 0, 255), 2)
                    cv2.imshow(window_name, color)
                    cv2.waitKey(1000)  # 显示1秒
                    break
                
                # 按ESC退出
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("❌ 用户取消选择")
                    break
                    
        finally:
            cv2.destroyWindow(window_name)
        
        return selected_point

def main():
    """显示可用的 RealSense 设备"""
    ctx = rs.context()
    devices = ctx.query_devices()
    
    print("可用的 RealSense 设备:")
    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"设备 {i}: {name} - 序列号: {serial}")

if __name__ == "__main__":
    main()
    device_id = input("请输入设备序列号: ")
    
    cam = CreateRealsense(device_id)
    try:
        cam.show_frame()
    finally:
        cam.release()
