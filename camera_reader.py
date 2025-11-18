import cv2
import numpy as np
import os
from datetime import datetime


###适用于视触觉相机

#与USB相机的通信机制: OpenCV会扫描系统的视频设备

class CameraReader:
    def __init__(self, camera_id=10, init_camera=True):
        """
        初始化相机读取器
        Args:
            camera_id: 相机ID, 默认为10
            init_camera: 是否初始化相机硬件，默认为True。如果只使用图像处理功能，可设为False
        """
        self.camera_id = camera_id
        self.cap = None
        self.init_image = None
        self.init_image_info = {}
        
        # 尝试打开相机（可选）
        if init_camera:
            self._open_camera()
        
    def _open_camera(self):
        """打开相机并获取初始图像"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                print(f"无法打开相机 {self.camera_id}")
                return False
            
            # 读取初始图像
            ret, frame = self.cap.read()
            if ret:
                self.init_image = frame.copy()
                # self._record_image_info(frame)
                print(f"成功获取相机 {self.camera_id} 的初始图像")
                return True
            else:
                print(f"无法从相机 {self.camera_id} 读取图像")
                return False
                
        except Exception as e:
            print(f"打开相机时发生错误: {e}")
            return False
    
    def reset_init(self):
        ret, frame = self.cap.read()
        self.init_image = frame.copy()

    
    # def _record_image_info(self, image):
    #     """记录影像信息"""
    #     if image is not None:
    #         # 计算影像的统计信息
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
    #         current_image_info = {
    #             'mean_intensity': np.mean(gray),
    #             'std_intensity': np.std(gray),
    #             'min_intensity': np.min(gray),
    #             'max_intensity': np.max(gray),
    #             'histogram': cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten(),
    #             'brightness': np.mean(image) if len(image.shape) == 3 else np.mean(gray),
    #             'contrast': np.std(image) if len(image.shape) == 3 else np.std(gray),
    #             'image_shape': image.shape,
    #             'image_size': image.size
    #         }
    #         print(f"初始影像信息: {current_image_info}")
    #         return current_image_info

    
    def get_current_frame(self):
        """获取当前帧"""
        if self.cap is None or not self.cap.isOpened():
            print("相机未打开")
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            print("无法读取当前帧")
            return None

    def get_init_image(self):
        """获取初始图像"""
        return self.init_image
    
    def get_init_image_info(self):
        """获取初始影像信息"""
        return self.init_image_info
    
    def get_current_image_info(self):
        """获取当前帧的影像信息"""
        current_frame = self.get_current_frame()
        if current_frame is not None:
            return self._record_image_info(current_frame)
        return None
    
    def detect_pixel_changes(self, image1, image2, threshold=30, min_area=5, save_dir=None, step_num=None):
        """
        检测两张图片之间的像素点变化
        
        Args:
            image1: 第一张图片 (numpy array)
            image2: 第二张图片 (numpy array)
            threshold: 像素差异阈值，用于判断像素是否发生变化
            min_area: 最小变化区域面积，小于此面积的变化将被忽略
            save_dir: 保存调试图像的目录（可选）
            step_num: 当前步骤编号，用于文件命名（可选）
            
        Returns:
            dict: 包含变化检测结果的字典
        """
        if image1 is None or image2 is None:
            return {'error': 'One or both images are None'}
        
        # 确保两张图片尺寸一致
        if image1.shape != image2.shape:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        # 转换为灰度图
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape) == 3 else image1
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape) == 3 else image2
        
        # 保存灰度图到指定目录
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            if step_num is not None:
                prefix = f"step_{step_num:03d}"
            else:
                timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
                prefix = f"frame_{timestamp}"
            
            cv2.imwrite(os.path.join(save_dir, f"{prefix}_gray1_before.jpg"), gray1)
            cv2.imwrite(os.path.join(save_dir, f"{prefix}_gray2_after.jpg"), gray2)
        
        # 计算像素差异
        diff = cv2.absdiff(gray1, gray2)
        
        # 应用阈值，找出变化区域
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学操作，去除噪声
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小面积变化
        significant_changes = []
        total_changed_pixels = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                significant_changes.append({
                    'area': area,
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2)
                })
                total_changed_pixels += area
        
        # 计算变化统计信息
        total_pixels = gray1.size
        change_percentage = (total_changed_pixels / total_pixels) * 100
        
        result = {
            'changed_pixels': total_changed_pixels,
            'total_pixels': total_pixels,
            'change_percentage': change_percentage,
            'num_changes': len(significant_changes),
            'changes': significant_changes,
            'diff_image': diff,
            'thresh_image': thresh,
            'has_changes': len(significant_changes) > 0
        }
        
        return result
    
    # def detect_motion(self, image1, image2, threshold=25, min_area=100):
    #     """
    #     检测运动变化（适用于视频序列）
        
    #     Args:
    #         image1: 前一帧
    #         image2: 当前帧
    #         threshold: 运动检测阈值
    #         min_area: 最小运动区域面积
            
    #     Returns:
    #         dict: 运动检测结果
    #     """
    #     if image1 is None or image2 is None:
    #         return {'error': 'One or both images are None'}
        
    #     # 确保两张图片尺寸一致
    #     if image1.shape != image2.shape:
    #         image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
    #     # 转换为灰度图
    #     gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape) == 3 else image1
    #     gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) if len(image2.shape) == 3 else image2
        
    #     # 计算帧差
    #     frame_diff = cv2.absdiff(gray1, gray2)
        
    #     # 应用高斯模糊减少噪声
    #     blurred = cv2.GaussianBlur(frame_diff, (5, 5), 0)
        
    #     # 应用阈值
    #     _, thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        
    #     # 形态学操作
    #     kernel = np.ones((5, 5), np.uint8)
    #     thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
    #     # 查找轮廓
    #     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    #     # 分析运动区域
    #     motion_regions = []
    #     total_motion_area = 0
        
    #     for contour in contours:
    #         area = cv2.contourArea(contour)
    #         if area >= min_area:
    #             x, y, w, h = cv2.boundingRect(contour)
    #             motion_regions.append({
    #                 'area': area,
    #                 'bbox': (x, y, w, h),
    #                 'center': (x + w//2, y + h//2),
    #                 'contour': contour
    #             })
    #             total_motion_area += area
        
    #     result = {
    #         'motion_detected': len(motion_regions) > 0,
    #         'num_motion_regions': len(motion_regions),
    #         'total_motion_area': total_motion_area,
    #         'motion_regions': motion_regions,
    #         'frame_diff': frame_diff,
    #         'motion_mask': thresh
    #     }
        
    #     return result
    
    def compare_with_init_image(self, current_image, method='pixel_changes', threshold=30):
        """
        将当前图像与初始图像进行对比
        
        Args:
            current_image: 当前图像
            method: 对比方法 ('pixel_changes', 'motion')
            threshold: 检测阈值
            
        Returns:
            dict: 对比结果
        """
        if self.init_image is None:
            return {'error': 'No initial image available'}
        
        if method == 'pixel_changes':
            return self.detect_pixel_changes(self.init_image, current_image, threshold)
        # elif method == 'motion':
        #     return self.detect_motion(self.init_image, current_image, threshold)
        else:
            return {'error': f'Unknown method: {method}'}
    
    def has_significant_change(self, image1, image2, change_threshold=1.0, pixel_threshold=30, min_area=5, 
                              save_dir=None, step_num=None):
        """
        简洁的变化检测方法：判断两帧之间是否发生显著变化
        
        Args:
            image1: 第一帧图像
            image2: 第二帧图像
            change_threshold: 变化百分比阈值（默认1.0%）
            pixel_threshold: 像素差异阈值（默认30）
            min_area: 最小变化区域面积（默认5）
            save_dir: 保存调试图像的目录（可选）
            step_num: 当前步骤编号（可选）
            
        Returns:
            bool: True表示检测到显著变化，False表示无显著变化
            
        Example:
            # 对比前后两帧
            frame1 = camera.get_current_frame()
            time.sleep(0.5)
            frame2 = camera.get_current_frame()
            
            if camera.has_significant_change(frame1, frame2, change_threshold=2.0):
                print("检测到显著变化！")
        """
        if image1 is None or image2 is None:
            return False
        
        # 调用底层检测方法
        result = self.detect_pixel_changes(image1, image2, 
                                           threshold=pixel_threshold, 
                                           min_area=min_area,
                                           save_dir=save_dir,
                                           step_num=step_num)
        
        # 判断是否超过阈值
        if 'error' in result:
            return False
        
        return result['change_percentage'] >= change_threshold
    
    # def release(self):
    #     """释放相机资源"""
    #     if self.cap is not None:
    #         self.cap.release()
    #     cv2.destroyAllWindows()
    #     print("相机资源已释放")