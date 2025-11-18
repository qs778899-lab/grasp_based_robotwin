#!/usr/bin/env python3
"""
ROS工具模块
包含ROS订阅者相关功能，用于接收跟踪数据和图像数据
"""

import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import time


class ROSSubscriberTest:
    """
    ROS订阅者类，用于订阅跟踪数据和图像数据
    
    订阅的话题:
    - 'tracking_data': Float64MultiArray类型，包含角度和位置信息
    - 'image_object_orientation': Image类型，物体方向标注图像
    - 'raw_image': Image类型，原始图像用于图像处理
    
    主要功能:
    - 线程安全的数据缓存机制
    - 非阻塞数据访问接口
    - 自动处理ROS消息转换
    """
    
    def __init__(self):
        """初始化ROS节点和订阅者"""
        rospy.init_node('ros_subscriber_test', anonymous=True)  
        # ros node name 只是告诉 ROS："我这个节点叫什么"，与任何话题名或函数名没有直接绑定关系；保持唯一性即可。
        
        # 初始化cv_bridge用于图像转换
        self.bridge = CvBridge()

        # 整体的ros接收信息方案： callback_function + 缓存(latest_tracking_data, latest_image) + 访问函数(get_latest_tracking_data, get_latest_image)
        
        # 缓存最新的tracking_data
        self.latest_tracking_data = {
            'angle_z_deg': 0.0,
            'b': 0.0,
            'x': 0.0,
            'y': 0.0,
            'timestamp': 0.0,
            'valid': False
        }
        self.data_lock = threading.Lock()
        
        # 缓存最新的image
        self.latest_image = None
        self.image_timestamp = 0.0
        self.image_lock = threading.Lock()
        
        # 缓存最新的原始图像（用于图像处理）
        self.latest_raw_image = None
        self.raw_image_timestamp = 0.0
        self.raw_image_lock = threading.Lock()

        
        # 订阅tracking_data topic
        self.tracking_sub = rospy.Subscriber(
            'tracking_data',  # topic name: tracking_data
            Float64MultiArray, 
            self.tracking_callback  # callback_function 
        )
        
        # 订阅object_orientation topic  
        self.image_sub = rospy.Subscriber(
            'image_object_orientation',  # topic name: image_object_orientation
            Image,
            self.image_callback
        )
        
        # 订阅raw_image topic (纯净原始图像，用于图像处理)
        self.raw_image_sub = rospy.Subscriber(
            'raw_image',  # topic name: raw_image
            Image,
            self.raw_image_callback
        )
        
        print("ROS订阅者已启动，等待数据...")
        
    def tracking_callback(self, msg):
        """处理tracking_data消息"""
        if len(msg.data) >= 4:
            angle_z_deg = msg.data[0]
            b = msg.data[1] 
            x = msg.data[2]
            y = msg.data[3]

            with self.data_lock:
                self.latest_tracking_data.update({
                    'angle_z_deg': angle_z_deg,
                    'b': b,
                    'x': x,
                    'y': y,
                    'timestamp': time.time(),
                    'valid': True
                })
            
            # print(f"📊 跟踪数据: 角度={angle_z_deg:.2f}°, 截距={b:.6f}, 位置=({x:.6f}, {y:.6f})")
        else:
            pass
            # print(f"⚠️  跟踪数据格式错误，期望4个值，实际收到{len(msg.data)}个值")
    
    def image_callback(self, msg):
        """处理object_orientation图像消息"""
        try:
            # print(f"[DEBUG] 图像回调被触发！消息类型: {type(msg)}")
            # print(f"[DEBUG] 图像编码: {msg.encoding}, 尺寸: {msg.width}x{msg.height}")
            
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 线程安全地保存图像
            with self.image_lock:
                self.latest_image = cv_image.copy()
                self.image_timestamp = time.time()
            
            # # 显示图像
            # cv2.imshow("Object Orientation", cv_image)
            # cv2.waitKey(1)  # 非阻塞等待，允许其他处理
            height, width = cv_image.shape[:2]
            # print(f"🖼️  成功接收并保存图像: {width}x{height} pixels")
            
        except Exception as e:
            print(f"❌ 图像处理错误: {e}")
            import traceback
            traceback.print_exc()
    
    def raw_image_callback(self, msg):
        """处理raw_image原始图像消息"""
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 线程安全地保存原始图像
            with self.raw_image_lock:
                self.latest_raw_image = cv_image.copy()
                self.raw_image_timestamp = time.time()
            
            height, width = cv_image.shape[:2]
            # print(f"📷 成功接收原始图像: {width}x{height} pixels")
            
        except Exception as e:
            # Fallback: 手动解析ROS Image消息（绕过cv_bridge的libffi问题）
            print(f"❌ raw 图像处理错误: {e}")
            
    
    # 在callback_function基础上，访问缓存的最新数据
    def get_latest_tracking_data(self):
        """获取最新的跟踪数据（线程安全）"""
        with self.data_lock:
            return self.latest_tracking_data.copy()
    
    def get_latest_image(self):
        """获取最新的图像数据（线程安全）"""
        with self.image_lock:
            if self.latest_image is not None:
                return self.latest_image.copy(), self.image_timestamp
            else:
                return None, 0.0
    
    def get_latest_raw_image(self):
        """获取最新的原始图像数据（线程安全）"""
        with self.raw_image_lock:
            if self.latest_raw_image is not None:
                return self.latest_raw_image.copy(), self.raw_image_timestamp
            else:
                return None, 0.0
    
    def run(self):
        """运行订阅者"""
        try:
            # 非阻塞保活循环：等待ROS事件，但不主动退出
            while not rospy.is_shutdown():
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n ros中断,正在退出...")


class DummySubscriber:
    """
    占位订阅者类，用于ROS初始化失败时的fallback
    提供相同的接口但返回默认值
    """
    
    def get_latest_tracking_data(self):
        """返回默认跟踪数据"""
        return {
            'valid': False, 
            'angle_z_deg': 0.0, 
            'b': 0.0, 
            'x': 0.0, 
            'y': 0.0, 
            'timestamp': 0.0
        }
    
    def get_latest_image(self):
        """返回空图像数据"""
        return None, 0.0
    
    def get_latest_raw_image(self):
        """返回空原始图像数据"""
        return None, 0.0

