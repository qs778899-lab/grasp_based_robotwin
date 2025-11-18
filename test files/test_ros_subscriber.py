#!/usr/bin/env python3
"""
简单的ROS订阅者测试脚本
用于接收和显示来自realtime_object_tracking.py的两个topic：
1. tracking_data (Float64MultiArray) - 包含角度和位置信息
2. object_orientation (Image) - 包含当前帧图像
"""

import rospy
import cv2
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ROSSubscriberTest:
    def __init__(self):
        """初始化ROS节点和订阅者"""
        rospy.init_node('ros_subscriber_test', anonymous=True)
        
        # 初始化cv_bridge用于图像转换
        self.bridge = CvBridge()
        
        # 订阅tracking_data topic
        self.tracking_sub = rospy.Subscriber(
            'tracking_data', 
            Float64MultiArray, 
            self.tracking_callback
        )
        
        # 订阅object_orientation topic  
        self.image_sub = rospy.Subscriber(
            'image_object_orientation',
            Image,
            self.image_callback
        )
        
        print("ROS订阅者已启动，等待数据...")
        print("订阅的topics:")
        print("  - tracking_data (Float64MultiArray)")
        print("  - image object_orientation (Image)")
        print("按Ctrl+C退出")
        
    def tracking_callback(self, msg):
        """处理tracking_data消息"""
        if len(msg.data) >= 4:
            angle_z_deg = msg.data[0]
            b = msg.data[1] 
            x = msg.data[2]
            y = msg.data[3]
            
            print(f"📊 跟踪数据: 角度={angle_z_deg:.2f}°, 截距={b:.6f}, 位置=({x:.6f}, {y:.6f})")
        else:
            print(f"⚠️  跟踪数据格式错误，期望4个值，实际收到{len(msg.data)}个值")
    
    def image_callback(self, msg):
        """处理object_orientation图像消息"""
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 显示图像
            cv2.imshow("Object Orientation", cv_image)
            cv2.waitKey(1)  # 非阻塞等待，允许其他处理
            
            # 打印图像信息
            height, width = cv_image.shape[:2]
            print(f"🖼️  接收到图像: {width}x{height} pixels")
            
        except Exception as e:
            print(f"❌ 图像处理错误: {e}")
    
    def run(self):
        """运行订阅者"""
        try:
            rospy.spin()  # 保持节点运行
        except KeyboardInterrupt:
            print("\n👋 用户中断，正在退出...")
        finally:
            cv2.destroyAllWindows()
            print("✅ 程序已退出")

def main():
    """主函数"""
    print("=" * 50)
    print("ROS Topic 订阅测试程序")
    print("=" * 50)
    
    # 创建并运行订阅者
    subscriber = ROSSubscriberTest()
    subscriber.run()

if __name__ == "__main__":
    main()
