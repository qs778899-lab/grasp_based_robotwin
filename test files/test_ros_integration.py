#!/usr/bin/env python3
"""
测试ROS数据集成功能
"""
import sys
sys.path.append("FoundationPose")
import rospy
from std_msgs.msg import Float64MultiArray
import threading
import time

class ROSDataGetter:
    def __init__(self):
        """初始化ROS数据获取器"""
        rospy.init_node('ros_data_getter', anonymous=True)
        
        # 线程安全的数据存储
        self.latest_tracking_data = {
            'angle_z_deg': 0.0,
            'b': 0.0,
            'x': 0.0,
            'y': 0.0,
            'timestamp': 0.0,
            'valid': False
        }
        self.data_lock = threading.Lock()
        
        # 订阅tracking_data topic
        self.tracking_sub = rospy.Subscriber(
            'tracking_data', 
            Float64MultiArray, 
            self.tracking_callback
        )
        
        print("ROS数据获取器已启动，等待数据...")
        
    def tracking_callback(self, msg):
        """处理tracking_data消息"""
        if len(msg.data) >= 4:
            angle_z_deg = msg.data[0]
            b = msg.data[1] 
            x = msg.data[2]
            y = msg.data[3]
            
            # 线程安全地更新数据
            with self.data_lock:
                self.latest_tracking_data.update({
                    'angle_z_deg': angle_z_deg,
                    'b': b,
                    'x': x,
                    'y': y,
                    'timestamp': time.time(),
                    'valid': True
                })
            
            print(f"📊 收到跟踪数据: 角度={angle_z_deg:.2f}°, 截距={b:.6f}, 位置=({x:.6f}, {y:.6f})")
        else:
            print(f"⚠️  跟踪数据格式错误，期望4个值，实际收到{len(msg.data)}个值")
    
    def get_latest_tracking_data(self):
        """获取最新的跟踪数据（线程安全）"""
        with self.data_lock:
            return self.latest_tracking_data.copy()
    
    def run(self):
        """运行ROS节点"""
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("\n👋 用户中断，正在退出...")

def main():
    """主程序测试"""
    print("=" * 50)
    print("ROS数据集成测试程序")
    print("=" * 50)
    
    # 初始化ROS数据获取器
    ros_getter = ROSDataGetter()
    
    # 在后台线程运行ROS节点
    ros_thread = threading.Thread(target=ros_getter.run, daemon=True)
    ros_thread.start()
    
    # 等待ROS节点启动
    time.sleep(2)
    
    print("开始测试数据获取...")
    
    try:
        for i in range(10):
            # 获取最新的跟踪数据
            tracking_data = ros_getter.get_latest_tracking_data()
            
            if tracking_data['valid']:
                angle_z_deg = tracking_data['angle_z_deg']
                print(f"🔄 第{i+1}次获取: angle_z_deg = {angle_z_deg:.2f}°")
                
                # 模拟使用这个角度进行计算
                target_angle = -90
                delta_angle = target_angle - angle_z_deg
                print(f"   目标角度: {target_angle}°, 差值: {delta_angle:.2f}°")
            else:
                print(f"⚠️  第{i+1}次获取: 暂无有效数据")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n👋 测试中断")
    
    print("✅ 测试完成")

if __name__ == "__main__":
    main()

