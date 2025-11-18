# 视触觉传感器使用说明

## 运行流程

1. 
cd /home/erlin/work/tasks
conda activate normalflow
bash create_virtualcam.bash
或者需要创建多个虚拟相机：
bash create_virtualcam_multi.bash


2. 
在终端终结者运行roscore

3. 
cd /home/erlin/work/tasks
conda activate normalflow
bash open_sn.bash
实时看到传感器画面

4. 计算object_pose (二维的pose)
open_sn.bash中运行的realtime_object_tracking.py会计算得到object pose, 也会发送ros topic

5. 测试是否已经发送topic(ros node和ros topic不同)

可以查看ros通信发送情况：
rqt

查看所有活跃的节点：
rosnode list

查看所有活跃的话题（存在但不一定有实际数据发出）：
rostopic list
（注意，当没有contact时，tracking_data没有数据会发出）


查看具体某个ros topic是否有数据流：
rostopic echo -p /tracking_data 
rostopic echo -p /image_object_orientation 


查看哪些node在发布the topic消息:
rostopic info tracking_data




6. 测试能否接收ros topic和print出正确的结果
运行test_ros_subscriber.py 测试文件


7. 执行抓取的代码文件
主文件1：glassbar_grasp_main.py
主文件2：glassbar_grasp_main_force.py



8. 边缘计算检测直线角度的合适参数记录：

img = cv2.GaussianBlur(img, (3, 3), 0) #适度模糊（模糊可以减少噪声造成的误判），保留边缘细节
edges = cv2.Canny(img, 55, 160)  #低阈值threshold1，高阈值threshold2（降低高阈值提高检测率）
lines = cv2.HoughLines(edges, 1, np.pi/180, threshold= 65)



9. 通过像素变化判断玻璃棒是否触碰桌面（主文件1）：
影响判断的参数：
gripper_close_pos=20, (21可能会检测不到orientation of glassbar)
change_threshold=0.06, 
pixel_threshold=2, 
min_area=2


9. 通过力控感知判断玻璃棒是否触碰桌面（主文件2）：



10. 常用调试文件:

test_pixel_change_monitor.py: 测试通过前后帧（视触觉退图像）像素变化是否可以判断玻璃棒触碰桌面。
test_force.py:  测试通过力控感知变化是否可以灵敏判断玻璃棒触碰桌面。
test_ros_subscriber.py: 测试能否正常接收realtime_object_tracking.py发送的topics。





11. 最后结束时，记得拔下传感器的usb接口（接在电脑主机上）







##grounding_dino 常见提示词使用记录

提示词示例：mask = get_mask_from_GD(color, "stirring rod")中的stirring rod是提示词

1. red marker:
无法精准定位玻璃棒上红色区域

2. Plastic dropper:
可以找到并抓取塑料滴管

3. stirring rod:
可以定位在玻璃棒上，但位置会上下浮动




##常见物体识别问题:


1. 物体放置角度 或 场景中有其他杂物

2. 相机的高度

3. 灯光太弱缺乏反光（透明物体）

4. grounding_dino提示词prompt不合适

5. mesh文件尺寸不对 或 mesh文件没有进行单位转换






## 其他常见硬件类问题：

1. 深度相机接口不稳定
如无法获取图像

2. 夹爪无法连接
因程序中断多次需要重启机械臂

3. 
