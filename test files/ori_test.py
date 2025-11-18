import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.camera_reader import CameraReader
import time
from dir_detection import preprocessing


def detect_dent_orientation(img):
    #将图像经过高斯模糊处理
    img = cv2.GaussianBlur(img, (11, 11), 0)
    #检测图像中物体的边缘（轮廓线），将图像转换为只包含边缘信息的二值图像
    edges = cv2.Canny(img, 50, 150)
    cv2.imshow('edges', edges) #只有边缘线条的黑白图像
    #cv2.waitKey(1)
    
    # 霍夫直线检测
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    print(lines)
    angles = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta)
            
            # 转换为0-180度范围
            if angle > 90:
                angle = angle - 180
            angles.append(angle)
            
            # 可视化直线
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # 分析角度分布
    if angles:
        avg_angle = np.mean(angles)
        print(f"平均朝向角度: {avg_angle:.2f}度")
        
        # 显示结果
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('检测到的直线')
        
        plt.subplot(1, 2, 2)
        plt.hist(angles, bins=20, alpha=0.7)
        plt.xlabel('角度 (度)')
        plt.ylabel('频次')
        plt.title('角度分布')
        plt.show()
    
    return angles



cam = CameraReader(camera_id=10)
time.sleep(5)
background = cam.get_current_frame()

# 先切片（在原始图上），再resize
height, width = background.shape[:2]
# background = background[int(height*0.2):int(height*0.8), int(width*0.2):int(width*0.8)]
background = cv2.resize(background, (background.shape[1]//10, background.shape[0]//10))

while True:
    image = cam.get_current_frame()
    
    # 先切片，再resize（正确的顺序）
    height, width = image.shape[:2]
    # image = image[int(height*0.2):int(height*0.8), int(width*0.2):int(width*0.8)]
    image = cv2.resize(image, (image.shape[1]//10, image.shape[0]//10))
    image = preprocessing(image, background=background, show_result=False)
    

    detect_dent_orientation(image)
    cv2.imshow('image', image)
    cv2.waitKey(1)
    time.sleep(0.1)