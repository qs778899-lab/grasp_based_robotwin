import cv2
import numpy as np


def segment_yellow_regions(image_path, output_mask_path=None, show_result=True):
    """
    分割图片中的黄色区域并生成mask

    参数:
        image_path: 输入图片路径
        output_mask_path: 输出mask图片路径，若为None则不保存
        show_result: 是否显示结果
    """
    # 读取图片
    image = cv2.imread(image_path)
    print(image.shape)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 转换到HSV颜色空间，HSV对颜色的区分更直观
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义HSV中黄色的范围（可以根据实际情况调整）
    # 黄色的H值大约在20-30之间
    # lower_yellow = np.array([20, 100, 100])
    # upper_yellow = np.array([30, 255, 255])
    lower_yellow = np.array([0, 0, 120])  # 调整下限
    upper_yellow = np.array([128, 256, 256])  # 调整上限
    # lower_yellow = np.array([200, 200, 200])  # 调整下限
    # upper_yellow = np.array([255, 255, 255])  # 调整上限

    # 根据阈值生成mask
    # mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.inRange(image, lower_yellow, upper_yellow)

    # 对mask进行形态学操作，去除噪点并填充空洞
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 先闭运算填充空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 再开运算去除噪点

    # 如果提供了输出路径，则保存mask
    if output_mask_path:
        cv2.imwrite(output_mask_path, mask)
        print(f"mask已保存到: {output_mask_path}")

    # 显示结果
    if show_result:
        # 应用mask到原图，只显示黄色区域
        yellow_region = cv2.bitwise_and(image, image, mask=mask)

        # 拼接原图、mask和结果图以便显示
        mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 转换为3通道以便拼接
        result = np.hstack((image, mask_3d, yellow_region))

        # 显示结果
        cv2.imshow('Original | Mask | Yellow Regions', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mask


if __name__ == "__main__":
    # 示例用法
    input_image = r'./demo_data/adaptor/rgb/000031.png'  # 替换为你的输入图片路径
    output_mask = r'./demo_data/adaptor/masks/000031.png'  # 输出mask路径
    # input_image = r'/home/jiang/Datasets/Yimu/foundation_grasp/adaptor/rgb/000031.png'  # 替换为你的输入图片路径
    # output_mask = r'/home/jiang/Datasets/Yimu/foundation_grasp/adaptor/masks/000031.png'  # 输出mask路径

    try:
        mask = segment_yellow_regions(input_image, output_mask)
        print("黄色区域分割完成")
    except Exception as e:
        print(f"处理过程中出错: {e}")
