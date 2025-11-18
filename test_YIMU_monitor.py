import cv2
import time
import rospy
from camera_reader import CameraReader


def main():
    camera = CameraReader(camera_id=10) #! 注意id有时会变化

    pixel_threshold = 2
    min_area = 2
    change_threshold = 3
    window_name = "Camera Preview"
    save_dir = None

    rospy.init_node('test', anonymous=True) 

    if camera.cap is None or not camera.cap.isOpened():
        print("无法启动相机，退出实时监测。")
        return

    prev_frame = camera.get_current_frame()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    step = 0
    while True:
        wait = rospy.Rate(10)  
        wait.sleep()
        frame = camera.get_current_frame()
        cv2.imshow(window_name, frame)

        result = camera.detect_pixel_changes(
            prev_frame,
            frame,
            threshold=pixel_threshold,
            min_area=min_area,
            save_dir=save_dir,
            step_num=step
        )

        if 'error' in result:
            print(f"[变化检测] 第{step}帧检测出错: {result['error']}")
        else:
            if result['change_percentage'] >= change_threshold:
                print(
                    f"[变化检测] 第{step}帧检测到变化: "
                    f"{result['change_percentage']:.2f}% "
                    f"({result['changed_pixels']} px, {result['num_changes']} 区域)"
                )
            else:
                print(
                    f"[变化检测] 第{step}帧无显著变化 "
                    # f"({result['change_percentage']:.2f}% < {change_threshold}%)"
                )

        prev_frame = frame
        step += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("检测结束，用户主动退出。")
            break

        time.sleep(0.01)

    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()

