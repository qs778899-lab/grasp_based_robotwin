import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime
import os


def get_intrinsics(profile, sensor_type='color'):
    intrinsics = profile.get_intrinsics()
    print(f"\n{sensor_type}相机内参:")
    print(f"  宽度: {intrinsics.width}")
    print(f"  高度: {intrinsics.height}")
    print(f"  焦距x: {intrinsics.fx}")
    print(f"  焦距y: {intrinsics.fy}")
    print(f"  主点x: {intrinsics.ppx}")
    print(f"  主点y: {intrinsics.ppy}")
    return intrinsics


## show rgb depth ir
def record_video():
    output_data_dir = r'./demo_data/adaptor/'
    if not os.path.exists(output_data_dir):
        os.makedirs(output_data_dir)
    if not os.path.exists(os.path.join(output_data_dir, 'rgb')):
        os.makedirs(os.path.join(output_data_dir, 'rgb'))
    if not os.path.exists(os.path.join(output_data_dir, 'depth')):
        os.makedirs(os.path.join(output_data_dir, 'depth'))
    if not os.path.exists(os.path.join(output_data_dir, 'masks')):
        os.makedirs(os.path.join(output_data_dir, 'masks'))

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    # config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
    # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
    # config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)

    print(help(rs.option))

    # Start streaming
    profile = pipeline.start(config)

    # 获取相机参数
    print("获取相机参数...\n")

    # 获取RGB传感器和参数
    color_sensor = profile.get_device().first_color_sensor()
    print(color_sensor)
    # color_sensor.set_option(rs.option.sharpness, 100.0)
    print('sharpness:', color_sensor.get_option(rs.option.sharpness)) # default 50

    # 获取深度传感器和参数
    depth_sensor = profile.get_device().first_depth_sensor()
    print(depth_sensor)
    print('depth baseline:', depth_sensor.get_option(rs.option.stereo_baseline))

    depth_scale = depth_sensor.get_depth_scale()
    print(f"深度比例: {depth_scale} (米/单位)")

    emitter_enabled = depth_sensor.get_option(rs.option.emitter_enabled)
    # depth_sensor.set_option(rs.option.emitter_enabled, True)
    emitter_enabled = depth_sensor.get_option(rs.option.emitter_enabled)
    print(emitter_enabled)

    # print(help(rs.rs400_visual_preset)) # {'custom': 0, 'default': 1, 'hand': 2, 'high_accuracy': 3, 'high_density': 4, 'medium_density': 5}
    # depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)
    # print('visual_preset:', depth_sensor.get_option(rs.option.visual_preset)) # default 0.0

    # 获取内参
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = get_intrinsics(depth_profile, sensor_type='depth')

    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    color_intrinsics = get_intrinsics(color_profile, sensor_type='color')
    with open(os.path.join(output_data_dir, 'cam_K.txt'), 'w') as f:
        fx, fy, cx, cy = color_intrinsics.fx, color_intrinsics.fy, color_intrinsics.ppx, color_intrinsics.ppy
        # fx, fy, cx, cy = depth_intrinsics.fx, depth_intrinsics.fy, depth_intrinsics.ppx, depth_intrinsics.ppy
        f.write(f'{fx:06f} {0:06f} {cx:06f}\n')
        f.write(f'{0:06f} {fy:06f} {cy:06f}\n')
        f.write(f'{0:06f} {0:06f} {1:06f}\n')


    # 创建对齐对象，将深度图对齐到RGB图
    align_to = rs.stream.color
    align = rs.align(align_to)


    try:
        frame_id = 0
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            frame_id += 1

            # 对齐深度图到RGB图
            aligned_frames = align.process(frames)
            # 获取对齐后的帧
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if frame_id<=30:
                continue

            # if frame_id>100:
            #     break

            # Get current date and time
            current_time = datetime.now()
            print("Current Time:", current_time)
            cv2.imwrite(os.path.join(output_data_dir, f'rgb/{frame_id:06d}.png'), color_image)
            cv2.imwrite(os.path.join(output_data_dir, f'depth/{frame_id:06d}.png'), depth_image)

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('rgb', color_image)
            cv2.imshow("depth", depth_image*100)

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == "__main__":
    record_video()
