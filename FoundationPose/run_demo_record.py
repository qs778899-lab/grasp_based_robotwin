# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
from dino_mask import get_mask_from_GD
import glob
import cv2
import imageio
import numpy as np

class CustomRecordReader:
    def __init__(self, record_dir):
        self.record_dir = record_dir
        self.color_files = sorted(glob.glob(f"{record_dir}/color/*.png"))
        self.depth_files = sorted(glob.glob(f"{record_dir}/depth/*.png"))
        
        # 读取相机内参
        cam_K_file = '/home/erlin/work/labgrasp/cam_K.txt'
        self.K = np.loadtxt(cam_K_file)
        
        self.id_strs = [f"frame_{i:06d}" for i in range(len(self.color_files))]
    
    def get_color(self, i):
        color = imageio.imread(self.color_files[i])
        return color
    
    def get_depth(self, i):
        depth = cv2.imread(self.depth_files[i], -1)
        return depth
    
    def get_mask(self, i):
        # 如果没有mask文件，返回一个全零的mask
        return np.zeros((480, 640), dtype=bool)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    
    parser.add_argument('--mesh_file', type=str, default='/home/erlin/work/labgrasp/mesh/1cm_10cm.obj')
    parser.add_argument('--record_dir', type=str, default='/home/erlin/work/labgrasp/RECORD/20251023_104936')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(args.mesh_file)
    mesh.vertices /= 1000.0

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    # 使用自定义的读取器
    reader = CustomRecordReader(record_dir=args.record_dir)

    # 存储pose的数组
    pose_array = []
    frame_indices = []
    start_index = 5
    end_index = 100

    for i in range(len(reader.color_files)):
        logging.info(f'i:{i}')
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        mask = get_mask_from_GD(color, "red cylinder")
        
        # 从start_index帧开始，每5帧获取一次pose,到end_index帧结束
        if i % 5 == 0 and i >= start_index and i <= end_index:
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

            print(f"第{i}帧检测完成，pose: {pose}")
            center_pose = pose@np.linalg.inv(to_origin) 
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[...,::-1])
            # cv2.waitKey(1)  # 短暂等待以显示图像
            cv2.waitKey(0) #waitKey(0) 是一种阻塞
            input("break001") #input也是一种阻塞

            center_pose_array = np.array(center_pose, dtype=float)
            pose_array.append(center_pose_array)
            frame_indices.append(i)
            logging.info(f'Pose computed for frame {i}')

        #   if debug>=3:
        #     m = mesh.copy()
        #     m.apply_transform(pose)
        #     m.export(f'{debug_dir}/model_tf.obj')
        #     xyz_map = depth2xyzmap(depth, reader.K)
        #     valid = depth>=0.001
        #     pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        #     o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)

      
        #   pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

        #   print("-------")
        #   print("pose", pose)
        # os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        # np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

        # if debug>=1:
        #   center_pose = pose@np.linalg.inv(to_origin)
        #   vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
        #   vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
        #   cv2.imshow('1', vis[...,::-1])
        #   cv2.waitKey(1)


        # if debug>=2:
        #   os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
        #   imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

    # 将pose数组保存到文件
    pose_array = np.array(pose_array)
    
    # 保存为CSV格式
    output_csv_file = os.path.join(args.record_dir, 'pose_array.csv')
    with open(output_csv_file, 'w') as f:
        # 写入每个pose（每行一个数组，用逗号分隔）
        for pose in pose_array:
            pose_flat = pose.flatten()  # 确保展平为一维数组
            f.write(','.join([f"{float(x):.6f}" for x in pose_flat]) + '\n')
    logging.info(f'Saved {len(pose_array)} poses to {output_csv_file}')

