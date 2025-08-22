import os, sys, argparse, logging, glob, cv2, copy
import rerun as rr
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def show_cam_poses_dtu():
    poses_path = '/home/zilong/Downloads/cameras_colmap.npz'

    rr.init('dtu')
    rr.spawn(connect=True)

    camera_dict = np.load(poses_path)
    print(camera_dict.items())
    # world_mat is a projection matrix from world to image
    for idx in range(len(camera_dict)):

        world_mat = camera_dict['world_mat_%d' % idx].astype(np.float32)
        intrinsics, pose = load_K_Rt_from_P(None, world_mat[:3, :4])

        rr.set_time_seconds("stable_time", idx)
        cam_quat = R.from_matrix(pose[:3, :3]).as_quat()

        rr.log(f"camera/{idx}", rr.Transform3D(
            translation=pose[:3, 3], rotation=rr.Quaternion(xyzw=cam_quat)))
        rr.log(f"camera/{idx}", rr.Pinhole(
            resolution=[640 ,480], image_from_camera=intrinsics[:3, :3], 
            camera_xyz=rr.ViewCoordinates.RDF))


def show_cam_poses_dfs(poses_out_dir):
    poses_path = f'{poses_out_dir}/dfs_c2w.txt'

    rr.init('dfs')
    rr.spawn(connect=True)

    K = np.array([[650, 0, 320], [0, 650, 240], [0,0,1]])

    camera_poses = np.loadtxt(poses_path)
    for idx in range(len(camera_poses)):

        rr.set_time_seconds("stable_time", idx)

        pose = camera_poses[idx].reshape(4, 4)
        cam_quat = R.from_matrix(pose[:3, :3]).as_quat()

        rr.log(f"camera/{idx}", rr.Transform3D(
            translation=pose[:3, 3], rotation=rr.Quaternion(xyzw=cam_quat)))
        rr.log(f"camera/{idx}", rr.Pinhole(
            resolution=[640 ,480], image_from_camera=K, 
            camera_xyz=rr.ViewCoordinates.RDF))
        
""" For Detector-free SfM results
python ~/Disk_sda6/colmap/scripts/python/read_write_model.py \
--input_model ~/Disk_sda6/DetectorFreeSfM/SfM_dataset/ho3d/MPM10/DetectorFreeSfM_loftr_official_coarse_only__scratch_no_intrin/colmap_refined --input_format .bin \
--output_model ~/Disk_sda6/DetectorFreeSfM/SfM_dataset/ho3d/MPM10 --output_format .txt
"""

""" For COLMAP
python ~/Disk_sda6/colmap/scripts/python/read_write_model.py \
--input_model ~/Disk_sda6/DetectorFreeSfM/colmap_result/chairwood_lift/sparse/0 --input_format .bin \
--output_model ~/Disk_sda6/DetectorFreeSfM/colmap_result/chairwood_lift --output_format .txt
"""

def convert_colmap_poses(file_dir):
    """Covert the txt file extracted from colmap results to easy-to-read c2w-pose file"""
    with open(file_dir+'/images.txt', 'r') as f:
        lines = f.readlines()

    total_num = 201
    poses = np.zeros((total_num, 4, 4))
    for i in range(total_num):
        poses[i] = np.eye(4)
        
    for i in range(4, len(lines), 2):  # Skip header and process each image entry (2 lines per image)
        line = lines[i].strip()
        parts = line.split()
        
        # Read image ID and pose components
        image_name = str(parts[9].split('.')[0])  # IMAGE_NAME
        image_id = int(parts[0])
        # poses are c2w
        qw, qx, qy, qz = map(float, parts[1:5])  # Quaternion
        tx, ty, tz = map(float, parts[5:8])  # Translation
        
        # Convert quaternion to rotation matrix
        rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
        
        # Create a 4x4 transformation matrix
        w2c = np.eye(4)
        w2c[:3, :3] = rotation_matrix
        w2c[:3, 3] = [tx, ty, tz]
        
        poses[image_id] = np.linalg.inv(w2c)
    
    # poses.sort(key=lambda x: x[0])
    # print([pose[0] for pose in poses])

    for i in range(total_num):
        if not np.allclose(poses[i], np.eye(4)):
            init_pose = poses[i]
            for j in range(i):
                poses[j] = init_pose
            flag_idx = i
            break

    prev_pose = poses[flag_idx]
    for i in range(flag_idx+1, total_num):
        cur_pose = poses[i]
        if np.allclose(cur_pose, np.eye(4)):
            poses[i] = prev_pose
        else:
            prev_pose = cur_pose
    
    # Extract the sorted c2w poses
    sorted_poses_array = poses.reshape(-1, 16)
    np.savetxt(f'{poses_out_dir}/colmap_c2w.txt', sorted_poses_array)



if __name__ == "__main__":

    poses_out_dir = '/home/zilong/Disk_sda6/DetectorFreeSfM/colmap_result/tablesquare_lift'
    convert_colmap_poses(poses_out_dir)
    # show_cam_poses_dfs(poses_out_dir)
    

    
