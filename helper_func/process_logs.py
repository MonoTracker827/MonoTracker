import os, sys, argparse, pickle, cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

code_dir = os.path.dirname(os.path.realpath(__file__))

sys.path.append("..")
from src.kf_manager import *
from src.utils import draw_pose, make_E_se3

LOG_DIR = f'{code_dir}/../logs'

# deprecated
# def extract_pose_for_evo_traj(keyframe_manager: KeyFrameManager, out_path: str):
#     data_reader = keyframe_manager.data_reader
#     frame0 = keyframe_manager.frames[0]
#     pose_offset = data_reader.get_gt_pose(frame0.file_index)
#     w2c_gt_list = []
#     w2c_pred_list = []
#     for frame in keyframe_manager.frames:
#         w2c_gt = data_reader.get_gt_pose(frame.file_index)
#         w2c_gt = make_E_se3(w2c_gt)
#         w2c_pred = np.eye(4)
#         w2c_pred[:3, :3] = frame.global_q.rotation_matrix
#         w2c_pred[:3, 3] = frame.global_t
#         w2c_pred = make_E_se3(w2c_pred)
#         # check R is in SO(3)
#         R = w2c_pred[:3, :3]
#         assert np.allclose(np.linalg.det(R), [1.0], atol=1e-6)
#         assert np.allclose(R.transpose().dot(R), np.eye(3), atol=1e-6)
#         R_gt = w2c_gt[:3, :3]
#         assert np.allclose(np.linalg.det(R_gt), [1.0], atol=1e-6)
#         assert np.allclose(R_gt.transpose().dot(R_gt), np.eye(3), atol=1e-6)

#         # print("R is in SO(3)")

#         w2c_pred = w2c_pred @ pose_offset
#         # flatten the matrix
#         w2c_gt_list.append(w2c_gt[:-1].flatten())
#         w2c_pred_list.append(w2c_pred[:-1].flatten())
#     w2c_gt_list = np.array(w2c_gt_list)
#     w2c_pred_list = np.array(w2c_pred_list)
#     np.savetxt(f"{out_path}/gt_traj.txt", w2c_gt_list)
#     np.savetxt(f"{out_path}/pred_traj.txt", w2c_pred_list)




def process_logs(args, frame_manager:KeyFrameManager):
    out_dir = f"{frame_manager.data_reader.out_dir}/result_files"
    seq_name = frame_manager.data_reader.seq_name

    if args.grid:
        frame_manager.visualize_gird_scale_shift()
    elif args.opt:
        frame_manager.visualize_optimization_log()
    elif args.pose:
        # visualize poses by drawing bounding box on the images
        out_dir_pose_global = f"{out_dir}/global_pose"
        pred_poses, gt_poses = draw_pose(frame_manager.data_reader, 
            out_dir_pose_global, f"{out_dir_pose_global}_vis", 
            use_gt_trans=False, global_pose=True, global_scale=1
            )
        # store those poses in N x 12 txt format (kitti), CAMs_T_refObj
        pred_traj_path = f"{LOG_DIR}/{seq_name}/Trajs/pred_traj.txt"
        gt_traj_path = f"{LOG_DIR}/{seq_name}/Trajs/gt_traj.txt"
        np.savetxt(pred_traj_path, pred_poses[:, :3, :].reshape(-1, 12))
        np.savetxt(gt_traj_path, gt_poses[:, :3, :].reshape(-1, 12))

    elif args.nerfstudio_json:
        # convert poses and camera info to json
        from prepare_nerf_data import NerfstudioDatasetTransformer
        out_dir_pose_global = f"{out_dir}/global_pose"
        K = frame_manager.data_reader.K

        transformer = NerfstudioDatasetTransformer(frame_manager,
            depth_type="depth_m3dv2", mask_type="masks_net", use_gt_pose=False)
        transformer.save_json(f"{LOG_DIR}/{seq_name}/nerf_est.json")


if __name__ == "__main__":
    # add argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--grid", action="store_true")   
    parser.add_argument("-o", "--opt", action="store_true")
    parser.add_argument("-p", "--pose", action="store_true")
    parser.add_argument("-j", "--nerfstudio_json", action="store_true")
    args = parser.parse_args()

    seq_name = 'bleach0'
    log_file = f"{LOG_DIR}/{seq_name}/frame_manager.pkl"
    with open(log_file, "rb") as f:
        frame_manager: KeyFrameManager = pickle.load(f)
    process_logs(args, frame_manager)