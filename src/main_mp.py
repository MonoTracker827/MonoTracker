import os, sys, argparse, logging, time
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import numpy as np
# import ruamel.yaml # type: ignore
# yaml = ruamel.yaml.YAML()

# set default log level to INFO
FORMAT = '%(asctime)s.%(msecs)06d %(levelname)-8s: [%(filename)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%H:%M:%S')

code_dir = os.path.dirname(os.path.realpath(__file__))

# self packages
sys.path.append("..")

from data_reader import *
from kf_manager import KeyFrameManager, Frame
from estimate_pose import run_loftr_then_PnP
from utils import *
from helper_func.eval_traj import align_pycolmap_pos, get_relative_rot, compute_metric

LOG_DIR = f'{code_dir}/../logs'


def dump_results(
        out_pose_dir, out_scale_dir, total_frames, 
        reader:BaseReader, kf_manager:KeyFrameManager
    ):

    all_Dscale = []
    num_frames = len(kf_manager.frames)

    f_cnt = 0
    frame_next:Frame = kf_manager.frames[0]
    frame_next.update_global_pose()
    file_idx_next = frame_next.file_index

    # iter through all known frames
    for f_iter in range(total_frames):
        file_idx = reader.map_iter_to_index(f_iter)
        f_name = reader.id_strs[file_idx]

        if file_idx == file_idx_next:
            F_global = frame_next.global_pose
            scale_shift = frame_next.scaleAndShift

            if f_cnt < num_frames-1:
                frame_last = frame_next
                file_idx_last = file_idx_next
                # get next frame in the valid frame list
                f_cnt += 1
                frame_next:Frame = kf_manager.frames[f_cnt]
                frame_next.update_global_pose()
                file_idx_next = frame_next.file_index
        # for frames skipped, not in the frame manager
        elif file_idx < file_idx_next:
            if file_idx-file_idx_last > file_idx_next-file_idx: # next frame is closer
                F_global = frame_next.global_pose
                scale_shift = frame_next.scaleAndShift
            else:
                F_global = frame_last.global_pose
                scale_shift = frame_last.scaleAndShift
        np.savetxt(f"{out_pose_dir}/{f_name}.txt", F_global)

        all_Dscale.append(scale_shift)
        np.savetxt(f"{out_scale_dir}/{f_name}.txt", scale_shift)

    return np.array(all_Dscale)


def main(args):
    log_dir_seq = f'{LOG_DIR}/{args.seq_name}'
    if not os.path.exists(log_dir_seq):
        os.makedirs(log_dir_seq)
        os.makedirs(f'{log_dir_seq}/BA_corres')
        os.makedirs(f'{log_dir_seq}/Trajs')
        os.makedirs(f'{log_dir_seq}/opt_vis')
        # os.makedirs(f'{log_dir_seq}/pnp_vis')

    torch.cuda.empty_cache()

    video_dir = f"{args.data_base_dir}/{args.seq_name}"
    out_dir = f"{args.out_base_dir}/{args.seq_name}"
    logging.info(f"Running sequence: {args.seq_name}")
    exp_name = args.exp_name

    out_dir_corres = f'{out_dir}/result_files/corres_cache'
    out_dir_pose_global = f'{out_dir}/result_files/global_pose'
    out_dir_scale = f'{out_dir}/result_files/scale_shift'

    start_frame = args.start_frame
    step = args.step_frame      # set frame step
    num_frames = args.num_frame # set total frames to process

    # create data reader
    if 'YCBInEOAT' in args.data_base_dir.split('/'):
        reader = YcbineoatReader(data_dir=video_dir, out_dir=out_dir, 
            downscale=1.0, shorter_side=None, start_frame=start_frame, step=step, 
            mono_type='depth_any2', normal_type='normal_m3dv2')
    elif 'BEHAVE' in args.data_base_dir.split('/'):
        reader = BEHAVEReader(data_dir=video_dir, out_dir=out_dir, 
            downscale=0.5, shorter_side=None, start_frame=start_frame, step=step, 
            mono_type='depth_any2', normal_type='normal_m3dv2')
    elif 'HO3D_v3' in args.data_base_dir.split('/'):
        reader = Ho3dReader(data_dir=video_dir, out_dir=out_dir, 
            downscale=1.0, shorter_side=None, start_frame=start_frame, step=step, 
            mono_type='depth_any2', normal_type='normal_m3dv2')
    

    # for segmentation 
    # RUN_SEGMENT = False # <<<<<<<<<<<
    # for tracking
    USE_gtD = args.use_gtD
    RUN_TRACK = True # <<<<<<<<<<<
    # for vis and eval
    RUN_VIS = True
    RUN_EVAL = True
    # whether to create video for segementation and pose estimation
    SAVE_VIDEO = False


    # if RUN_SEGMENT:
    #     crop_size_seg = -1
    #     if max(W, H) >= 1024:
    #         crop_size_seg = 768
    #     args_XMem = argparse.Namespace(
    #         xmem_model=f'/home/zilong/BundelSDF/XMem/saves/XMem-s012.pth', 
    #         max_mid_term_frames=10, min_mid_term_frames=5, max_long_term_elements=1000, 
    #         num_prototypes=128, top_k=30, mem_every=10, deep_update_every=-1, 
    #         enable_long_term=True, crop_size=crop_size_seg
    #         )
    #     config_XMem = vars(args_XMem)

    #     # set output dir
    #     out_dir_segment = f"{out_dir}/masks_net"
    #     vis_dir_segment = f"{out_dir}/rgb_mask_vis"
    #     os.system(f'rm -rf {out_dir_segment} && mkdir -p {out_dir_segment}')
    #     os.system(f'rm -rf {vis_dir_segment} && mkdir -p {vis_dir_segment}')

    #     logging.info(f"Segmenting images by XMem network...")
    #     eval_XMem.eval_XMem(config_XMem, f"{video_dir}/rgb", f"{video_dir}/masks", 
    #                         out_dir_segment, vis_dir_segment)
    #     if SAVE_VIDEO:
    #         save_video(video_name="mask_vis", frames_dir=vis_dir_segment)

    if USE_gtD:
        out_dir_pose_global = out_dir_pose_global+'_gtD'
        logging.info(f"Running with ground truth depth !")

    if RUN_TRACK:
        # re-create folders
        if not os.path.exists(out_dir_corres):
            os.system(f'mkdir -p {out_dir_corres}')
            os.system(f"chmod -R 777 {out_dir}")
        os.system(f'rm -rf {out_dir_pose_global} && mkdir -p {out_dir_pose_global}')
        os.system(f'rm -rf {out_dir_scale} && mkdir -p {out_dir_scale}')

        # NOTE configs - Mono-Depth
        coarse_min_matches = 20 # 20 | 20 | 10
        BA_Step = 5 # TODO change to adaptive trigger in the future
        MAX_KFs = 15 # window size for key frame selection
        BA_min_matches = 15 # 15 | 15 | 5

        # NOTE configs - Sparse-GT-Depth
        # coarse_min_matches = 10 # 10 | 20 | 10
        # BA_Step = 5 # TODO change to adaptive trigger in the future
        # MAX_KFs = 15
        # BA_min_matches = 5 # 5 | 15 | 5

        # create modules
        loftr = LoftrRunner()
        frame_manager = KeyFrameManager(reader, loftr, use_gtD=USE_gtD, 
            max_keyframes=MAX_KFs, kf_min_matches=BA_min_matches)
        

        time_record = []
        skip_list = [] # mark the skipped frames
        f_iter_prev = 0
        for f_iter in range(1, num_frames):
            # =============================== Tracking ==================================
            skip_flag, estimation_t = run_loftr_then_PnP(
                f_iter_prev, f_iter, loftr, reader, frame_manager, 
                out_dir_corres=out_dir_corres, 
                use_gtD=USE_gtD, use_gtPose=False, 
                min_matches=coarse_min_matches, BA_step=BA_Step
                )
            
            if skip_flag:
                skip_list.append(f_iter)
            else:
                f_iter_prev = f_iter
                # logging.info(f"Time for estimation: {estimation_t:.4f} sec")
                time_record.append(estimation_t)

            # ### NOTE For step by step debug
            # if f_iter % BA_Step == 0 and f_iter > 1:
            #     dump_results(out_dir_pose_global, out_dir_scale, num_frames, reader, frame_manager)
            # ### For step by step debug
            # if f_iter % BA_Step == 0 and f_iter > 1:
            #     input("Press Enter to continue...")

        logging.info(f"Skipped frames id: {skip_list}")

        avg_time_total = np.sum(time_record) / num_frames
        logging.info(f"Average time per frame: {avg_time_total:.2f} sec")
        avg_time_loftr = loftr.get_total_time() / num_frames
        logging.info(f"Average loftr time: {avg_time_loftr:.2f} sec")
        # logging.info(f"Average time except loftr: {avg_time_total-avg_time_loftr:.2f} sec")

        
        # ======================= dump main results =======================
        # # save pickle file for visulization
        # save_obj_as_pickle(frame_manager, f"{log_dir_seq}/frame_manager.pkl")
        # save poses and scale factors into txt files
        all_Dscale = dump_results(
            out_dir_pose_global, out_dir_scale, num_frames, reader, frame_manager
            )
        # # np.save(f'{out_dir}/result_files/Dscale.npy', all_Dscale) # save all depth scale as npy
        
        # # plot the scale factors
        # fig, ax = plt.subplots(figsize=(6, 6))
        # ax.set_title('Scale and Shift')
        # ax.plot(all_Dscale[:, 0], label='Scale', color='blue')
        # ax.set_ylabel('Scale', color='blue')
        # ax.tick_params(axis='y', labelcolor='blue')
        # ax2 = ax.twinx()
        # ax2.plot(all_Dscale[:, 1], label='Shift', color='red')
        # ax2.set_ylabel('Shift', color='red')
        # ax2.tick_params(axis='y', labelcolor='red')
        # plt.savefig(f'{log_dir_seq}/scale_factors_{exp_name}.png')
        # plt.close()

        
        


    if RUN_VIS:
        # =================== visualize the results ===================
        # poses are saved here as per-frame txt file
        draw_pose_dir = out_dir_pose_global

        # align the coordinate system of two trajs and visualize them
        pred_poses, gt_poses = draw_pose(reader, 
            draw_pose_dir, f"{draw_pose_dir}_vis", use_gt_trans=False)
        
        if SAVE_VIDEO:
            video_name = 'pose_gtD' if USE_gtD else 'pose_'+exp_name
            save_video(video_name=video_name, frames_dir=f"{draw_pose_dir}_vis")


        # store those poses in N x 12 txt format (kitti), Cam_T_baseObj
        pred_traj_path = f"{log_dir_seq}/Trajs/pred_traj_{exp_name}.txt"
        if USE_gtD:
            pred_traj_path = f"{log_dir_seq}/Trajs/pred_traj_gtD_{exp_name}.txt"
        np.savetxt(pred_traj_path, pred_poses[:, :3, :].reshape(-1, 12))

        gt_traj_path = f"{log_dir_seq}/Trajs/gt_traj.txt"
        if not os.path.exists(gt_traj_path):
            np.savetxt(gt_traj_path, gt_poses[:, :3, :].reshape(-1, 12))
        
        # visualize the trajectories
        traj_pred_poses = {'traj_np':pred_poses, 'color':'pink', 'name':'Pred. Obj Traj'}
        traj_gt_poses = {'traj_np':gt_poses, 'color':'cyan', 'name':'GT. Obj Traj'}
        traj_list = [traj_pred_poses, traj_gt_poses]
        
        num_poses = pred_poses.shape[0]
        traj_name = 'traj3d_'+exp_name # 'traj3d_gtD' if USE_gtD else 
        logging.info("Visualizing the trajectories...")
        vis_tracking(num_poses, traj_list, log_dir_seq, traj_name)
    

    if RUN_EVAL:
        # ========================== evaluate ==========================
        ref_poses = gt_poses
        aligned_est_poses = pred_poses

        get_relative_rot(reader, ref_poses, pred_poses)

        if not USE_gtD:
            aligned_est_poses = align_pycolmap_pos(ref_poses, pred_poses)
            
            ref_pts = ref_poses[:, :3, 3]
            pred_trans_pts = aligned_est_poses[:, :3, 3]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(ref_pts[:, 0], ref_pts[:, 1], ref_pts[:, 2], 
                    'x-', label="Reference Trajectory", color="green")
            ax.plot(pred_trans_pts[:, 0], pred_trans_pts[:, 1], pred_trans_pts[:, 2], 
                    'x-', label="Aligned Target Trajectory", color="red")
            ax.legend()
            plt.savefig(f"{log_dir_seq}/Trajs/aligned_traj_{exp_name}.png")
            plt.close()

            # draw_pose_dir = f'{log_dir_seq}/Trajs/aligned_pose_vis'
            # draw_poses_align(reader, ref_poses, aligned_est_poses, draw_pose_dir)
            # save_video('aligned_pose_'+exp_name, draw_pose_dir)
            # vis_rerun(seq_name, ref_poses, aligned_est_poses, reader)
        
        # get metrics - AUC, R & t err, AUC
        compute_metric(reader, ref_poses, aligned_est_poses)

    # reset the access
    os.system(f"chmod -R 777 {out_dir}")
    os.system(f"chmod -R 777 {log_dir_seq}")
    print("\n\n")

# 

if __name__ == "__main__":
    ycbineoat_list = {
        'bleach0':                  [0, 3, 111], 
        'cracker_box_reorient':     [0, 2, 101], 
        'cracker_box_yalehand0':    [0, 8, 121], 
        'mustard0':                 [0, 6, 111], 
        'sugar_box1':               [0, 7, 96], 
        'sugar_box_yalehand0':      [0, 8, 101], 
        'tomato_soup_can_yalehand0': [0, 8, 121], 
    }
    behave_list = {
        'boxmedium_hand': [140, 1, 201], 'chairwood_hand': [50, 1, 151],
        'plasticcontainer': [50, 1, 101], 'stool_move': [80, 1, 81],
        'tablesmall_lift': [0, 1, 121], 'tablesquare_lift': [0, 1, 151],
    }
    ho3d_list = {
        'AP10': [0, 5, 101], 'AP11': [0, 5, 101], 'AP12': [0, 5, 101],
        'MPM10': [280, 5, 121], 'MPM11': [1200, 5, 81], 'MPM12': [80, 8, 111],
        'SB11': [880, 5, 101], 'SB13': [880, 5, 101], 'SM1': [0, 3, 151],
    }
    # run main, when press ctrl-c, run ohter func
    try:
        parser = argparse.ArgumentParser()
        # NOTE change the data and output folder first
        # YCBInEOAT | BEHAVE/Date02 | HO3D_v3
        parser.add_argument('--data_base_dir', type=str, 
            default="/media/zilong/Documents/MasterProject/BEHAVE/Date02")
        # ycbineoat | behave/date02 | ho3d
        parser.add_argument('--out_base_dir', type=str, 
            default="/tmp/behave/date02")
        parser.add_argument('--use_gtD', action='store_true', default=False)
        parser.add_argument('--exp_name', type=str, default="ablation_wo_disp")
        
        parser.add_argument('--seq_name', type=str, default="mustard0")
        parser.add_argument('--start_frame', type=int, default=0)
        parser.add_argument('--step_frame', type=int, default=1)
        parser.add_argument('--num_frame', type=int, default=100)
        args = parser.parse_args()

        RUN_Single = False
        if RUN_Single:
            # NOTE only need to change these two lines
            args.seq_name = 'AP10'
            exp_setting = ho3d_list[args.seq_name]

            args.start_frame = exp_setting[0]
            args.step_frame = exp_setting[1]
            args.num_frame = exp_setting[2]
            main(args)
        else:
            # run multiple seqs TODO change the cfg list
            for k, v in behave_list.items():
                args.seq_name = k
                args.start_frame = v[0]
                args.step_frame = v[1]
                args.num_frame = v[2]
                main(args)
    
    except KeyboardInterrupt:
        print("\nInterrupt by ctrl-c")