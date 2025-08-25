import os, sys, argparse, logging, glob, cv2, time
import matplotlib.pyplot as plt
import poselib

# from scipy.spatial.transform import Rotation
# from pyquaternion import Quaternion
import numpy as np

code_dir = os.path.dirname(os.path.realpath(__file__))

# add ../ to the path
sys.path.append("..")
from data_reader import BaseReader
from utils import LoftrRunner
from utils import make_Rot_so3
from optimizer.build import pyPnPMutualRefine
from kf_manager import KeyFrameManager, Frame

LOG_DIR = f'{code_dir}/../logs'
MAX_ITER_BA = 100



def run_loftr_then_PnP(
        f_iter_prev, f_iter, matcher: LoftrRunner, reader:BaseReader, 
        keyframe_manager: KeyFrameManager, 
        out_dir_corres, use_gtD=False, use_gtPose=False,
        min_matches=30, BA_step=5):
    """ Main function for robust pose estimation

    """
    # min_matches = cfgs['coarse_min_matches']
    # BA_method = cfgs['BA']['BA_method']
    # BA_step = cfgs['BA']['BA_step']

    # verbose = cfgs['verbose']
    verbose = 0
    start_time = time.time()
    
    # read the images
    if len(keyframe_manager.frames) == 0:
        file_idx0 = reader.map_iter_to_index(f_iter_prev)
        image0, mask0, depth0, _, f_name0 = reader.get_frame_pkg(file_idx0, use_gtD)
    else:
        f_id0 = len(keyframe_manager.frames)-1 
        # prev frame is the last frame in list
        _frame_0:Frame = keyframe_manager.frames[f_id0]
        file_idx0 = _frame_0.file_index
        image0 = _frame_0.rgb
        mask0 = _frame_0.mask
        depth0 = _frame_0.depth
        f_name0 = _frame_0.f_n
        

    file_idx1 = reader.map_iter_to_index(f_iter)
    image1, mask1, depth1, _, f_name1 = reader.get_frame_pkg(file_idx1, use_gtD)

    K = reader.K
    # norm_factor_2d = np.array([K[0][2], K[1][2]])[None]

    print("\n")
    logging.info(f"============ [Iter {f_iter}] Processing pair {file_idx0}->{file_idx1} ({f_name0}->{f_name1}) ============")

    # ============================= Get kpt matches from LoFTR =============================
    corres_path = f"{out_dir_corres}/{f_name0}_{f_name1}.txt"
    success_match, kpts0, kpts1 = matcher.match_features(
        corres_path, image0, image1, mask0, mask1, depth0, depth1)


    # if there's not enough feature matching, try earlier frames
    if not success_match or kpts0.shape[0] < min_matches:
        logging.warning(f"(Skip) Not enough matched features ({kpts0.shape[0]}) with frame {file_idx0}!")
        return True, 0.0
    
    if verbose > 1:
        logging.info(f"{len(kpts0)} valid kpts left after filtering.")
    
    # create the first frame
    if len(keyframe_manager.frames) == 0:
        logging.info("Create the first frame")
        f_id0 = 0
        _frame_0 = Frame(f_id0, file_idx0, image0, mask0, depth0, f_name0)
        if use_gtPose:
            _frame_0.set_local_pose(np.eye(4))
            pose_gt0 = reader.get_gt_pose(file_idx0)
            _frame_0.set_global_pose(pose_gt0)
        else:
            _frame_0.set_local_pose(np.eye(4))
            _frame_0.set_global_pose(np.eye(4))
        
        keyframe_manager.frames.append(_frame_0)
        keyframe_manager.check_and_add_keyframe(f_id=f_id0)


    # ================= estimate the intial scale for the frame =================
    # _frame_ref:Frame = keyframe_manager.frames[0]
    # f_id_ref = _frame_ref.f_id
    # image_ref, mask_ref, depth_ref, _, _ = reader.get_frame_pkg(f_id_ref, use_gtD)
    
    # depth_static_ref = depth_ref.copy()
    # depth_static_ref[mask_ref] = 0
    # depth_static1 = depth1.copy()
    # depth_static1[mask1] = 0
    # thres_ref = np.percentile(depth_static_ref, 90)
    # thres1 = np.percentile(depth_static1, 90)

    # valid_mask1 = (depth_static_ref > 0) & (depth_static_ref < thres_ref)
    # valid_mask2 = (depth_static1 > 0) & (depth_static1 < thres1)
    # valid_depth_joint = np.logical_and(valid_mask1, valid_mask2)
    # depth_static_ref = depth_static_ref[valid_depth_joint]
    # depth_static1 = depth_static1[valid_depth_joint]
    # a1_static, b1_static = estimate_bg_scale(depth_static_ref, depth_static1)

    # valid_depth1 = depth1[mask1]
    # thres_high1 = np.percentile(depth1[mask1], 95)
    # thres_high1 = valid_depth1.mean() + 3.0* valid_depth1.std()
    # thres_low1 = np.percentile(depth1[mask1], 5)


    # *********** ablation study ************
    # _frame_0.set_scaleAndShift(1.0, 0.0)
    # ***************************************

    # set initial scale ad shift
    a0_origin, b0_origin = _frame_0.scaleAndShift
    a0, b0 = a0_origin, b0_origin # frame0's from past
    # a1, b1 = a1_static, b1_static # frame1
    a1, b1 = a0, b0

    # ================= Obtain kpts from Loftr and P3ds from Mono-D =================
    # get the depth of the keypoints
    depth0_obj = depth0[kpts0[:, 1], kpts0[:, 0]].reshape(-1,1)
    depth1_obj = depth1[kpts1[:, 1], kpts1[:, 0]].reshape(-1,1)
    kpts0 = kpts0.astype('float32')
    kpts1 = kpts1.astype('float32')

    # logging.info('Calculating Depth boundaries')
    thres_d0 = np.mean(depth0_obj) + 3.0* np.std(depth0_obj)
    thres_d1 = np.mean(depth1_obj) + 3.0* np.std(depth1_obj)
    if verbose > 1:
        logging.info(f"D-bds -- thres-d0: {thres_d0:.4f}, thres-d1: {thres_d1:.4f}")

    # drop kpts with noisy depth value
    depth_mask = np.logical_and(
        depth0_obj < thres_d0, depth1_obj < thres_d1
        )[:, 0].copy()
    kpts0 = kpts0[depth_mask]
    kpts1 = kpts1[depth_mask]
    depth0_obj = depth0_obj[depth_mask]
    depth1_obj = depth1_obj[depth_mask]


    # ========================== initial guess ==========================
    # pose here means the transformation from the refObj model to the camera
    T_0to1 = np.eye(4)
    use_PnP = True
    filter_large_depth_diff = False
    # if the gap between the current frame and the prev frame is large
    # use_PnP = (use_PnP and (f_iter - f_iter_prev) > 3)
    
    if use_PnP:
        # ======== solve PnP problem to get a initial Transform ========
        camera = {
            'model': 'PINHOLE', 
            'width': reader.W, 'height': reader.H, 
            'params': [K[0][0], K[1][1], K[0][2], K[1][2]]
        }
        ransac_opt = poselib.RansacOptions()
        
        thres_pnp_reproj = 5.0
        thres_pnp_3d_dist = 0.4 # 0.2
        ransac_opt['max_reproj_error'] = thres_pnp_reproj

        # convert to 3D points, use identity scale factors
        kpts0_homo = np.vstack((kpts0.T, np.ones(kpts0.shape[0])))
        kpts0_unit = np.linalg.inv(K) @ kpts0_homo
        P3d_0 = kpts0_unit.T * (1.0 * depth0_obj + 0.0)
        # kpts1_homo = np.vstack((kpts1.T, np.ones(kpts1.shape[0])))
        # kpts1_unit = np.linalg.inv(K) @ kpts1_homo
        # P3d_1 = kpts1_unit.T * (1.0 * depth1_obj + 0.0)

        logging.info("Running PnP as initialization.")
        res_pose, stats = poselib.estimate_absolute_pose(
            kpts1, P3d_0, camera, ransac_opt, {})
        
        T_0to1_pnp = np.eye(4)
        T_0to1_pnp[:3, :3] = res_pose.R
        T_0to1_pnp[:3, 3] = res_pose.t
        pnp_inliers = np.array(stats['inliers'])

        # P3d0_to1 = P3d_0 @ T_0to1_pnp[:3,:3].T + T_0to1_pnp[:3,3:4].T

        # if filter_large_depth_diff:
        #     p3d_inliers = (np.linalg.norm(P3d0_to1-P3d_1, axis=-1) < thres_pnp_3d_dist)
        #     pnp_inliers = np.logical_and(pnp_inliers, p3d_inliers)

        P3d_0 = P3d_0[pnp_inliers]
        # P3d_1 = P3d_1[pnp_inliers]
        # P3d0_to1 = P3d0_to1[pnp_inliers]
        
        # logging.info(f"=====> PnP Result, T_0to1:\n{T_0to1_pnp}")
        T_0to1 = T_0to1_pnp

        ratio_inlier = np.sum(pnp_inliers) / len(kpts0)
        if verbose > 1:
            logging.info(f"PnP inlier ratio: {ratio_inlier:.2f}\n")

    if use_PnP and P3d_0.shape[0] < 0.5*min_matches:
        logging.warning(f"(Skip) Not enough pts after PnP ({P3d_0.shape[0]})!")
        return True, 0.0

    # Cam1_T_refObj = Cam1_T_Cam0 @ Cam0_T_refObj
    pose1_init = T_0to1 @ _frame_0.global_pose
    # =========================== Make new Frame ===========================
    # set the local pose and global pose as ground truth
    f_id1 = f_id0 + 1
    
    _frame_1 = Frame(f_id1, file_idx1, image1, mask1, depth1, f_name1)
    if use_gtPose:
        pose_gt1 = reader.get_gt_pose(file_idx1)
        T_0to1_gt = pose_gt1 @ np.linalg.inv(pose_gt0)
        _frame_1.set_local_pose(T_0to1_gt.copy())
        _frame_1.set_global_pose(pose_gt1)
    else:
        _frame_1.set_local_pose(T_0to1)
        _frame_1.set_global_pose(pose1_init)

    # a1, b1 = 1.0, 0.0
    _frame_1.set_scaleAndShift(a1, b1)


    keyframe_manager.frames.append(_frame_1)

    # *********** ablation study ************
    # # test pure PnP
    # keyframe_manager.check_and_add_keyframe(f_id1)
    # estimation_t = time.time() - start_time
    # return False, estimation_t
    # ***************************************

    # =================== Prepare Data and visualizations ====================
    if use_PnP:
        kpts0 = kpts0[pnp_inliers]
        kpts1 = kpts1[pnp_inliers]
        # kpts0_unit = kpts0_unit[:,pnp_inliers]
        # kpts1_unit = kpts1_unit[:,pnp_inliers]
        depth0_obj = depth0_obj[pnp_inliers]
        depth1_obj = depth1_obj[pnp_inliers]
    
    # geometry infos
    uvds0 = np.hstack([kpts0, depth0_obj])
    uvds1 = np.hstack([kpts1, depth1_obj])

    # ========================= local optimization =========================
    # Reminder: Global Pose -- (q0,t0)-CamW_T_Cam_i-1 and (q1,t1)-CamW_T_Cam_i
    
    # NOTE lambda for losses: spatial-3d, reproj-2d, disparity-z^-1
    # weights = np.array([5.0, 80.0, 20.0]) # YCB, HO3D
    weights = np.array([5.0, 100.0, 25.0]) # BEHAVE
    # *********** ablation study ************
    # remove some losses
    # weights[0] = 0.0
    # weights[1] = 0.0
    # weights[2] = 0.0
    # ***************************************
    loss_thres = np.array([1.0, 2.0, 1.0])

    # *********** ablation study ************
    # GT-Depth
    # weights = np.array([50.0, 10.0, 0.01]) # YCB, HO3D
    # weights = np.array([50.0, 10.0, 0.01]) # BEHAVE
    # loss_thres = np.array([1.0, 2.0, 1.0])
    # ***************************************

    if verbose > 1:
        logging.info(">>>>>>>>>> before optimization <<<<<<<<<<")
        logging.info(f"Pose0 - q_0:{_frame_0.global_q}\t t_0:{_frame_0.global_t.T}")
        logging.info(f"Pose1 - q_1:{_frame_1.global_q}\t t_1:{_frame_1.global_t.T}")
        logging.info(f"Scales - 0: {_frame_0.scaleAndShift} | 1: {_frame_1.scaleAndShift}")
    if verbose > 2:
        logging.info("Errors before neighbor optimization")
        keyframe_manager.eval_errors(f_id0, f_id1, uvds0, uvds1, weights)
        keyframe_manager.visualize_debug(f_id0, f_id1, uvds0, uvds1, 
            f"iter{f_id0}_before_BA", f"{LOG_DIR}/{reader.seq_name}/BA_corres")


    BAsolver = pyPnPMutualRefine.BAsolver()

    # fix the scale of the first frame when only process neighbor frames
    fix_ab = False
    if use_gtD:
        fix_ab = True
    
    # # ************* test PnP + no Scale Optimization *************
    # fix_ab = True

    BAsolver.addAnEdgeGlobal(
        fix_ab, True, uvds0, uvds1, K, weights, loss_thres, 
        _frame_0.global_q.elements, _frame_0.global_t, 
        _frame_1.global_q.elements, _frame_1.global_t, 
        _frame_0.scaleAndShift, _frame_1.scaleAndShift
        )

    if verbose > 0:
        logging.info(f"=====> Start neighbor frames optimization <=====")
    BAsolver.solve(MAX_ITER_BA)

    # update the global pose matrix
    T_0_global = _frame_0.update_global_pose() # Cam_i-1_T_refObj
    T_1_global = _frame_1.update_global_pose() # Cam_i_T_refObj
    # Cam_i_T_Cam_i-1 = Cam_i_T_refObj @ Cam_i-1_T_refObj^-1
    T_0to1_opt = T_1_global @ np.linalg.inv(T_0_global)
    _frame_1.set_local_pose(T_0to1_opt)

    BAsolver.reset()

    if verbose > 1:
        logging.info(">>>>>>>>>> after 2-frame optimization <<<<<<<<<<")
        logging.info(f"Pose0 - q_0:{_frame_0.global_q}\t t_0:{_frame_0.global_t.T}")
        logging.info(f"Pose1 - q_1:{_frame_1.global_q}\t t_1:{_frame_1.global_t.T}")
        logging.info(f"Scales - 0: {_frame_0.scaleAndShift} | 1: {_frame_1.scaleAndShift}")
    if verbose > 2:
        logging.info("Errors after neighbor optimization")
        keyframe_manager.eval_errors(f_id0, f_id1, uvds0, uvds1, weights)

    

    # ======================= global optimization (BA)=======================
    if f_id1 % BA_step == 0 and f_id1 > 1:
        if verbose > 0:
            logging.info(f"=====> Start global frames optimization")
        keyframe_manager.run_BA_with_given_frame(
            BAsolver, f_id1, weights, loss_thres, out_dir_corres
        )

        T_0_global = _frame_0.update_global_pose() # Cam_i-1_T_refObj
        T_1_global = _frame_1.update_global_pose() # Cam_i_T_refObj
        T_0to1_opt = T_1_global @ np.linalg.inv(T_0_global)
        _frame_1.set_local_pose(T_0to1_opt)

        if verbose > 1:
            logging.info("====> after global optimization")
            logging.info(f"q_0:{_frame_0.global_q}\t t_0:{_frame_0.global_t.T}")
            logging.info(f"q_1:{_frame_1.global_q}\t t_1:{_frame_1.global_t.T}")
            logging.info(f"0: {_frame_0.scaleAndShift} | 1: {_frame_1.scaleAndShift}")
        if verbose > 2:
            logging.info("Errors after multi-frame optimization")
            keyframe_manager.eval_errors(f_id0, f_id1, uvds0, uvds1, weights)

            keyframe_manager.visualize_debug(f_id0, f_id1, uvds0, uvds1, 
                f"iter{f_id0}_after_BA", f"{LOG_DIR}/{reader.seq_name}/BA_corres")


    keyframe_manager.check_and_add_keyframe(f_id1)

    if verbose > 0:
        logging.info("======== Tracking Finish =====================")
    estimation_t = time.time() - start_time
    
    return False, estimation_t


