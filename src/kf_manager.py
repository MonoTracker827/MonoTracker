import os, logging, math, cv2, pickle, logging
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
from numpy.linalg import inv, norm
import pyquaternion as pyq
from typing import List, Dict, Tuple
import poselib

from data_reader import BaseReader
from utils import LoftrRunner, huber_loss, make_Rot_so3
from utils import compute_cos_similarity, compute_Chi_squared_distance

code_dir = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = f'{code_dir}/../logs'
MAX_ITER_BA = 100


def max_pooling(img: np.ndarray, win_size: int) -> np.ndarray:
    assert len(img.shape) == 2
    res = np.zeros((math.ceil(img.shape[0]/win_size), math.ceil(img.shape[1]/win_size)))
    res.astype(np.uint8)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = np.max(img[i*win_size:(i+1)*win_size, j*win_size:(j+1)*win_size])
    return res



class Frame:
    """
    pose is transformation from refObj to cam 
    """
    def __init__(self, f_id, file_index, f_n):
        self.f_id = f_id                # id for frames in the frame manager
        self.f_n = f_n
        self.file_index = file_index    # frame index in the video sequence
        self.local_pose = np.eye(4)
        self.global_pose = np.eye(4)    # cam_T_refObj
        self.global_q = pyq.Quaternion(matrix = np.eye(3))
        self.global_t = np.zeros(3).copy()

        # ****************
        self.sift_feature = np.zeros(8)
        self.gt_rot = np.eye(3)
        self.max_depth = 1.0
        # ****************

        self.is_keyframe = False
        self.scaleAndShift = np.array([1.0, 0.0])

        

    def __init__(self, f_id, file_index, rgb, mask, depth, f_n):
        self.f_id = f_id                # id for frames in the frame manager
        self.f_n = f_n
        self.file_index = file_index    # frame index in the video sequence
        self.local_pose = np.eye(4)
        self.global_pose = np.eye(4)    # cam_T_refObj
        self.global_q = pyq.Quaternion(matrix = np.eye(3))
        self.global_t = np.zeros(3).copy()

        # ****************
        self.sift_feature = np.zeros(8)
        self.gt_rot = np.eye(3)
        self.max_depth = 1.0
        # ****************

        self.is_keyframe = False
        self.scaleAndShift = np.array([1.0, 0.0])

        # pre-load
        self.rgb = rgb
        self.mask = mask
        self.depth = depth
        

    
    def set_global_pose(self, g_pose):
        """ Set global pose as initialization, then update the quat & trans
        Input:
        - global pose matrix, shape [4, 4]
        """
        self.global_pose = g_pose
        # NOTE return a unit quaternion in order of [s, x, y, z]
        self.global_q = pyq.Quaternion(matrix = g_pose[:3, :3].copy())
        # NOTE use copy() to make sure the global_t is continuous in memory
        # so that it can be passed to C++ code
        self.global_t = g_pose[:3, 3].copy().reshape(-1)

    def update_global_pose(self):
        """
        update global pose according to new quat & trans
        """
        new_g_pose = np.eye(4) # Cam_i_T_refObj
        new_g_pose[:3,:3] = self.global_q.rotation_matrix
        new_g_pose[:3, 3] = self.global_t
        self.global_pose = new_g_pose.copy()
        return new_g_pose
    
    def set_local_pose(self, l_pose):
        """
        Args:
            l_pose (4x4 ndarray): the transformation matrix from last frame to current frame
        """
        self.local_pose = l_pose.copy()

    def set_scaleAndShift(self, scale, shift):
        """set scale and shift for the frame's mono depth map
        """
        self.scaleAndShift = np.array([scale, shift])

    def __str__(self) -> str:
        res = f"""=====Frame {self.file_index}=====
            is_keyframe: {self.is_keyframe} q: {self.global_q}\t t: {self.global_t}\n"""
        
        return res



class FeaturePair:
    def __init__(self, f_id0, f_id1, uvds0, uvds1):
        self.f_id0 = f_id0
        self.f_id1 = f_id1
        self.uvds0 = uvds0.copy()
        self.uvds1 = uvds1.copy()



class OptimizationLog:
    """Stores necessary information for the matched point clouds"""
    def __init__(self, 
            feature_pairs_dict: Dict[Tuple[int,int],FeaturePair], 
            frames_before: Dict[int,Frame], 
            frames_after: Dict[int,Frame]
        ):
        self.feature_pairs_dict = feature_pairs_dict
        self.frames_before = frames_before
        self.frames_after = frames_after
        self.selected_frames_id = list(frames_before.keys())




class KeyFrameManager:
    def __init__(self, data_reader, detector, use_gtD=False, 
                 max_keyframes=10, kf_min_matches=20, vis_angle=70):
        self.keyframes_id:List[int] = []
        self.frames:List[Frame] = []
        self.data_reader:BaseReader = data_reader
        self.detector:LoftrRunner = detector

        self.use_gtD = use_gtD
        self.max_keyframes = max_keyframes
        self.kf_min_matches = kf_min_matches
        # Visble angle between normal and point to camera origin
        self.cos_vis_angle = np.cos(vis_angle/180*np.pi)
        # for visualization
        self.optimization_log:List[OptimizationLog] = []

        K = self.data_reader.K
        self.norm_factor_2d = np.array([K[0][2], K[1][2]])[None]
    
   
    def add_keyframe(self, f_id):
        if len(self.keyframes_id) >= self.max_keyframes:
            # remove the 2nd keyframe, keep the 1st keyframe for global scale maintainance
            # self.keyframes_id.pop(1)
            self.keyframes_id.pop(0)
        self.frames[f_id].is_keyframe = True
        self.keyframes_id.append(f_id)
    
    # Get the keyframe index that is closest to the given index
    def get_closest_keyframe_id(self, f_id):
        if len(self.keyframes_id) == 0:
            return None
        return min(self.keyframes_id, key = lambda x: abs(x - f_id))
    
    def check_and_add_keyframe(self, f_id):
        """Decide if add a frame into the keyframe list
        1. Find closest keyframe
        2. Ensure not too large covisibility
        """
        if f_id in self.keyframes_id:
            return
        if f_id == 0:
            self.add_keyframe(f_id)
            print("Added 1st keyframe")
            return

        self.add_keyframe(f_id)
        return
        
        # check if enough rotation and translation
        closet_kf_id = self.get_closest_keyframe_id(f_id)
        if closet_kf_id != None:
            # check if enough rotation and translation
            T_k = self.frames[closet_kf_id].global_pose
            T_i = self.frames[f_id].global_pose
            # Camk_T_Cami = Camk_T_refObj @ refObj_T_Cami
            T_i_k = T_k @ inv(T_i)
            # get angle and translation
            angle = cv2.Rodrigues(T_i_k[:3, :3])[0]
            angle = norm(angle)
            translation = norm(T_i_k[:3, 3])
            logging.info(f"[Keyframe selection] angle: {angle}, translation: {translation}")

            ANGLE_thres = 0.1 # 0.1 radian
            TRANS_thres = 0.5 # 0.1 meter
            
            # if angle > ANGLE_thres or translation > TRANS_thres:
            if True:
                self.add_keyframe(f_id)
                logging.info(f"Add keyframe, index {self.frames[f_id].file_index}")
    
        

    def compute_covisibility(self, pose_f, pcd_f, normal_f, kf_id):
        """
        pcd_f: [N, 3]
        """
        pose_kf = self.frames[kf_id].global_pose # cam-kf_T_refObj
        Camkf_T_Camk = pose_kf @ inv(pose_f)

        num_pcd = pcd_f.shape[0]
        # transform the point cloud into new frame
        pcd_kf = pcd_f @ Camkf_T_Camk[:3, :3].T + Camkf_T_Camk[:3, 3:4].T
        # get the converted normal in new frame's perspective
        normal_in_kf = (Camkf_T_Camk[:3, :3] @ normal_f.T).T
        # get view direction
        view_dir = -pcd_kf / norm(pcd_kf, axis=-1, keepdims=True)
        # calculate the cos-angle between the view direction and the pt's normal
        noraml_ray_angle = np.sum(view_dir * normal_in_kf, axis=-1)

        visible = np.sum(noraml_ray_angle > self.cos_vis_angle)
        return visible / (num_pcd+1e-7)


    def run_BA_with_given_frame(self, BAsolver, f_id, 
        weights, loss_thres, out_dir_corres
        ):
        '''
        @description: 
            add the new frame with all keyframes to the BA
        @param:
            f_id: the index of the current frame to be added to the BA
            BAsolver: the BA solver
            weights: list of lambdas for the losses
        '''
        weights = weights.copy()
        loss_thres = loss_thres.copy()
        # 'BF' - bruct force
        # 'Near_Rot' - nearest g.t. rotation
        # 'Near_normal_orient' - 
        kf_selection_method = 'Near_Rot'

        # bruct force
        if kf_selection_method == 'BF':
            selected_frames_id = [f_id]
        
        # if True:
        if kf_selection_method == 'Near_Rot':
            # ======================= nearest rotation =======================
            selected_frames_id = [f_id]
            cur_gt_rot = self.frames[f_id].global_pose[:3, :3] # cam-f_R_baseObj
            diff_rot = {}
            for kf_id in range(f_id):
                # kf_rot = self.frames[kf_id].gt_rot # cam-kf_R_baseObj
                kf_rot = self.frames[kf_id].global_pose[:3, :3]
                R_diff = make_Rot_so3(kf_rot @ cur_gt_rot.T)
                angle = np.arccos((np.trace(R_diff) - 1) / 2)
                diff_rot[kf_id] = angle
            # in range [0, pi], lower means smaller changes in rotation
            diff_rot = dict(sorted(
                diff_rot.items(), key=lambda item: item[1], reverse=False
            ))
            for kf_id in list(diff_rot.keys()):
                selected_frames_id.append(kf_id)
                if len(selected_frames_id) >= self.max_keyframes:
                    break
            print("From Greedy Rotation", selected_frames_id)
        
        if kf_selection_method == 'Near_normal_orient':
            # ============ nearest noraml_orientation ============
            selected_frames_id = [f_id]
            # prepare data
            frame_f = self.frames[f_id]
            pose_f = frame_f.global_pose # cam-f_T_refObj
            _, mask_f, depth_f, normal_f, _ = self.data_reader.get_frame_pkg(frame_f.file_index, self.use_gtD)
            valid_mask = np.logical_and(mask_f, depth_f > 0.05)
            # get pcd 
            yy, xx = np.meshgrid(
                np.arange(self.data_reader.H),
                np.arange(self.data_reader.W), indexing='ij'
                )
            K = self.data_reader.K
            fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
            scale_f = frame_f.scaleAndShift
            # Z = scale_f[0] * depth_f[valid_mask] + scale_f[1]
            Z = depth_f[valid_mask]
            X = (xx[valid_mask] - cx) / fx * Z
            Y = (yy[valid_mask] - cy) / fy * Z
            # construct pcd in f
            pcd_f = np.stack([X, Y, Z], axis=-1)
            normal_f = normal_f[valid_mask]

            visible_score = {}
            for kf_id in range(0, f_id):
                # key_frame = self.frames[kf_id]
                vis_ratio = self.compute_covisibility(pose_f, pcd_f, normal_f, kf_id)
                visible_score[kf_id] = vis_ratio

            # in range [0, 1], larger ratio means more visible in the kf viewing
            visible_score = dict(sorted(
                visible_score.items(), key=lambda item: item[1], reverse=True
            ))
            for kf_id in list(visible_score.keys()):
                selected_frames_id.append(kf_id)
                if len(selected_frames_id) >= self.max_keyframes:
                    break
            print("From near normal-orient", selected_frames_id)


        selected_frames_id.sort()
        logging.info(f"[BA] Selected frames: {selected_frames_id}")

        # *********** ablation study ************
        # for id in selected_frames_id:
        #     self.frames[id].set_scaleAndShift(1.0, 0.0)
        # ***************************************

        # configs for Poselib PnP
        use_PnP_filter = True
        K = self.data_reader.K
        camera = {
            'model': 'PINHOLE', 
            'width': self.data_reader.W, 'height': self.data_reader.H, 
            'params': [K[0][0], K[1][1], K[0][2], K[1][2]]
        }
        ransac_opt = poselib.RansacOptions()
        thres_pnp_reproj = 6.0
        thres_pnp_3d_dist = 0.5 # 0.3
        filter_large_depth_diff = False
        ransac_opt['max_reproj_error'] = thres_pnp_reproj
        thre_ratio_inlier = 0.30

        # import networkx as nx
        # G = nx.Graph()
        graph_dir = f"{LOG_DIR}/{self.data_reader.seq_name}/BA_graphs"
        BA_debug_dir = f"{LOG_DIR}/{self.data_reader.seq_name}/BA_kfs_vis/{f_id}"
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        if not os.path.exists(BA_debug_dir):
            os.makedirs(BA_debug_dir)

        draw_BA_before = False

        # for visualization and debug
        kpts_pairs_dict = {}
        for i in range(len(selected_frames_id)):
            for j in range(i+1, len(selected_frames_id)):
                f_id0 = selected_frames_id[i]
                f_id1 = selected_frames_id[j]
                frame0:Frame = self.frames[f_id0]
                frame1:Frame = self.frames[f_id1]
                
                # ============================= Get kpt matches from LoFTR =============================
                corres_path = f"{out_dir_corres}/{frame0.f_n}_{frame1.f_n}.txt"
                success_match, kpts0, kpts1 = self.detector.match_features(
                    corres_path, frame0.rgb, frame1.rgb, frame0.mask, 
                    frame1.mask, frame0.depth, frame1.depth
                )
                kpts_pairs_dict[(f_id0, f_id1)] = {
                    "match_flag": success_match,
                    "kpts0": kpts0, "kpts1": kpts1,
                }
                # logging.info(f"*********** Frame {f_id0}-{f_id1} (file {file_idx0}-{file_idx1}) ***********")
                # logging.info(f"#valid kpts: {len(kpts0)}")

        # ======================== Batch process the kpt matches ====================
        # image0_pairs = []
        # image1_pairs = []
        # mask0_pairs = []
        # mask1_pairs = []
        # depth0_pairs = []
        # depth1_pairs = []
        # b_i = 0

        # for i in range(len(selected_frames_id)):
        #     for j in range(i+1, len(selected_frames_id)):
        #         f_id0 = selected_frames_id[i]
        #         f_id1 = selected_frames_id[j]
        #         frame0 = self.frames[f_id0]
        #         frame1 = self.frames[f_id1]

        #         image0_pairs.append(frame0.rgb)
        #         image1_pairs.append(frame1.rgb)
        #         mask0_pairs.append(frame0.mask)
        #         mask1_pairs.append(frame1.mask)
        #         depth0_pairs.append(frame0.depth)
        #         depth1_pairs.append(frame1.depth)

        #         kpts_pairs_dict[(f_id0, f_id1)] = {
        #             "batch_id": b_i, 
        #             "corres_path": f"{out_dir_corres}/{frame0.f_n}_{frame1.f_n}.txt", 
        #         }
        #         b_i += 1

        # # run prediction and save the kpts into storage
        # kpts_pairs_dict = self.detector.match_features_batch(
        #     kpts_pairs_dict, image0_pairs, image0_pairs, 
        #     mask0_pairs, mask1_pairs, depth0_pairs, depth1_pairs
        # )
        # ========================================================================  


        # for visualization and debug
        for i in range(len(selected_frames_id)):
            for j in range(i+1, len(selected_frames_id)):
                f_id0 = selected_frames_id[i]
                f_id1 = selected_frames_id[j]
                frame0:Frame = self.frames[f_id0]
                frame1:Frame = self.frames[f_id1]


                pair_dict = kpts_pairs_dict[(f_id0, f_id1)]
                success_match = pair_dict["match_flag"]
                kpts0 = pair_dict["kpts0"]
                kpts1 = pair_dict["kpts1"]
                if not success_match or len(kpts0) < self.kf_min_matches:
                    logging.warning(f"[Skip] NOT enough matched feature between kfs.")
                    continue
                # logging.info(f"*********** Frame {f_id0}-{f_id1} ***********")
                # logging.info(f"#valid kpts: {len(kpts0)}")

                depth0 = frame0.depth
                depth1 = frame1.depth

                # =================== UNCHANGE ===================
                
                depth0_obj = depth0[kpts0[:, 1], kpts0[:, 0]].reshape(-1, 1)
                depth1_obj = depth1[kpts1[:, 1], kpts1[:, 0]].reshape(-1, 1)
                kpts0 = kpts0.astype('float32')
                kpts1 = kpts1.astype('float32')

                thres_d0 = np.mean(depth0_obj) + 3.0* np.std(depth0_obj)
                thres_d1 = np.mean(depth1_obj) + 3.0* np.std(depth1_obj)
                # logging.info(f"thres-d0: {thres_d0:.4f}, thres-d1: {thres_d1:.4f}")

                # drop kpts with noisy depth value
                depth_mask = np.logical_and(
                    depth0_obj < thres_d0, depth1_obj < thres_d1
                    )[:, 0].copy()
                kpts0 = kpts0[depth_mask]
                kpts1 = kpts1[depth_mask]
                depth0_obj = depth0_obj[depth_mask]
                depth1_obj = depth1_obj[depth_mask]

                if use_PnP_filter:
                    # ======================= robust bootstarp =======================
                    kpts0_homo = np.vstack((kpts0.T, np.ones(kpts0.shape[0]))) # [3, N]
                    # TODO try to see whether scale factor influence the result
                    kpts0_unit = inv(K) @ kpts0_homo
                    P3d_0 = kpts0_unit.T * (1.0 * depth0_obj + 0.0) # [N, 3] * [N, 1]
                    # kpts1_homo = np.vstack((kpts1.T, np.ones(kpts1.shape[0])))
                    # kpts1_unit = inv(K) @ kpts1_homo
                    # P3d_1 = kpts1_unit.T * (1.0 * depth1_obj + 0.0)

                    # maps from the world system into the camera system
                    pose_0to1_pnp, stats = poselib.estimate_absolute_pose(
                        kpts1, P3d_0, camera, ransac_opt, {})
                    inlier_mask = np.array(stats['inliers'])

                    # P3d_0_to1 = P3d_0 @ pose_0to1_pnp.R.T + pose_0to1_pnp.t[None]
                    # if filter_large_depth_diff:
                    #     p3d_inliers = (norm(P3d_0_to1-P3d_1, axis=-1) < thres_pnp_3d_dist)
                    #     inlier_mask = np.logical_and(inlier_mask, p3d_inliers)

                    ratio_inlier = (inlier_mask.sum() / kpts1.shape[0])
                    # logging.info(f"Ratio-inlier after robust pnp: {ratio_inlier:.2f}")
                    
                    if ratio_inlier < (thre_ratio_inlier):
                        continue

                # ========================================================================
                uvds0 = np.hstack([kpts0, depth0_obj])
                uvds1 = np.hstack([kpts1, depth1_obj])
                if use_PnP_filter:
                    uvds0 = uvds0[inlier_mask]
                    uvds1 = uvds1[inlier_mask]

                # self.eval_errors(f_id0, f_id1, uvds0, uvds1, weights)
                # if 3d_err > 0.1:
                    # logging.info(f"error too big, skip")
                    # continue

                # ======================= BA =======================
                fix_ab = False
                if self.use_gtD:
                    fix_ab = True

                # # ************* test PnP + no Scale Optimization *************
                # fix_ab = True
                
                BAsolver.addAnEdgeGlobal(
                    fix_ab, (i==0), uvds0, uvds1, K, weights, loss_thres, 
                    frame0.global_q.elements, frame0.global_t, 
                    frame1.global_q.elements, frame1.global_t, 
                    frame0.scaleAndShift, frame1.scaleAndShift
                    )
                
                # logging.info(f">>>>> Connection between Frame {f_id0}-{f_id1} added!")
                # _feature_pair = FeaturePair(f_id0, f_id1, uvds0, uvds1)
                # feature_pairs_list.append(_feature_pair)
                # feature_pairs_dict[(f_id0, f_id1)] = _feature_pair

                # node0 = f"{f_id0}"
                # node1 = f"{f_id1}"
                # G.add_edge(node0, node1)
        
        
        # logging.info("Before BA") # dense log
        # for id in selected_frames_idx:
        #     logging.info(f"Frame: {self.frames[id]}")
        
        # ======================== Optimization ========================
        frames_before = {id:deepcopy(self.frames[id]) for id in selected_frames_id}
        BAsolver.solve(MAX_ITER_BA)
        frames_after = {id:deepcopy(self.frames[id]) for id in selected_frames_id}

        # add to opt log
        # self.optimization_log.append(
        #     OptimizationLog(feature_pairs_dict, frames_before, frames_after))
        # visualize the plot
        # self.visualize_all_edges(feature_pairs_list, filename="after_BA")
        
        # logging.info("After BA") # dense log
        # for id in selected_frames_idx:
        #     logging.info(f"Frame: {self.frames[id]}")

        BAsolver.reset()

        # fixup_pose_after_optimization
        for kf_id in selected_frames_id:
            self.frames[kf_id].update_global_pose()
        self.frames[f_id].update_global_pose()
            




    def uvds_to_P3ds(self, uvds, K, scale, shift):
        """Convert list of uvd array to point cloud
        Input:
        - uvds: [N, 3] list of the piexl of each keypoint and its given depth
        """
        P3ds = uvds.copy()
        P3ds[:, 2] = scale * P3ds[:, 2] + shift
        P3ds[:, :2] *= P3ds[:, 2].reshape(-1, 1)
        P3ds = P3ds @ inv(K).T
        return P3ds
    

    def get_transformed_2D_3D(self, f_id0, f_id1, uvds0, uvds1):
        """For visulization and evaluation"""
        uvds0 = uvds0.copy()
        uvds1 = uvds1.copy()
        frame0:Frame = deepcopy(self.frames[f_id0])
        frame1:Frame = deepcopy(self.frames[f_id1])

        K = self.data_reader.K
        q0 = frame0.global_q
        t0 = frame0.global_t
        q1 = frame1.global_q
        t1 = frame1.global_t
        scaleAndShift0 = frame0.scaleAndShift
        scaleAndShift1 = frame1.scaleAndShift
        
        # 1. calculate the relative pose
        q0_inv = q0.inverse
        q_0to1 = q1 * q0_inv
        t_0in1 = q_0to1.rotate(t0)
        t_0to1 = t1 - t_0in1
        # 2. get 3d point clouds
        p3d_0 = self.uvds_to_P3ds(uvds0, K, scaleAndShift0[0], scaleAndShift0[1])
        p3d_1 = self.uvds_to_P3ds(uvds1, K, scaleAndShift1[0], scaleAndShift1[1])
        # 3. transform the 3d points to the second frame
        p3d_0to1 = np.apply_along_axis(lambda x: q_0to1.rotate(x) + t_0to1, 1, p3d_0)

        p2d_1 = uvds1[:, :2]
        p2d_0to1 = p3d_0to1 @ K.T
        p2d_0to1 = p2d_0to1[:, :2] / p2d_0to1[:, 2:3]
        return p2d_1, p2d_0to1, p3d_1, p3d_0to1



    def eval_errors(self, f_id0, f_id1, uvds0, uvds1, weigths):
        uvds0 = uvds0.copy()
        uvds1 = uvds1.copy()
        p2d_1, p2d_0to1, p3d_1, p3d_0to1 = self.get_transformed_2D_3D(f_id0, f_id1, uvds0, uvds1)

        # / p3d_1[:, 2:3]
        spatial_err = np.sum((weigths[0] * (p3d_1 - p3d_0to1)) **2, axis=-1).mean()
        disparity_err = np.mean(weigths[1] * ((1.0/p3d_1[:,2]) - (1.0/p3d_0to1[:,2])) **2)
        reproj_err = np.sum((weigths[2] * (p2d_1 - p2d_0to1) / self.norm_factor_2d )**2, axis=-1).mean()
        
        logging.info(
        f"spatial:{spatial_err:.6f}, disparity:{disparity_err:.6f}, reproj:{reproj_err:.6f}")

        return [spatial_err, disparity_err, reproj_err]


    def visualize_debug(self, f_id0, f_id1, uvds0, uvds1, filename, vis_folder):
        """visualize the 2D and 3D correspondences"""
        uvds0 = uvds0.copy()
        uvds1 = uvds1.copy()

        vis_folder = os.path.join(vis_folder, f"{f_id0}_{f_id1}")
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder)
        file_path = os.path.join(vis_folder, filename)

        p2d_1, p2d_0to1, p3d_1, p3d_0to1 = self.get_transformed_2D_3D(f_id0, f_id1, uvds0, uvds1)
        
        fig = plt.figure(figsize=(10, 8))
        fig.suptitle(f"{f_id0}-{f_id1}")
        img1 = self.data_reader.get_color(self.frames[f_id1].file_index)
    
        # visualize p3d_1 and p3d_0to1 in the second frame
        err_3d = norm(p3d_1 - p3d_0to1, axis=-1).mean()
        err_disparity = np.abs((1.0/p3d_1[:,2]) - (1.0/p3d_0to1[:,2])).mean()
        ax = plt.subplot(121, projection='3d')
        ax.set_title(f'3d-error: {err_3d:.4f} | d-error {err_disparity:.4f}')
        # draw the 3d points
        ax.scatter(p3d_1[:, 0], p3d_1[:, 1], p3d_1[:, 2],
                    c = 'r', label = '3d points in frame 1')
        ax.scatter(p3d_0to1[:, 0], p3d_0to1[:, 1], p3d_0to1[:, 2],
                    c = 'g', label = '3d points transformed to frame 1')
        ax.set_aspect("equal")
        
        # visualize projected p3d_0to1
        err_2d = norm((p2d_1 - p2d_0to1), axis=-1).mean()
        ax = plt.subplot(122)
        ax.set_title(f'2d-error: {err_2d:.4f}')
        ax.imshow(img1)
        ax.scatter(p2d_1[:, 0], p2d_1[:, 1], 
                   c = 'r', label = '2d points in frame 1')
        ax.scatter(p2d_0to1[:, 0], p2d_0to1[:, 1], 
                   c = 'g', label = '2d points re-projected to frame 1')
        # crop the image to fit correspondences
        valid_x_range = (np.min(p2d_1[:, 0])-10, np.max(p2d_1[:, 0])+10)
        valid_y_range = (np.min(p2d_1[:, 1])-10, np.max(p2d_1[:, 1])+10)
        ax.set_xlim(valid_x_range)
        ax.set_ylim(valid_y_range)
        ax.invert_yaxis()

        plt.savefig(f"{file_path}.png")
        plt.close()
    

    def visualize_all_edges(self, feature_pairs_list, filename, vis_folder = "/home/pop/peter_ws/CVG_proj/BundleSDF/PnPtest/err_vis"):
        for feature_pair in feature_pairs_list:
            assert isinstance(feature_pair, FeaturePair)
            self.visualize_debug(
                feature_pair.f_id0, feature_pair.f_id1, 
                feature_pair.uvds0, feature_pair.uvds1,
                filename, vis_folder
                )




    def visualize_optimization_log(self):
        # from input get opt_id, f_id0, f_id1
        print("select an optimization log from the following list:")
        for i, opt_log in enumerate(self.optimization_log):
            print(f"Optimization log {i}")
            print(f"selected frames: {opt_log.selected_frames_id}")
        
        opt_id = int(input("Enter the optimization log id: "))
        assert opt_id < len(self.optimization_log)
        opt_log : OptimizationLog = self.optimization_log[opt_id]

        feature_pairs_dict = opt_log.feature_pairs_dict
        frames_before = opt_log.frames_before
        frames_after = opt_log.frames_after

        # draw the nx graph
        import networkx as nx
        G = nx.Graph()
        for feature_pair in feature_pairs_dict.values():
            scales0 = opt_log.frames_after[feature_pair.id0].scaleAndShift
            scales1 = opt_log.frames_after[feature_pair.id1].scaleAndShift
            node0 = f"{feature_pair.f_id0}\na{scales0[0]:.2f}\nb{scales0[1]:.3f}"
            node1 = f"{feature_pair.f_id1}\na{scales1[0]:.2f}\nb{scales1[1]:.3f}"
            G.add_edge(node0, node1)
        plt.figure(0)
        nx.draw(G, with_labels=True, node_size= 2000, font_size = 10, font_color = 'white')
        plt.ion()
        plt.show()

        def find_closest_frame_id(f_id):
            return min(frames_before.keys(), key = lambda x: abs(x - f_id))
        def _get_T_and_scaleAndShift(f_id, is_before = True):
            if is_before:
                _frames = frames_before
            else:
                _frames = frames_after

            if f_id in opt_log.selected_frames_id:
                q0 = _frames[f_id].global_q
                t0 = _frames[f_id].global_t
                T = np.eye(4)
                T[:3, :3] = q0.rotation_matrix
                T[:3, 3] = t0
                scaleAndShift = _frames[f_id].scaleAndShift
            else:
                scaleAndShift = self.frames[f_id].scaleAndShift
                closest_selected_frame_id = find_closest_frame_id(f_id)
                q0_base = _frames[closest_selected_frame_id].global_q
                t0_base = _frames[closest_selected_frame_id].global_t
                # q t to 4x4 matrix
                T_base = np.eye(4)
                T_base[:3, :3] = q0_base.rotation_matrix
                T_base[:3, 3] = t0_base
                if f_id < closest_selected_frame_id:
                    # if the frame is before the closest selected frame
                    for i in range(closest_selected_frame_id, f_id, -1):
                        l_pose = _frames[i].local_pose
                        # l_pose if from i to i+1, now we need to get the inverse
                        l_pose_inv = inv(l_pose)
                        T = l_pose_inv @ T_base
                else:
                    # if the frame is after the closest selected frame
                    for i in range(closest_selected_frame_id+1, f_id+1):
                        l_pose = _frames[i].local_pose
                        T = l_pose @ T_base
            return T, scaleAndShift



        while True:
            # get the frame id0 and frame id1 together
            f_id0, f_id1 = map(int, input("Enter the frame id 0 & 1: ").split())

            file_index0 = self.frames[f_id0].file_index
            file_index1 = self.frames[f_id1].file_index
            img0 = self.data_reader.get_color(file_index0)
            img1 = self.data_reader.get_color(file_index1)

            K = self.data_reader.K
            # 4 subplots, 2 rows, 2 columns
            fig = plt.figure(1, figsize=(12, 20))
            fig.suptitle(f"{f_id0}-{f_id1} ({file_index0}-{file_index1})")

            # ==================== before BA ====================
            T0_before, s_factor0_before = _get_T_and_scaleAndShift(f_id0)
            T1_before, s_factor1_before = _get_T_and_scaleAndShift(f_id1)
            if f_id0 in frames_before.keys() and f_id1 in frames_before.keys():
                uvds0 = self.optimization_log[opt_id].feature_pairs_dict[(f_id0, f_id1)].uvds0
                uvds1 = self.optimization_log[opt_id].feature_pairs_dict[(f_id0, f_id1)].uvds1
                assert len(uvds0) == len(uvds1)

                # 1. 3d points in the first frame
                p3d_0 = self.uvds_to_P3ds(uvds0, K, s_factor0_before[0], s_factor0_before[1])
                p3d_1 = self.uvds_to_P3ds(uvds1, K, s_factor1_before[0], s_factor1_before[1])
                # 3. reproject the 3d points to the second frame
                T_0to1_before = T1_before @ inv(T0_before)
                p3d_0to1_before = np.apply_along_axis(
                    lambda x: T_0to1_before[:3, :3] @ x + T_0to1_before[:3, 3], 1, p3d_0)
                err_3d_before = norm(p3d_1 - p3d_0to1_before, axis=1).mean()
                
                # 5. visualize p3d_1 and p3d_0to1 in the second frame
                ax = plt.subplot(321, projection='3d')
                ax.set_title(f'[before] 3d error: {err_3d_before:.6f}')
                ax.scatter(p3d_1[:, 0], p3d_1[:, 1], p3d_1[:, 2], 
                           c = 'r', label = '3d points in frame 1')
                ax.scatter(p3d_0to1_before[:, 0], p3d_0to1_before[:, 1], p3d_0to1_before[:, 2], 
                           c = 'g', label = '3d points transformed to frame 1')
                ax.set_aspect("equal")
            
                # 6. visualize projected p3d_0to1
                p2d_0to1_before = p3d_0to1_before @ K.T
                p2d_0to1_before = p2d_0to1_before[:, :2] / p2d_0to1_before[:, 2:3]
                p2d_1 = uvds1[:, :2]
                err_2d_before = norm(p2d_1 - p2d_0to1_before, axis=1).mean()

                ax = plt.subplot(322)
                ax.set_title(f'[before] 2d error: {err_2d_before:.6f}')
                ax.imshow(img1)
                ax.scatter(p2d_1[:, 0], p2d_1[:, 1], 
                           c = 'r', label = '2d points in frame 1')
                ax.scatter(p2d_0to1_before[:, 0], p2d_0to1_before[:, 1], 
                           c = 'g', label = '2d points re-projected to frame 1')
                # crop the image to fit correspondences
                valid_x_range = (np.min(p2d_1[:, 0]), np.max(p2d_1[:, 0]))
                valid_y_range = (np.min(p2d_1[:, 1]), np.max(p2d_1[:, 1]))
                ax.set_xlim(valid_x_range)
                ax.set_ylim(valid_y_range)
        
            #### ==================== after BA ====================
            T0_after, s_factor0_after = _get_T_and_scaleAndShift(f_id0, is_before = False)
            T1_after, s_factor1_after = _get_T_and_scaleAndShift(f_id1, is_before = False)
            if f_id0 in frames_after.keys() and f_id1 in frames_after.keys():
                uvds0 = self.optimization_log[opt_id].feature_pairs_dict[(f_id0, f_id1)].uvds0
                uvds1 = self.optimization_log[opt_id].feature_pairs_dict[(f_id0, f_id1)].uvds1
                assert len(uvds0) == len(uvds1)

                # 1. 3d points in the first frame
                p3d_0 = self.uvds_to_P3ds(uvds0, K, s_factor0_after[0], s_factor0_after[1])
                p3d_1 = self.uvds_to_P3ds(uvds1, K, s_factor1_after[0], s_factor1_after[1])
                # 3. reproject the 3d points to the second frame
                T_0to1_after = T1_after @ inv(T0_after)
                p3d_0to1_after = np.apply_along_axis(lambda x: T_0to1_after[:3, :3] @ x + T_0to1_after[:3, 3], 1, p3d_0)
                reproj_error_after = norm(p3d_1 - p3d_0to1_after, axis=1).mean()
                
                # 5. visualize p3d_1 and p3d_0to1 in the second frame
                ax = plt.subplot(323, projection='3d')
                ax.set_title(f'[after] 3d error: {reproj_error_after:.6f}')
                ax.scatter(p3d_1[:, 0], p3d_1[:, 1], p3d_1[:, 2], 
                           c = 'r', label = '3d points in frame 1')
                ax.scatter(p3d_0to1_after[:, 0], p3d_0to1_after[:, 1], p3d_0to1_after[:, 2], 
                           c = 'g', label = '3d points transformed to frame 1')
                ax.set_aspect("equal")

                # 6. project p3d_0to1 to the image plane
                p2d_0to1_after = p3d_0to1_after @ K.T
                p2d_0to1_after = p2d_0to1_after[:, :2] / p2d_0to1_after[:, 2:3]
                p2d_1 = uvds1[:, :2]
                err_2d_after = norm(p2d_1 - p2d_0to1_after, axis=1).mean()

                ax = plt.subplot(324)
                ax.set_title(f'[after] 2d error: {err_2d_after:.6f}')
                ax.imshow(img1)
                ax.scatter(p2d_1[:, 0], p2d_1[:, 1], 
                           c = 'r', label = '2d points in frame 1')
                ax.scatter(p2d_0to1_after[:, 0], p2d_0to1_after[:, 1], 
                           c = 'g', label = '2d points re-projected to frame 1')
                # crop the image to fit correspondences
                valid_x_range = (np.min(p2d_1[:, 0]), np.max(p2d_1[:, 0]))
                valid_y_range = (np.min(p2d_1[:, 1]), np.max(p2d_1[:, 1]))
                ax.set_xlim(valid_x_range)
                ax.set_ylim(valid_y_range)


                # ==================== correspondences with line ==================
                ax = plt.subplot(313)
                # 1. concatenate the 2 images
                img0_mask = self.data_reader.get_mask(file_index0)
                img1_mask = self.data_reader.get_mask(file_index1)
                # corp the images to fit the correspondences
                padding = 20
                # mask0 min x, max x, min y, max y
                x_mask_range = np.where(img0_mask.any(axis = 0))
                valid_x_range0 = (max(0, x_mask_range[0][0] - padding), 
                                  min(img0.shape[1], x_mask_range[0][-1] + padding))
                y_mask_range = np.where(img0_mask.any(axis = 1))
                valid_y_range0 = (max(0, y_mask_range[0][0] - padding), 
                                  min(img0.shape[0], y_mask_range[0][-1] + padding))
                
                # mask1 min x, max x, min y, max y
                x_mask_range = np.where(img1_mask.any(axis = 0))
                valid_x_range1 = (max(0, x_mask_range[0][0] - padding), 
                                  min(img1.shape[1], x_mask_range[0][-1] + padding))
                y_mask_range = np.where(img1_mask.any(axis = 1))
                valid_y_range1 = (max(0, y_mask_range[0][0] - padding), 
                                  min(img1.shape[0], y_mask_range[0][-1] + padding))
                # merge the 2 ranges
                valid_x_range = (min(valid_x_range0[0], valid_x_range1[0]), 
                                 max(valid_x_range0[1], valid_x_range1[1]))
                valid_y_range = (min(valid_y_range0[0], valid_y_range1[0]), 
                                 max(valid_y_range0[1], valid_y_range1[1]))
                img0_crop = img0[valid_y_range[0]:valid_y_range[1], 
                                 valid_x_range[0]:valid_x_range[1]]
                img1_crop = img1[valid_y_range[0]:valid_y_range[1], 
                                 valid_x_range[0]:valid_x_range[1]]
                
                img_horizontal = np.hstack([img0_crop, img1_crop])
                ax.imshow(img_horizontal)
                ax.scatter(uvds0[:, 0] - valid_x_range[0], uvds0[:, 1] - valid_y_range[0], 
                           c = 'r', label = '2d points in frame 0', marker = 'x')
                ax.scatter(uvds1[:, 0] - valid_x_range[0] + img0_crop.shape[1], uvds1[:, 1] - valid_y_range[0], 
                           c = 'b', label = '2d points in frame 1', marker = 'x')
                for i in range(len(uvds0)):
                    ax.plot(
                        [uvds0[i, 0] - valid_x_range[0], uvds1[i, 0] - valid_x_range[0] + img0_crop.shape[1]], 
                        [uvds0[i, 1] - valid_y_range[0], uvds1[i, 1] - valid_y_range[0]], 
                        linewidth = 0.5
                        )
                
            plt.show()
            quit = input("Press q to quit, or press Enter to continue...")
            fig = plt.figure(1)
            fig.clf()
            if quit == 'q':
                exit()
        
