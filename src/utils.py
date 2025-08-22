import os, sys, logging, glob, time
import cv2, imageio, trimesh
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import repeat

import numpy as np
from numpy.linalg import inv, norm
import torch, torchvision
from scipy.optimize import minimize

code_dir = os.path.dirname(os.path.realpath(__file__))

# self packages
sys.path.append("..")
from third_party.LoFTR.src.loftr import *



def get_sift_feature(rgb, mask):
    mask_img = rgb * mask[..., None]
    gray = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    H, W = mask.shape

    sift = cv2.SIFT_create()
    patch_r = 16
    patch_size = 2* patch_r

    desc_list = []
    for y in range(patch_r, H-patch_r, patch_size):
        for x in range(patch_r, W-patch_r, patch_size):
            patch_mask = mask[y-patch_r:y+patch_r, x-patch_r:x+patch_r]
            # less than one eighth of the total pixels
            if np.sum(patch_mask) < 32:
                continue

            # compute SIFT descroptor
            keypoints = [cv2.KeyPoint(8, 8, patch_size)]
            patch = gray[y-patch_r:y+patch_r, x-patch_r:x+patch_r]
            _, descriptors = sift.compute(patch, keypoints)
            descriptors = np.array(descriptors[0]).reshape((16, 8))

            for block_idx in range(16):
                x0 = 4* int(block_idx % 4)
                y0 = 4* int(block_idx / 4)
                # black = patch[y0:y0+4, x0:x0+4]
                block_mask = patch_mask[y0:y0+4, x0:x0+4]
                # only add the histogram in block with valid mask > 50%
                if np.sum(block_mask) > 8:
                    desc_list.append(descriptors[block_idx])

    desc_list = np.array(desc_list).mean(axis=0)
    return desc_list


def compute_cos_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / norm(vec_a) / norm(vec_b)

def compute_Chi_squared_distance(hist1, hist2):
    return np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-10))

# ============================== LOFTR ==============================
class LoftrRunner:
    def __init__(self):
        default_cfg["match_coarse"]["thr"] = 0.2
        print("\nLoftr default config: \n", default_cfg)
        self.matcher = LoFTR(config=default_cfg)

        loftr_ckpt_path = f"{code_dir}/../third_party/LoFTR/weights/outdoor_ds.ckpt"
        self.matcher.load_state_dict(torch.load(loftr_ckpt_path)["state_dict"])

        if torch.cuda.is_available():
            self.matcher = self.matcher.eval().cuda()
        else:
            self.matcher = self.matcher.eval().cpu()

        self.time_record = []
    
    @torch.no_grad()
    def predict_single(self, image0: np.ndarray, image1: np.ndarray):
        """
        Input:
            [B, H, W, C]
        Return:
            [Num_corres, 5] compose of [kpt_in_img0, kpt_in_img1, confindence]
        """
        # transfer to tensor and to grayscale [B, 1, H, W]
        if torch.cuda.is_available():
            image0 = torch.from_numpy(image0).permute(0, 3, 1, 2).float().cuda()
            image1 = torch.from_numpy(image1).permute(0, 3, 1, 2).float().cuda()
        else:
            image0 = torch.from_numpy(image0).permute(0, 3, 1, 2).float().cpu()
            image1 = torch.from_numpy(image1).permute(0, 3, 1, 2).float().cpu()

        if image0.shape[-3] == 3:
            image0 = torchvision.transforms.functional.rgb_to_grayscale(image0)
            image1 = torchvision.transforms.functional.rgb_to_grayscale(image1)
        image0 = image0 / 255.0
        image1 = image1 / 255.0

        batch = {"image0": image0, "image1": image1}
        s_t = time.time()
        with torch.no_grad():
            self.matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
        torch.cuda.synchronize() # NOTE add this when measuring time
        loftr_time = time.time() - s_t
        self.time_record.append(loftr_time)
        # logging.info(f"LoFTR took {loftr_time:.4f} sec to process a single pair")

        corres = (np.concatenate(
                (mkpts0.reshape(-1, 2), mkpts1.reshape(-1, 2), mconf.reshape(-1, 1)),
                axis=-1,
            ).reshape(-1, 5).astype(np.float32))
        
        # clean cache
        del batch, image0, image1
        torch.cuda.empty_cache()

        return corres
    
    @torch.no_grad()
    def predict(self, rgbAs: np.ndarray, rgbBs: np.ndarray):
        """
        @rgbAs: (N,H,W,C)
        """
        image0 = torch.from_numpy(rgbAs).permute(0, 3, 1, 2).float().cuda()
        image1 = torch.from_numpy(rgbBs).permute(0, 3, 1, 2).float().cuda()
        if image0.shape[-3] == 3:
            image0 = torchvision.transforms.functional.rgb_to_grayscale(image0)
            image1 = torchvision.transforms.functional.rgb_to_grayscale(image1)
        image0 = image0 / 255.0
        image1 = image1 / 255.0
        last_data = {"image0": image0, "image1": image1}

        batch_size = 3 # NOTE!!! change this to resolve VMEN problem
        ret_keys = ["mkpts0_f", "mkpts1_f", "mconf", "m_bids"]
        with torch.cuda.amp.autocast(enabled=True):
            i_b = 0
            for b in range(0, len(last_data["image0"]), batch_size):
                img_pair_batch = {
                    "image0": last_data["image0"][b : b + batch_size],
                    "image1": last_data["image1"][b : b + batch_size],
                }
                s_t = time.time()
                with torch.no_grad():
                    self.matcher(img_pair_batch)
                torch.cuda.synchronize() # NOTE add this when measuring time
                loftr_time = time.time() - s_t
                self.time_record.append(loftr_time)
                # logging.info(f"LoFTR took {loftr_time:.4f} sec to process a single pair")
                img_pair_batch["m_bids"] += i_b
                for k in ret_keys:
                    if k not in last_data:
                        last_data[k] = []
                    last_data[k].append(img_pair_batch[k])
                i_b += len(img_pair_batch["image0"])

        for k in ret_keys:
            last_data[k] = torch.cat(last_data[k], dim=0)

        total_n_matches = len(last_data["mkpts0_f"])
        mkpts0 = last_data["mkpts0_f"].cpu().numpy()
        mkpts1 = last_data["mkpts1_f"].cpu().numpy()
        mconf = last_data["mconf"].cpu().numpy()
        pair_ids = last_data["m_bids"].cpu().numpy()

        corres = (np.concatenate(
                (mkpts0.reshape(-1, 2), mkpts1.reshape(-1, 2), mconf.reshape(-1, 1)),
                axis=-1,
            ).reshape(-1, 5).astype(np.float32))
        corres_tmp = []
        for i in range(len(rgbAs)):
            cur_corres = corres[pair_ids == i]
            corres_tmp.append(cur_corres)
        corres = corres_tmp

        del last_data, image0, image1
        torch.cuda.empty_cache()

        return corres
    


    def match_features(self, corres_path, image0, image1, 
            mask0, mask1, depth0, depth1):
        if os.path.exists(corres_path):
            # logging.info(f"==> Loading LoFTR matches from file")
            # avoid empty txt file
            if os.path.getsize(corres_path) == 0:
                return False, np.empty((1,2)), np.empty((1,2))
                
            kpts_0and1 = np.loadtxt(corres_path)
            if len(kpts_0and1.shape) < 2:
                return False, np.empty((1,2)), np.empty((1,2))

            kpts0 = kpts_0and1[:,:2].astype(int)
            kpts1 = kpts_0and1[:,2:].astype(int)

            kpt_ind_valid_depth = np.logical_and(
                depth0[kpts0[:, 1], kpts0[:, 0]] > 0.05, depth1[kpts1[:, 1], kpts1[:, 0]] > 0.05)
            kpts0 = kpts0[kpt_ind_valid_depth]
            kpts1 = kpts1[kpt_ind_valid_depth]
        else:
            # run loftr
            logging.info(f"==> Running LoFTR to get matches")
            corres = self.predict_single(np.array([image0]), np.array([image1]))

            # extract the keypoints
            kpts0 = corres[:, :2].astype(int) # pixel
            kpts1 = corres[:, 2:4].astype(int) # sub-pixel

            kpt_ind_in_mask = np.logical_and(
                mask0[kpts0[:, 1], kpts0[:, 0]], mask1[kpts1[:, 1], kpts1[:, 0]])
            kpts0 = kpts0[kpt_ind_in_mask]
            kpts1 = kpts1[kpt_ind_in_mask]
            
            # save kpts0 and kpts1 in one cache file in shape [N, 4] - [u0_i, v0_i, u1_i, v1_i]
            np.savetxt(corres_path, np.hstack([kpts0, kpts1]))

            kpt_ind_valid_depth = np.logical_and(
                depth0[kpts0[:, 1], kpts0[:, 0]] > 0.05, depth1[kpts1[:, 1], kpts1[:, 0]] > 0.05)
            kpts0 = kpts0[kpt_ind_valid_depth]
            kpts1 = kpts1[kpt_ind_valid_depth]

        return True, kpts0, kpts1
    
    
    def match_features_batch(self, kpts_pairs_dict, 
            image0_b, image1_b, 
            mask0_b, mask1_b, 
            depth0_b, depth1_b
        ):
        selected_ids = []
        selected_pair = []

        for pair_n in list(kpts_pairs_dict.keys()):
            pair_dict = kpts_pairs_dict[pair_n]
            b_id = pair_dict["batch_id"]
            corres_path = pair_dict["corres_path"]

            if os.path.exists(corres_path):
                if os.path.getsize(corres_path) == 0:
                    pair_dict["match_flag"] = False
                    pair_dict["kpts0"] = np.empty((1,2))
                    pair_dict["kpts1"] = np.empty((1,2))
                    kpts_pairs_dict[pair_n] = pair_dict
                    continue

                kpts_0and1 = np.loadtxt(corres_path)
                if len(kpts_0and1.shape) < 2:
                    pair_dict["match_flag"] = False
                    pair_dict["kpts0"] = np.empty((1,2))
                    pair_dict["kpts1"] = np.empty((1,2))
                    kpts_pairs_dict[pair_n] = pair_dict
                    continue
                kpts0 = kpts_0and1[:,:2].astype(int)
                kpts1 = kpts_0and1[:,2:].astype(int)

                depth0 = depth0_b[b_id]
                depth1 = depth1_b[b_id]
                kpt_ind_valid_depth = np.logical_and(
                    depth0[kpts0[:, 1], kpts0[:, 0]] > 0.05, depth1[kpts1[:, 1], kpts1[:, 0]] > 0.05)
                kpts0 = kpts0[kpt_ind_valid_depth]
                kpts1 = kpts1[kpt_ind_valid_depth]
                pair_dict["match_flag"] = True
                pair_dict["kpts0"] = kpts0
                pair_dict["kpts1"] = kpts1
                kpts_pairs_dict[pair_n] = pair_dict
            else:
                logging.info(f"==> Need to run LoFTR for pair {pair_n}")
                selected_ids.append(b_id)
                selected_pair.append(pair_n)

        # run loftr for the remaining pairs
        # selected_ids = np.array(selected_ids)
        if len(selected_ids) > 0:
            logging.info(f"==> Running LoFTR for batches to get matches")
            corres_b = self.predict(
                np.array(image0_b)[selected_ids], np.array(image1_b)[selected_ids])

            # collect results
            for i, b_id in enumerate(selected_ids):
                corres = corres_b[i]
                pair_n = selected_pair[i]
                pair_dict = kpts_pairs_dict[pair_n]
                corres_path = pair_dict["corres_path"]

                # extract the keypoints
                kpts0 = corres[:, :2].astype(int) # pixel
                kpts1 = corres[:, 2:4].astype(int) # sub-pixel

                mask0 = mask0_b[b_id]
                mask1 = mask1_b[b_id]
                kpt_ind_in_mask = np.logical_and(
                    mask0[kpts0[:, 1], kpts0[:, 0]], mask1[kpts1[:, 1], kpts1[:, 0]])
                kpts0 = kpts0[kpt_ind_in_mask]
                kpts1 = kpts1[kpt_ind_in_mask]
            
                # save kpts0 and kpts1 in one cache file in shape [N, 4] - [u0_i, v0_i, u1_i, v1_i]
                np.savetxt(corres_path, np.hstack([kpts0, kpts1]))

                depth0 = depth0_b[b_id]
                depth1 = depth1_b[b_id]
                kpt_ind_valid_depth = np.logical_and(
                    depth0[kpts0[:, 1], kpts0[:, 0]] > 0.05, depth1[kpts1[:, 1], kpts1[:, 0]] > 0.05)
                kpts0 = kpts0[kpt_ind_valid_depth]
                kpts1 = kpts1[kpt_ind_valid_depth]

                pair_dict["match_flag"] = True
                pair_dict["kpts0"] = kpts0
                pair_dict["kpts1"] = kpts1
                kpts_pairs_dict[pair_n] = pair_dict

        return kpts_pairs_dict
    
    def get_total_time(self):
        sum_time = np.array(self.time_record).sum()
        return sum_time




# ========================= Scale Alignment =========================
def estimate_bg_scale(depth_static_ref, depth_static_tar):
    init_trans = [1.0, 0.0]
    result = minimize(
        Minimize_scale_shift, init_trans, 
        args=(depth_static_ref, depth_static_tar), 
        method='L-BFGS-B', bounds=((0.1, 5.0), (-5.0, 5.0)))
    scale_pred, shift_pred = result.x
    # print(f"Opt-scale from depth_tar to depth_ref: Scale {scale_pred:.6f}, Shift {shift_pred:.6f}")
    return scale_pred, shift_pred



# ====================================================================


def huber_loss(err, delta=1.0):
    """
    Compute the Huber loss
    Params:
    - err (numpy array): The residuals (differences between true and predicted values).
    - delta (float): The threshold at which to change between quadratic and linear loss.
    Returns:
    - numpy array: The Huber loss for each residual.
    """
    abs_err = np.abs(err)
    quadratic = np.minimum(abs_err, delta)
    linear = abs_err - quadratic
    return 0.5 * quadratic ** 2 + delta * linear

def cauchy_loss(err, sigma=1.0):
    """
    Compute the Cauchy loss
    Parameters:
    - err (numpy array): The residuals (differences between true and predicted values).
    - sigma (float): Scale parameter for Cauchy loss. Default is 1.0.

    """
    return np.log(1 + (err ** 2) / (sigma ** 2))



def Minimize_scale_shift(params, depths_ref, depths_tar):
    """
    Depths should be passed as an array
    """
    scale, shift = params
    residual_depths = depths_ref - (scale * depths_tar + shift) # depth_ref are with 1.0 scale and 0 shift
    # L1 loss
    # loss = np.mean(np.abs(residual_depths))
    # Huber loss
    loss = np.mean(huber_loss(residual_depths, delta=0.05))
    return loss





# ======================================== Visualization ========================================
def make_E_se3(E: np.ndarray) -> np.ndarray:
    """make Essential matrix valid SE(3)"""
    R = E[:3, :3].copy()
    U, _, Vt = np.linalg.svd(R)
    E[:3, :3] = U @ Vt
    return E

def make_Rot_so3(rot_mat):
    R = rot_mat.copy()
    U, _, Vt = np.linalg.svd(R)
    R_SO3 = np.dot(U, Vt) # Ensure orthogonality
    # Ensure determinant is +1
    if np.linalg.det(R_SO3) < 0:
        U[:, -1] *= -1
        R_SO3 = np.dot(U, Vt)

    return R_SO3

def draw_pose(
    reader, pose_dir, pose_vis_dir, 
    use_gt_trans=False, global_pose=True, global_scale=1.0
    ):

    os.system(f'rm -rf {pose_vis_dir} && mkdir -p {pose_vis_dir}')
    logging.info(f"Re-create folder {pose_vis_dir}")

    K = reader.K
    bbox = reader.bbox
    poses_trans = []
    poses_gt = []
    rgb_list = []
    f_name_list = []

    pred_traj = []
    gt_traj = []

    model_offset = reader.model_offset

    num_frames = len(os.listdir(pose_dir))
    for img_iter in range(num_frames):
        frame_idx = reader.map_iter_to_index(img_iter)

         # get pose offset refObj_T_baseObj - from obj_base coord. to obj_ref coord.
        if img_iter == 0:
            pose_offset = reader.get_gt_pose(frame_idx)
            # pose_offset[:3, :3] = make_Rot_so3(pose_offset[:3, :3])
            # inv_pose_offset = inv(pose_offset)

        frame_name = reader.id_strs[frame_idx]
        rgb_list.append(reader.get_color(frame_idx))
        f_name_list.append(frame_name)

        if not global_pose: # use the relative pose
            if img_iter == 0:
                pose_cur = np.eye(4)
                pose_last = pose_cur
                continue
            T_ij = np.loadtxt(f"{pose_dir}/{frame_name}.txt") # curCam_T_lastCam
            pose_cur = T_ij @ pose_last
            pose_last = pose_cur
        else: # use the global pose
            pose_cur = np.loadtxt(f"{pose_dir}/{frame_name}.txt")
        
        gt_pose = reader.get_gt_pose(frame_idx)
        gt_pose[:3, :3] = make_Rot_so3(gt_pose[:3, :3])
        
        pose_cur[:3, 3] = global_scale * pose_cur[:3, 3]
        trans_cur_pose = pose_cur @ pose_offset
        trans_cur_pose[:3, :3] = make_Rot_so3(trans_cur_pose[:3, :3])
        
        # only export correct gt poses for post-evaluation
        if not np.allclose(gt_pose, model_offset):
            gt_traj.append(gt_pose)
            pred_traj.append(trans_cur_pose)

        # add pose offset and use gt_tranlate if needed
        pose_temp = trans_cur_pose
        if use_gt_trans:
            pose_temp[:3, 3] = gt_pose[:3, 3].copy()
        poses_trans.append(pose_temp)
        poses_gt.append(gt_pose)

    num_poses = len(poses_trans)
    logging.info(f"Num poses: {num_poses}")
    
    import multiprocessing as mp
    idx_s = 0
    with tqdm(total=num_poses) as pbar:
        while(idx_s < num_poses):
            idx_e = min(num_poses, idx_s + 50)
            with mp.Pool(mp.cpu_count()-2) as pool:
                pool.starmap(vis_pose, zip(
                    rgb_list[idx_s:idx_e], poses_trans[idx_s:idx_e], poses_gt[idx_s:idx_e], 
                    repeat(K), repeat(bbox), repeat(pose_vis_dir), f_name_list[idx_s:idx_e]
                    ))
            pbar.update(idx_e - idx_s)
            idx_s = idx_e

    return np.array(pred_traj), np.array(gt_traj)



def save_obj_as_pickle(obj, filename):
    import pickle
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    logging.info(f"Save the KeyFrameManager object to {filename}")


def vis_tracking(num_poses, traj_list, out_dir, file_name):
    CAM_SIZE = 5e-3
    BAR_LEN = 8e-3

    def draw_camera(ax, W_T_Cam, size, color='blue'):
        """[R|t]: Transformation Mat from Cam to World"""
        # Create cam_pts for camera visualization (a simple pyramid)
        cam_pts = size * np.array([
            [0, 0, 0], [1, 1, 1], [1, -1, 1], 
            [0, 0, 0], [-1, -1, 1], [-1, 1, 1], [1, 1, 1], [-1, 1, 1], 
            [0, 0, 0], [-1, -1, 1], [1, -1, 1]
        ]).T
        cam_pts = W_T_Cam[:3, :3] @ cam_pts + W_T_Cam[:3, 3:4] # [3, N]
        # Draw camera
        ax.plot(cam_pts[0,:], cam_pts[1,:], cam_pts[2,:], c=color)

    def draw_xyzbar(ax, Cam_T_Obj, size):
        """[R|t]: Transformation Mat from Obj frame to Cam"""
        coord_bar = size * np.array([
            [0, 0, 0], [1, 0, 0], 
            [0, 0, 0], [0, 1, 0],
            [0, 0, 0], [0, 0, 1],
        ]).T
        coord_bar = Cam_T_Obj[:3, :3] @ coord_bar + Cam_T_Obj[:3, 3:4] # [3, N]
        # Draw camera
        ax.plot(coord_bar[0,0:2], coord_bar[1,0:2], coord_bar[2,0:2], c='red', lw=2)
        ax.plot(coord_bar[0,2:4], coord_bar[1,2:4], coord_bar[2,2:4], c='green', lw=2)
        ax.plot(coord_bar[0,4:6], coord_bar[1,4:6], coord_bar[2,4:6], c='blue', lw=2)

    fig = plt.figure(0, figsize=(10, 10))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    
    for traj in traj_list:
        traj_np = traj['traj_np'] # [N, 4, 4]
        traj_name = traj['name']
        traj_c = traj['color']

        for frame_id in range(num_poses):
            # cur_pose = inv(traj_np[frame_id]) # refObj_T_curCam
            # draw_camera(ax, cur_pose, CAM_SIZE, traj_c)

            cur_pose = traj_np[frame_id] # fixCam_T_refObj_T_baseObj
            draw_xyzbar(ax, cur_pose, BAR_LEN)

            cur_pos = cur_pose[:3, 3].copy()
            if frame_id == 0:
                src_pos = cur_pos
                ax.plot([src_pos[0], src_pos[0]], [src_pos[1], src_pos[1]], [src_pos[2], src_pos[2]], 
                    c=traj_c, linewidth=2, label=traj_name
                    )
            else:
                src_pos = last_pos
                dst_pos = cur_pos
                ax.plot([src_pos[0], dst_pos[0]], [src_pos[1], dst_pos[1]], [src_pos[2], dst_pos[2]], 
                    c=traj_c, linewidth=2
                    )
            last_pos = cur_pos
    
    # ax.set_xlim3d(-0.15, 0.15)
    # ax.set_ylim(-0.1, 0.2)
    ax.set_aspect('equal', adjustable='box')
    plt.title("Trajectories of the tracked object")
    plt.legend()

    output_file_name = f"{out_dir}/{file_name}.png"
    plt.savefig(output_file_name)


def vis_pose(rgb, obj_pose, gt_pose, K, bbox, out_dir, ind_str):
    bbox_c_pred = (255, 255, 0)
    rgb_bbox = draw_posed_3d_box(K, rgb, obj_pose, bbox, bbox_c_pred, lw=2)
    bbox_c_gt = (0, 255, 255)
    rgb_bbox = draw_posed_3d_box(K, rgb_bbox, gt_pose, bbox, bbox_c_gt, lw=1, axis=False)
    imageio.imwrite(f'{out_dir}/{ind_str}.jpg', rgb_bbox)


def to_homo(pts):
    '''
    @pts: (N,3 or 2) will homogeneliaze the last dimension
    '''
    assert len(pts.shape) == 2, f'pts.shape: {pts.shape}'
    homo = np.concatenate((pts.copy(), np.ones((pts.shape[0], 1))), axis=-1)
    return homo


def to_homo_torch(pts):
    '''
    @pts: shape can be (...,N,3 or 2) or (N,3) will homogeneliaze the last dimension
    '''
    ones = torch.ones((*pts.shape[:-1], 1)).to(pts.device).float()
    homo = torch.cat((pts.copy(), ones), dim=-1)
    return homo


def draw_posed_3d_box(K, img, obj_pose, bbox, bbox_c, lw=1, axis=True):
    '''
    obj_pose: Transformation obj-to-cam
    bbox: (2,3) min/max
    '''
    xmin, ymin, zmin = bbox.min(axis=0)
    xmax, ymax, zmax = bbox.max(axis=0)
    
    def draw_line3d(start, end, img, color, linewidth):
        pts = np.stack((start, end), axis=0).reshape(-1, 3) # [2,3]
        pts = (obj_pose @ to_homo(pts).T)[:3, :]  # [3, 2]
        projected = (K @ pts).T # [2, 3]
        uv = np.round(projected[:, :2] / projected[:,2].reshape(-1, 1)).astype(int)  # (2,2)
        img = cv2.line(img, tuple(uv[0].tolist()), tuple(uv[1].tolist()), color=color, thickness=linewidth)
        return img
    
    # bounding box
    for y in [ymin, ymax]:
        for z in [zmin, zmax]:
            start = np.array([xmin, y, z])
            end = start+np.array([xmax-xmin, 0, 0])
            img = draw_line3d(start, end, img, bbox_c, lw)
    for x in [xmin, xmax]:
        for z in [zmin, zmax]:
            start = np.array([x, ymin, z])
            end = start+np.array([0, ymax-ymin, 0])
            img = draw_line3d(start, end, img, bbox_c, lw)
    for x in [xmin, xmax]:
        for y in [ymin, ymax]:
            start = np.array([x, y, zmin])
            end = start+np.array([0, 0, zmax-zmin])
            img = draw_line3d(start, end, img, bbox_c, lw)
    if axis:
        # X, Y, Z axises - Red Green Blue
        for i in range(3):
            start = np.array([0.0, 0.0, 0.0])
            end = np.array([0.0, 0.0, 0.0])
            end[i] = 0.8 * abs(bbox[1, i])
            color = [0, 0, 0]
            color[i] = 255
            img = draw_line3d(start, end, img, tuple(color), lw*2)

    return img



def save_video(video_name, frames_dir, fps=15):
    video_name = os.path.join(os.path.dirname(frames_dir), f"{video_name}.mp4")
    logging.info(f"Saving video... in {video_name}")

    frame_lists = sorted(glob.glob(os.path.join(frames_dir, '*')))
    img0 = cv2.imread(frame_lists[0])
    H, W = img0.shape[:2]
    writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for f in frame_lists:
        src_img = cv2.imread(f)
        writer.write(src_img)
    writer.release()

