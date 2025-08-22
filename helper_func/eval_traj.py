import os, sys, argparse, logging, glob, cv2, copy
import rerun as rr
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import numpy as np
from numpy.linalg import inv, norm
from itertools import repeat
from scipy.optimize import minimize
import pycolmap

sys.path.append("..")
from src.data_reader import *

# global rela_rot_errors
# global rotate_errs

code_dir = os.path.dirname(os.path.realpath(__file__))
LOG_DIR = f'{code_dir}/../logs'



def to_homo(pts):
    '''@pts: (N,3 or 2) will homogeneliaze the last dimension'''
    assert len(pts.shape) == 2, f'pts.shape: {pts.shape}'
    homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
    return homo

def make_E_se3(E: np.ndarray) -> np.ndarray:
    """make Essential matrix or Transformation matrix a valid SE(3)"""
    R = E[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    E[:3, :3] = U @ Vt
    return E

def make_Rot_so3(rot_mat):
    U, _, Vt = np.linalg.svd(rot_mat)
    R_SO3 = np.dot(U, Vt) # Ensure orthogonality
    # Ensure determinant is +1
    if np.linalg.det(R_SO3) < 0:
        U[:, -1] *= -1
        R_SO3 = np.dot(U, Vt)

    return R_SO3

def add_err(pred, gt, model_pts):
    """ Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).
    @pred: T_baseObj2Cam [4, 4]
    @model_pts: (N,3)
    """
    pred_pts = (pred @ to_homo(model_pts).T).T[:, :3]
    gt_pts = (gt @ to_homo(model_pts).T).T[:, :3]
    add_err = norm(pred_pts - gt_pts, axis=1).mean()
    return add_err


def adi_err(pred, gt, model_pts):
    """
    @pred: T_baseObj2Cam [4, 4]
    @model_pts: (N,3)
    """
    pred_pts = (pred @ to_homo(model_pts).T).T[:, :3]
    gt_pts = (gt @ to_homo(model_pts).T).T[:, :3]
    nn_index = cKDTree(pred_pts)
    nn_dists, _ = nn_index.query(gt_pts, k=1, workers=-1)
    e = nn_dists.mean()
    return e


def compute_auc(rec, max_val=0.1):
    ''' Computer AUC for the metric
    https://github.com/wenbowen123/iros20-6d-pose-tracking/blob/2df96b720e8e499b9f0d5fcebfbae2bcfa51ab19/eval_ycb.py#L45
    '''
    if len(rec) == 0:
        return 0
    # get sorted list with recall values below max_val
    rec = np.sort(np.array(rec))
    n = len(rec)
    prec = np.arange(1, n+1) / float(n)
    rec = rec.reshape(-1)
    prec = prec.reshape(-1)
    index = np.where(rec < max_val)[0]
    rec = rec[index]
    prec = prec[index]

    if len(rec) == 0:
        return 0
    
    # mrec - modified recall
    # mpre - modified precision
    mrec = [0, *list(rec), max_val]
    mpre = [0, *list(prec), prec[-1]]

    for i in range(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i-1])
    mpre = np.array(mpre)
    mrec = np.array(mrec)
    # find all index where the recal changes
    i = np.where(mrec[1:] != mrec[0:len(mrec)-1])[0] + 1
    # compute the AUC by summing up trapezoidal sections
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) / max_val
    return ap

def cauchy_loss(err, sigma=1.0):
    """
    Compute the Cauchy loss
    Parameters:
    - err (numpy array): The residuals (differences between true and predicted values).
    - sigma (float): Scale parameter for Cauchy loss. Default is 1.0.

    """
    return np.log(1 + (err ** 2) / (sigma ** 2))




def get_relative_rot_all_pairs(reader, ref_poses, raw_pred_poses):
    rela_rot_errors = []

    for idx in tqdm(range(len(ref_poses))):
        cur_rot_pred = make_Rot_so3(raw_pred_poses[idx][:3, :3])
        cur_rot_gt = make_Rot_so3(ref_poses[idx][:3, :3])

        for idx_j in range(idx+1, len(ref_poses)):
            tar_rot_pred = make_Rot_so3(raw_pred_poses[idx_j][:3, :3])
            tar_rot_gt = make_Rot_so3(ref_poses[idx_j][:3, :3])

            rela_rot_pred = np.dot(cur_rot_pred.T, tar_rot_pred)
            rela_rot_gt = np.dot(cur_rot_gt.T, tar_rot_gt)

            # Calculate the rotation error using Rodrigues Formula - deg
            cos_angle = (np.trace(np.dot(rela_rot_pred.T, rela_rot_gt)) - 1.0) / 2.0
            rot_error = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            rela_rot_errors.append(rot_error * 180.0 / np.pi)
            
    # translate_errs = np.array(translate_errs)
    rela_rot_errors = np.array(rela_rot_errors)
    avg_rot = np.mean(rela_rot_errors)
    median_rot = np.median(rela_rot_errors)

    auc_5deg = compute_auc(rela_rot_errors, max_val=5.0)
    auc_10deg = compute_auc(rela_rot_errors, max_val=10.0)
    auc_20deg = compute_auc(rela_rot_errors, max_val=20.0)

    print(f"R_err(deg) - avg | median: {avg_rot:.4f} {median_rot:.4f}")
    print(f"R_err_AUC - 5 | 10 | 20: {auc_5deg:.4f} {auc_10deg:.4f} {auc_20deg:.4f}")



def get_relative_rot(reader, ref_poses, raw_pred_poses):
    # global rela_rot_errors
    # translate_errs = []
    rela_rot_errors = []

    for idx in range(len(ref_poses)):
        pose_gt = ref_poses[idx]
        pose_pred = raw_pred_poses[idx]

        # # Translation error - centimeters
        # trans_pred = pose_pred[:3, 3]
        # trans_gt = pose_gt[:3, 3]
        # translate_errs.append(norm(trans_pred - trans_gt) * 100.0)

        rot_pred = make_Rot_so3(pose_pred[:3, :3])
        rot_gt = make_Rot_so3(pose_gt[:3, :3])
        # get the relative rotation between the current frame 
        # and the previous frame
        if idx > 0:
            rela_rot_pred = np.dot(prev_rot_pred.T, rot_pred)
            rela_rot_gt = np.dot(prev_rot_gt.T, rot_gt)
            # Calculate the rotation error using Rodrigues Formula - deg
            cos_angle = (np.trace(np.dot(rela_rot_pred.T, rela_rot_gt)) - 1.0) / 2.0
            rot_error = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            rela_rot_errors.append(rot_error * 180.0 / np.pi)
            
        # update the previous rotation
        prev_rot_pred = rot_pred
        prev_rot_gt = rot_gt

    # translate_errs = np.array(translate_errs)
    rela_rot_errors = np.array(rela_rot_errors)
    avg_rot = np.mean(rela_rot_errors)
    median_rot = np.median(rela_rot_errors)

    auc_1deg = compute_auc(rela_rot_errors, max_val=1.0)
    auc_2deg = compute_auc(rela_rot_errors, max_val=2.0)
    auc_5deg = compute_auc(rela_rot_errors, max_val=5.0)

    print(f"R_err_avg | R_err_median | R_err_AUC - 1/2/5(deg)")
    print(f"{avg_rot:.4f}\t{median_rot:.4f}\t{auc_1deg:.4f}\t{auc_2deg:.4f}\t{auc_5deg:.4f}")

    # fig = plt.figure(figsize=(8, 8))
    # # errors on relative rotation
    # ax = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    # ax.plot(rela_rot_errors)
    # ax.axhline(avg_rot, color='red', label=f'Avg: {avg_rot:.2f}')
    # ax.axhline(median_rot, color='green', label=f'Median: {median_rot:.2f}')
    # ax.set_title(f"{reader.seq_name} - Relative Rotation Errors (w/o alignment)")
    # ax.legend()
    # # histogram
    # ax = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    # bins = range(0, 41, 2)
    # ax.hist(rela_rot_errors, bins=bins, edgecolor='black')
    # ax.set_xticks(bins, labels=[f'{int(x)}' for x in bins])
    # ax.axvline(avg_rot, color=# Translation error - centimeters
    # ax.axvline(median_rot, color='green')

    # plt.savefig(f'rot_err_plot/{reader.seq_name}_rela_rot_err.png')
        


# =========================== metric functions ===========================

def compute_metric(reader, ref_poses, aligned_pred_poses):
    # global rotate_errs

    rotate_errs = []
    translate_errs = []
    adi_errs = []
    add_errs = []

    thres_ADD = 0.3 # 10cm
    # if isinstance(reader, BEHAVEReader):
    #     thres_ADD = 0.3 # 30cm
    thres_ADD2 = 0.1 # 10cm

    
    model_pts = reader.mesh.vertices.copy()
    # 
    for idx in range(len(ref_poses)):
        pose_gt = ref_poses[idx]
        pose_pred = aligned_pred_poses[idx]

        # Translation error - centimeters
        trans_pred = pose_pred[:3, 3]
        trans_gt = pose_gt[:3, 3]
        translate_errs.append(norm(trans_pred - trans_gt) * 100.0)

        # Rotation error - degree
        rot_pred = make_Rot_so3(pose_pred[:3, :3])
        rot_gt = make_Rot_so3(pose_gt[:3, :3])
        # Calculate the rotation error using Rodrigues Formula - deg
        cos_angle = (np.trace(np.dot(rot_pred.T, rot_gt)) - 1.0) / 2.0
        rot_error = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180.0 / np.pi
        rotate_errs.append(rot_error)
        
        # get ADD-S & ADD error
        adi = adi_err(pose_pred, pose_gt, model_pts)
        add = add_err(pose_pred, pose_gt, model_pts)
        # adi, add = 0.0, 0.0
        adi_errs.append(adi)
        add_errs.append(add)

    # convert to ndarrays
    rotate_errs = np.array(rotate_errs)
    translate_errs = np.array(translate_errs)
    adi_errs = np.array(adi_errs)
    add_errs = np.array(add_errs)

    # get stats
    cnt_below_10cm = np.sum((translate_errs < 10.))
    cnt_below_5cm = np.sum((translate_errs < 5.))
    cnt_below_2cm = np.sum((translate_errs < 2.))
    cnt_below_5cm10deg = np.sum(
        (translate_errs < 10.) & (translate_errs < 5.)
    )

    avg_rot_err = np.mean(rotate_errs)

    # fig = plt.figure(figsize=(8, 5))
    # ax = fig.add_subplot(221)
    # ref_pts = ref_poses[:, :3, 3]
    # pred_trans_pts = aligned_pred_poses[:, :3, 3]
    # ax.set_title(f' {reader.seq_name} - rotational alignment')
    # ax.plot(ref_pts[:, 0], '--', label="ref pos-x", color="cyan")
    # ax.plot(pred_trans_pts[:, 0], label="aligned pos-x", color="blue")
    # ax.legend()
    # ax = fig.add_subplot(222)print(f"t_err(cm) | R_err(deg): {np.mean(translate_errs):.4f} {avg_rot_err:.4f}")
    # ax.plot(ref_pts[:, 1], '--', label="ref pos-y", color="orange")
    # ax.plot(pred_trans_pts[:, 1], label="aligned pos-y", color="red")
    # ax.legend()
    # ax = fig.add_subplot(223)
    # ax.plot(ref_pts[:, 2], '--', label="ref pos-z", color="lime")
    # ax.plot(pred_trans_pts[:, 2], label="aligned pos-z", color="green")
    # ax.legend()
    # ax = fig.add_subplot(224)
    # ax.plot(rotate_errs, color='black')
    # ax.axhline(avg_rot_err, color='red', label=f'avg. rot. err. {avg_rot_err:.2f}\u00B0')
    # ax.legend()

    # plt.savefig(f"detailed graphs/{reader.seq_name}_rot_align.png")
    # plt.close()


    # fig = plt.figure(figsize=(8, 8))
    # # errors on absolute rotation
    # ax = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    # ax.plot(rotate_errs)
    # ax.axhline(avg_rot, color='red', label=f'Avg: {avg_rot:.2f}')
    # ax.set_title(f"{reader.seq_name} - Absolute Rotation Errors (after alignment)")
    # ax.legend()
    # # histogram
    # ax = plt.subplot2grid((2, 2), (1, 0), colspan=2)
    # bins = range(0, 181, 10)
    # ax.hist(rotate_errs, bins=bins, edgecolor='black')
    # ax.set_xticks(bins, labels=[f'{int(x)}' for x in bins])
    # ax.axvline(avg_rot, color='red')

    # plt.savefig(f'rot_err_plot/{reader.seq_name}_abs_rot_err.png')
    # plt.close()

    print("***************************************")
    print(f"t_err(cm) | Acc 10/5/2(cm) | AUC 10/5/2(cm) | ADD-S/ADD @{thres_ADD}(m) | ADD-S/ADD @{thres_ADD2}(m) | R_err(deg)")

    recall_2cm = cnt_below_2cm / (idx+1)
    recall_5cm = cnt_below_5cm / (idx+1)
    recall_10cm = cnt_below_10cm / (idx+1)
    # recall_5cm10deg = cnt_below_5cm10deg / (idx+1)

    AUC_2cm = compute_auc(copy.deepcopy(translate_errs), 2.0)
    AUC_5cm = compute_auc(copy.deepcopy(translate_errs), 5.0)
    AUC_10cm = compute_auc(copy.deepcopy(translate_errs), 10.0)

    AUC_ADD = compute_auc(copy.deepcopy(add_errs), thres_ADD)
    AUC_ADDS = compute_auc(copy.deepcopy(adi_errs), thres_ADD)

    AUC_ADD_new = compute_auc(copy.deepcopy(add_errs), thres_ADD2)
    AUC_ADDS_new = compute_auc(copy.deepcopy(adi_errs), thres_ADD2)

    print(f"{np.mean(translate_errs):.2f}\t{recall_10cm:.2f}\t{recall_5cm:.2f}\t{recall_2cm:.2f}\t{AUC_10cm:.2f}\t{AUC_5cm:.2f}\t{AUC_2cm:.2f}\t{AUC_ADDS:.2f}\t{AUC_ADD:.2f}\t{AUC_ADDS_new:.2f}\t{AUC_ADD_new:.2f}\t{avg_rot_err:.2f}")

    print("***************************************")




# =========================== Alignment functions ===========================

def align_pycolmap_pos(ref_poses, pred_poses):
    # Cam_T_refObj -- w2c
    # convert to c2w
    # ref_poses = inv(ref_poses)
    # pred_poses = inv(pred_poses)

    reset_origin = False
    if reset_origin:
        # move the origin of both trajs to [0,0,0]
        origin_offset = ref_poses[0][:3, 3]
        ref_poses[:, :3, 3] -= origin_offset[None]
        pred_poses[:, :3, 3] -= origin_offset[None]


    ref_recon = pycolmap.Reconstruction()
    pred_recon = pycolmap.Reconstruction()
    Cam_ref = pycolmap.Camera(
        camera_id=1, model='SIMPLE_PINHOLE',
        width=640, height=480, params=[1000, 320, 240],
    )
    ref_recon.add_camera(Cam_ref)
    Cam_pred = pycolmap.Camera(
        camera_id=2, model='SIMPLE_PINHOLE',
        width=640, height=480, params=[1000, 320, 240],
    )
    pred_recon.add_camera(Cam_pred)
    # add images into reconstruction
    for i, pose_pair in enumerate(zip(ref_poses, pred_poses)):
        img_id = i + 1  # Image IDs start from 1 in COLMAP

        ref_pose, pred_pose = pose_pair # w2c
        # we treat the fixed camera as the world here, 
        # and the moving objects are the cmaeras in the Reconstrcution
        ref_pose = inv(ref_pose)
        pred_pose = inv(pred_pose)

        I_ref = pycolmap.Image()
        rigid_ref = pycolmap.Rigid3d(
            R.from_matrix(ref_pose[:3, :3]).as_quat(), ref_pose[:3, 3]
        )
        I_ref = pycolmap.Image(name=f'{img_id}:04d', 
            cam_from_world=rigid_ref, camera_id=1, id=img_id
        )
        ref_recon.add_image(I_ref)
        ref_recon.register_image(img_id)

        rigid_pred = pycolmap.Rigid3d(
            R.from_matrix(pred_pose[:3, :3]).as_quat(), pred_pose[:3, 3]
        )
        I_pred = pycolmap.Image(name=f'{img_id}:04d', 
            cam_from_world=rigid_pred, camera_id=2, id=img_id
        )
        pred_recon.add_image(I_pred)
        pred_recon.register_image(img_id)

    # ****** align ******
    max_proj_center_err = 0.1 # 10 cm
    # src is the predicted traj, target os the ref traj
    # this estimate the alignment of traj of the cameras' locations, 
    # i.e. traj of the inversed pose
    aligned_transform = pycolmap.align_reconstructions_via_proj_centers(
        pred_recon, ref_recon, max_proj_center_err
    )
    # print(aligned_transform)
    scale = aligned_transform.scale
    # print('Scale:', scale)
    rot_mat = R.from_quat(aligned_transform.rotation.quat).as_matrix()
    t_vec = aligned_transform.translation

    sim3_tran = np.eye(4)
    sim3_tran[:3, :3] = (scale * np.eye(3)) @ rot_mat
    sim3_tran[:3, 3] = t_vec
    trans_poses = np.matmul(sim3_tran[None], pred_poses)
    if reset_origin:
        trans_poses[:, :3, 3] += origin_offset

    return trans_poses


def align_globalsfm_rot(ref_poses, pred_poses):
    # convert to c2w
    ref_rots = inv(ref_poses)[:, :3, :3]
    tar_rots = inv(pred_poses)[:, :3, :3]

    # align_res_rot = ref_rots[0] @ tar_rots[0].T

    from global_sfm.build import pyglobalsfm
    # rot_robust_thres, max_iter, thread, enable_init, enable_opt, enable_log
    align_res_rot = pyglobalsfm.align_orientations(
        tar_rots, ref_rots, 0.5, 1000, 11, False, True, False)
    
    cos_rot = (np.trace(align_res_rot) - 1) / 2
    angle = np.arccos(np.clip(cos_rot, -1.0, 1.0)) * 180.0 / np.pi
    print("The opt-angle from globalsfm is", angle)

    sim3_tran = np.eye(4)
    sim3_tran[:3, :3] = align_res_rot

    trans_poses = np.matmul(sim3_tran[None], pred_poses)
    return trans_poses, align_res_rot



def vis_obj_poses(reader:BaseReader, ref_poses, tar_poses, name):
    rr.init(name, recording_id=name, spawn=True)

    K, H, W = reader.K, reader.H, reader.W
    bbox_size_half = reader.bbox.max(axis=0)
    
    # a fix camera in origin
    cam_quat = R.from_matrix(np.eye(3)).as_quat()
    rr.log("camera", rr.Transform3D(translation=np.zeros(3), 
            rotation=rr.Quaternion(xyzw=cam_quat)))
    rr.log("camera", rr.Pinhole(resolution=[W, H], 
            image_from_camera=K, camera_xyz=rr.ViewCoordinates.RDF))

    # full trajectories
    rr.log("gt/traj", rr.LineStrips3D(np.array(ref_poses)[:, :3, 3].reshape(-1, 3), radii=0.0003, colors=[0, 0, 255]), static=True)
    rr.log("pred/traj", rr.LineStrips3D(np.array(tar_poses)[:, :3, 3].reshape(-1, 3), radii=0.0003, colors=[255, 0, 0]), static=True)

    for idx, pose_gt in enumerate(ref_poses):
        pose_pred = tar_poses[idx]
        rot_gt = pose_gt[:3, :3]
        trans_gt = pose_gt[:3, 3]
        rot_pred = make_Rot_so3(pose_pred[:3, :3])
        trans_pred = pose_pred[:3, 3]

        rr.set_time_seconds("stable_time", idx)
        # gt
        rot_gt_quat = R.from_matrix(rot_gt).as_quat()
        rr.log(f"gt/pose",
            rr.Transform3D(translation=trans_gt, 
                quaternion=rr.Quaternion(xyzw=rot_gt_quat))
        )
        rr.log(f"gt/poses/{idx}",
            rr.Transform3D(translation=trans_gt, 
                quaternion=rr.Quaternion(xyzw=rot_gt_quat), axis_length=0.003)
        )
        rr.log("gt/bbox", rr.Boxes3D(
                centers=trans_gt, half_sizes=bbox_size_half, 
                quaternions=rr.Quaternion(xyzw=rot_gt_quat),
                radii=0.0001, colors=(0, 127, 255), fill_mode=1,
            )
        )
        rr.log(f"gt/pose_rot",
            rr.Transform3D(translation=np.zeros(3), 
                quaternion=rr.Quaternion(xyzw=rot_gt_quat))
        )


        # pred
        rot_pred_quat = R.from_matrix(rot_pred).as_quat()
        rr.log(f"pred/pose",
            rr.Transform3D(translation=trans_pred, 
                quaternion=rr.Quaternion(xyzw=rot_pred_quat))
        )
        rr.log(f"pred/poses/{idx}",
            rr.Transform3D(translation=trans_pred, 
                quaternion=rr.Quaternion(xyzw=rot_pred_quat), axis_length=0.003)
        )
        rr.log("pred/bbox", rr.Boxes3D(
                centers=trans_pred, half_sizes=bbox_size_half, 
                quaternions=rr.Quaternion(xyzw=rot_pred_quat),
                radii=0.0003, colors=(255, 255, 0), fill_mode=1,
            )
        )
        rr.log(f"pred/pose_rot",
            rr.Transform3D(translation=np.zeros(3), 
                quaternion=rr.Quaternion(xyzw=rot_pred_quat))
        )


def vis_cams_traj(ref_poses, tar_poses, name):
    rr.init(name, recording_id=name, spawn=True)

    W, H = 640, 480
    K = [[600, 0, 320], [0, 600, 240], [0, 0, 1]]

    green_image = np.zeros((H, W, 3), dtype=np.uint8)
    green_image[:] = (0, 255, 0)
    red_image = np.zeros((H, W, 3), dtype=np.uint8)
    red_image[:] = (255, 0, 0)

    for idx, pose_pair in enumerate(zip(ref_poses, tar_poses)):
        
        c2w_ref = inv(pose_pair[0])
        c2w_tar = inv(pose_pair[1])

        rot_cam_ref = c2w_ref[:3, :3]
        trans_cam_ref = c2w_ref[:3, 3]
        rot_cam = make_Rot_so3(c2w_tar[:3, :3])
        trans_cam = c2w_tar[:3, 3]

        rr.set_time_seconds("stable_time", idx)

        cam_quat_ref = R.from_matrix(rot_cam_ref).as_quat()
        rr.log(f"camera_ref/{idx}", rr.Transform3D(
            translation=trans_cam_ref, rotation=rr.Quaternion(xyzw=cam_quat_ref)))
        rr.log(f"camera_ref/{idx}", rr.Pinhole(
            resolution=[W, H], image_from_camera=K, camera_xyz=rr.ViewCoordinates.RDF))
        rr.log(f"camera_ref/{idx}", rr.Image(green_image))

        rr.log(f"camera_ref_rot",
            rr.Transform3D(translation=np.zeros(3), 
                quaternion=rr.Quaternion(xyzw=cam_quat_ref))
        )

        cam_quat_tar = R.from_matrix(rot_cam).as_quat()
        rr.log(f"camera_tar/{idx}", rr.Transform3D(
            translation=trans_cam, rotation=rr.Quaternion(xyzw=cam_quat_tar)))
        rr.log(f"camera_tar/{idx}", rr.Pinhole(
            resolution=[W, H], image_from_camera=K, camera_xyz=rr.ViewCoordinates.RDF))
        rr.log(f"camera_tar/{idx}", rr.Image(red_image))

        rr.log(f"camera_tar_rot",
            rr.Transform3D(translation=np.zeros(3), 
                quaternion=rr.Quaternion(xyzw=cam_quat_tar))
        )


def vis_stack_pcl(reader:BaseReader, pose_files, use_gtD, name):
    rr.init(name, recording_id=name, spawn=True)

    K, H, W = reader.K, reader.H, reader.W

    grid_u, grid_v = np.meshgrid(np.arange(W), np.arange(H))
    inv_K = np.linalg.inv(K)
    viewer_radii = 1e-3
    max_depth = 1.5

    num_frames = len(pose_files)
    for f_iter in range(num_frames):
        file_idx = reader.map_iter_to_index(f_iter)
        image_i, mask_i, depth_i, _, f_name_i = reader.get_frame_pkg(file_idx, use_gtD)
        pose_i = np.loadtxt(pose_files[f_iter]) # Cam-i_T_refObj
        cam2obj = np.linalg.inv(pose_i)

        rot_cam = cam2obj[:3, :3]
        trans_cam = cam2obj[:3, 3]

        rr.set_time_seconds("stable_time", f_iter)

        # plot current camera w.r.t the object
        cam_quat = R.from_matrix(rot_cam).as_quat()
        rr.log(f"camera/{f_iter}", rr.Transform3D(
            translation=trans_cam, rotation=rr.Quaternion(xyzw=cam_quat)))
        rr.log(f"camera/{f_iter}", rr.Pinhole(resolution=[W, H], 
            image_from_camera=K, camera_xyz=rr.ViewCoordinates.RDF))

        depth_mask = depth_i * mask_i
        valid_pixel = (depth_mask[grid_v, grid_u] > 0.05) & (depth_mask[grid_v, grid_u] < max_depth)
        u = grid_u[valid_pixel]
        v = grid_v[valid_pixel]
        homo_uv = np.vstack((u.T, v.T, np.ones(u.shape[0]))) # [3, N]
        depth_arr = depth_mask[v, u]    # [N]
        P3ds_i = (depth_arr * (inv_K @ homo_uv)).T # [N, 3]
        trans_pts = P3ds_i @ rot_cam.T + trans_cam.T

        colors = image_i[v, u].astype('float') / 255.0  # [N, 3]
        # /{f_iter}
        rr.log(f"raw_pcl", rr.Points3D(
            trans_pts, colors=colors, radii=viewer_radii))



def draw_poses_align(reader, ref_poses, aligned_pred_poses, draw_pose_dir):
    from src.utils import vis_pose
    os.system(f'rm -rf {draw_pose_dir} && mkdir -p {draw_pose_dir}')

    K = reader.K
    bbox = reader.bbox
    rgb_list = []
    f_name_list = []

    num_frames = len(ref_poses)
    num_poses = num_frames
    for img_iter in range(num_frames):
        frame_idx = reader.map_iter_to_index(img_iter)

        rgb_list.append(reader.get_color(frame_idx))
        f_name_list.append(reader.id_strs[frame_idx])

    import multiprocessing as mp
    idx_s = 0
    with tqdm(total=num_poses) as pbar:
        while(idx_s < num_poses):
            idx_e = min(num_poses, idx_s + 50)
            with mp.Pool(mp.cpu_count()-2) as pool:
                pool.starmap(vis_pose, zip(
                    rgb_list[idx_s:idx_e], aligned_pred_poses[idx_s:idx_e], ref_poses[idx_s:idx_e], 
                    repeat(K), repeat(bbox), repeat(draw_pose_dir), f_name_list[idx_s:idx_e]
                    ))
            pbar.update(idx_e - idx_s)
            idx_s = idx_e




def align_and_eval(file_dir, exp_name, use_gtD, reader:BaseReader):
    from evo.tools import file_interface
    from src.utils import save_video
    
    gt_traj_path = f"{file_dir}/gt_traj.txt"
    traj_ref = file_interface.read_kitti_poses_file(gt_traj_path)
    ref_poses = traj_ref.poses_se3
    ref_poses = np.array(ref_poses)

    pred_traj_path = f"{file_dir}/pred_traj_{exp_name}.txt"
    if use_gtD:
        pred_traj_path = f"{file_dir}/pred_traj_{exp_name}.txt"

    traj_est = file_interface.read_kitti_poses_file(pred_traj_path)
    tar_poses = copy.deepcopy(traj_est.poses_se3)

    # ==============================================================
    # vis_cams_traj(ref_poses, tar_poses, seq_name+'_bevor')

    # vis_obj_poses(reader, ref_poses, tar_poses, seq_name+'_bevor')

    print("w/o alignment")
    get_relative_rot(reader, ref_poses, tar_poses)

    # ===================== Alignment =====================
    if not use_gtD:
        
        # =========== align the position by pycolmap ===========
        aligned_res = align_pycolmap_pos(ref_poses, tar_poses)

        # =========== align the rotation by globalsfm =========== 
        # aligned_res, rot_trans = align_globalsfm_rot(ref_poses, tar_poses)

        # # ==== Try to align the position first and then align the rotation ====
        # aligned_res = align_pycolmap_pos(ref_poses, tar_poses)
        # aligned_res, rot_trans = align_globalsfm_rot(ref_poses, aligned_res)


        # vis_cams_traj(ref_poses, aligned_res, seq_name+'_after')

        # vis_obj_poses(reader, ref_poses, tar_poses, seq_name+'_bevor')

        Show_Align = False
        # ************ Show the aligned trajectories ***********
        if Show_Align:
            ref_pts = ref_poses[:, :3, 3]
            pred_trans_pts = aligned_res[:, :3, 3]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(ref_pts[:, 0], ref_pts[:, 1], ref_pts[:, 2], 
                    'x-', label="Reference Trajectory", color="blue")
            ax.plot(pred_trans_pts[:, 0], pred_trans_pts[:, 1], pred_trans_pts[:, 2], 
                    'x-', label="Aligned Target Trajectory", color="red")
            ax.legend()
            plt.savefig(f"{file_dir}/aligned_traj_{exp_name}.png")
            # plt.show()
            plt.close()
        
        print("After alignment")
        compute_metric(reader, ref_poses, aligned_res)

        
        draw_pose_dir = f'{file_dir}/aligned_pose_vis'
        draw_poses_align(reader, ref_poses, aligned_res, draw_pose_dir)
        video_name = 'aligned_pose_'+exp_name
        save_video(video_name=video_name, frames_dir=draw_pose_dir)
        # # visualize the traj in rerun
        # vis_rerun(seq_name, ref_poses, aligned_res, reader)
    else:
        compute_metric(reader, ref_poses, tar_poses)
    

def compare_align_traj(f_dir1, f_dir2, exp_name, reader:BaseReader):
    from evo.tools import file_interface

    gt_traj_path = f"{f_dir1}/gt_traj.txt"
    pred_traj_path = f"{f_dir1}/pred_traj_{exp_name}.txt"
    pred_traj_path_SDF = f"{f_dir2}/pred_traj_{exp_name}.txt"

    traj_ref = file_interface.read_kitti_poses_file(gt_traj_path)
    traj_est = file_interface.read_kitti_poses_file(pred_traj_path)
    traj_est2 = file_interface.read_kitti_poses_file(pred_traj_path_SDF)
    traj_ref = traj_ref.poses_se3
    traj_est = traj_est.poses_se3
    traj_est2 = traj_est2.poses_se3

    # align the translation of all poses
    aligned_est_poses = align_pycolmap_pos(traj_ref, traj_est)
    aligned_est_poses2 = align_pycolmap_pos(traj_ref, traj_est2)

    # # cut a segment
    # s, t = 70, 90
    # traj_ref = traj_ref[s:t]
    # aligned_est_poses = aligned_est_poses[s:t]
    # aligned_est_poses2 = aligned_est_poses2[s:t]

    # BAR_LEN = 8e-3

    # def draw_xyzbar(ax, Cam_T_Obj, size):
    #     """[R|t]: Transformation Mat from Obj frame to Cam"""
    #     coord_bar = size * np.array([
    #         [0, 0, 0], [1, 0, 0], 
    #         [0, 0, 0], [0, 1, 0],
    #         [0, 0, 0], [0, 0, 1],
    #     ]).T
    #     coord_bar = Cam_T_Obj[:3, :3] @ coord_bar + Cam_T_Obj[:3, 3:4] # [3, N]
    #     # Draw camera
    #     ax.plot(coord_bar[0,0:2], coord_bar[1,0:2], coord_bar[2,0:2], c='red', lw=1)
    #     ax.plot(coord_bar[0,2:4], coord_bar[1,2:4], coord_bar[2,2:4], c='green', lw=1)
    #     ax.plot(coord_bar[0,4:6], coord_bar[1,4:6], coord_bar[2,4:6], c='blue', lw=1)

    # num_poses = len(traj_ref)
    # traj_pred_SDF = {'traj_np':np.array(aligned_est_poses), 'color':'red', 'name':'BundleSDF'}
    # traj_pred_MonoT = {'traj_np':np.array(aligned_est_poses2), 'color':'blue', 'name':'MonoTracker'}
    # traj_gt_poses = {'traj_np':np.array(traj_ref), 'color':'green', 'name':'GT'}
    # traj_list = [traj_pred_SDF, traj_pred_MonoT, traj_gt_poses]

    # fig = plt.figure()
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('x')
    # ax.set_ylabel('z')
    
    # for traj in traj_list:
    #     traj_np = traj['traj_np'] # [N, 4, 4]
    #     traj_name = traj['name']
    #     traj_c = traj['color']
    #     line_type = '-'
    #     if traj_name == 'GT':
    #         line_type = '--'

    #     for frame_id in range(num_poses):
    #         cur_pose = traj_np[frame_id] # fixCam_T_refObj_T_baseObj
    #         draw_xyzbar(ax, cur_pose, BAR_LEN)

    #         cur_pos = cur_pose[:3, 3].copy()
    #         if frame_id == 0:
    #             src_pos = cur_pos
    #             ax.plot([src_pos[0], src_pos[0]], [src_pos[1], src_pos[1]], [src_pos[2], src_pos[2]], 
    #                 line_type, c=traj_c, linewidth=2, label=traj_name)
    #         else:
    #             src_pos = last_pos
    #             dst_pos = cur_pos
    #             ax.plot([src_pos[0], dst_pos[0]], [src_pos[1], dst_pos[1]], [src_pos[2], dst_pos[2]], 
    #                 line_type, c=traj_c, linewidth=2)
    #         last_pos = cur_pos
    
    # ax.set_aspect('equal', adjustable='box')
    # plt.legend()
    # plt.show()
    

    # ========================= Only Draw Positions =========================
    # ref_pts = np.array(traj_ref)[:, :3, 3]
    # pred_trans_pts = np.array(aligned_est_poses)[:, :3, 3]
    # pred_trans_pts2 = np.array(aligned_est_poses2)[:, :3, 3]

    # # visualization
    # fig = plt.figure()
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(ref_pts[:, 0], ref_pts[:, 1], ref_pts[:, 2], 
    #         'o--', color="green", lw=2)
    # ax.plot(pred_trans_pts[:, 0], pred_trans_pts[:, 1], pred_trans_pts[:, 2], 
    #         'o-', color="red", lw=2)
    # ax.plot(pred_trans_pts2[:, 0], pred_trans_pts2[:, 1], pred_trans_pts2[:, 2], 
    #         'o-', color="blue", lw=2)
    # # ax.legend()
    # ax.set_title(seq_name)
    # # plt.savefig(f"{f_dir1}/aligned_traj_compare.png")
    # plt.show()
    


def main(dataset, seq_name, start_frame, step, use_gtD, exp_name):

    data_base_dir = '/media/zilong/Documents/MasterProject'
    out_base_dir = '/home/zilong/tmp'
    if dataset == 'YCB':
        data_dir = f"{data_base_dir}/YCBInEOAT/{seq_name}"
        out_dir = f"{out_base_dir}/ycbineoat/{seq_name}"
        reader = YcbineoatReader(data_dir=data_dir, out_dir=out_dir, 
            downscale=1.0, shorter_side=None, start_frame=start_frame, step=step, 
            mono_type='depth_any2', normal_type='normal_m3dv2')
    elif dataset == 'BEHAVE':
        data_dir = f"{data_base_dir}/BEHAVE/Date02/{seq_name}"
        out_dir = f"{out_base_dir}/behave/date02/{seq_name}"
        reader = BEHAVEReader(data_dir=data_dir, out_dir=out_dir, 
            downscale=0.5, shorter_side=None, start_frame=start_frame, step=step, 
            mono_type='depth_any2', normal_type='normal_m3dv2')
    elif dataset == 'HO3D_v3':
        data_dir = f"{data_base_dir}/HO3D_v3/{seq_name}"
        out_dir = f"{out_base_dir}/ho3d/{seq_name}"
        reader = Ho3dReader(data_dir=data_dir, out_dir=out_dir, 
            downscale=1.0, shorter_side=None, start_frame=start_frame, step=step, 
            mono_type='depth_any2', normal_type='normal_m3dv2')
    
    print("Running evaluation for sequence -", reader.seq_name)

    # ============ Vis the stacked pcl with opt-poses ============
    # pose_dir = f'{reader.out_dir}/result_files/global_pose'
    # if use_gtD:
    #     pose_dir = pose_dir+'_gtD'
    # pose_files = sorted(glob.glob(pose_dir+'/*'))
    # vis_stack_pcl(reader, pose_dir, use_gtD, seq_name)

    # ============ align and compute the metrics ============
    # NOTE change the dir name accordingly - for BundleSDF it is Trajs_SDF
    file_dir = f"{LOG_DIR}/{seq_name}/Trajs"
    align_and_eval(file_dir, exp_name, use_gtD, reader)

    # ============ compare results from two methods ============
    # f_dir1 = f"{LOG_DIR}/{seq_name}/Trajs"
    # f_dir2 = f"{LOG_DIR}/{seq_name}/Trajs_SDF"
    # compare_align_traj(f_dir1, f_dir2, exp_name, reader)

    print('\n')


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
        'boxmedium_hand': [140, 1, 201], 
        'chairwood_hand': [50, 1, 151],
        'plasticcontainer': [50, 1, 101], 
        'stool_move': [80, 1, 81],
        'tablesmall_lift': [0, 1, 121], 
        'tablesquare_lift': [0, 1, 151],
    }
    ho3d_list = {
        'AP10': [0, 5, 101], 'AP11': [0, 5, 101], 'AP12': [0, 5, 101],
        'MPM10': [280, 5, 121], 'MPM11': [1200, 5, 81], 'MPM12': [80, 8, 111],
        'SB11': [880, 5, 101], 'SB13': [880, 5, 101], 'SM1': [0, 3, 151],
    }

    # dataset = 'YCB'
    # dataset = 'BEHAVE'
    dataset = 'HO3D_v3'
    USE_gtD = False

    # rotate_errs = []
    # rela_rot_errors = []
    
    # NOTE you have to enter the exact exp name postfix to the traj file
    # so the function can locate your traj file
    exp_name = 'Raydiffusion'

    # ********* run for single frame *********
    seq_name = 'AP10'
    exp_setting = ho3d_list[seq_name]

    start_frame = exp_setting[0]
    step_frame = exp_setting[1]
    main(dataset, seq_name, start_frame, step_frame, 
         USE_gtD, exp_name)

    # ********* run for the whole dataset *********
    # for k, v in behave_list.items():
    #     main(dataset, k, v[0], v[1], USE_gtD, exp_name)
    
