# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os,sys,logging,copy,json,pickle,glob
import cv2,imageio,trimesh,pdb
import numpy as np

code_dir = os.path.dirname(os.path.realpath(__file__))

YCB_MODEL_DIR = '/media/zilong/Documents/MasterProject/YCBInEOAT/obj_models'
BEHAVE_MODEL_DIR = '/media/zilong/Documents/MasterProject/BEHAVE/obj_models'

class BaseReader:
    def __init__(self, 
            data_dir, out_dir, 
            downscale=1, shorter_side=None, 
            start_frame=0, step=1,
            mono_type='depth_m3dv2', 
            normal_type='normal_m3dv2'
        ):
        """
        downscale: ratio to downscale, 0.5 means the images is  in  half size of the original one
        shorter_side: length of the shorter side for the downscaled image
        """
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.seq_name = os.path.basename(data_dir)
        self.start_frame = start_frame
        self.step = step

        self.color_files = sorted(glob.glob(os.path.join(data_dir, 'rgb', '*')))
        # list of file names are stored according to the rgb files
        self.id_strs = []
        for color_file in self.color_files:
            id_str = os.path.basename(color_file).replace(".jpg","").replace(".png","")
            self.id_strs.append(id_str)
        self.H, self.W = cv2.imread(self.color_files[0]).shape[:2]

        self.downscale = downscale
        if shorter_side is not None:
            self.downscale = shorter_side/min(self.H, self.W)

        self.H = int(self.H*self.downscale)
        self.W = int(self.W*self.downscale)
        logging.info(f"Frame will be resized to {self.H} x {self.W} during tracking")

        # get camera intrinsic matrix
        self.get_cam_intrinsics()
        logging.info(f"kx: {self.K[0][0]:.2f}, ky: {self.K[1][1]:.2f}, cx: {self.K[0][2]:.2f}, cy: {self.K[1][2]:.2f}")
        self.K[:2] *= self.downscale

        self.gt_depth_dir = os.path.join(data_dir, 'depth')
        self.mask_dir = os.path.join(out_dir, 'masks_net')
        self.depth_dir = os.path.join(out_dir, mono_type)
        self.normal_dir = os.path.join(out_dir, normal_type)

        self.configs = {}
        
        # only for visualization
        self.model_offset = np.eye(4)
        self.gt_pose_dir = os.path.join(self.data_dir, 'annotated_poses')
        self.gt_pose_files = sorted(glob.glob(os.path.join(self.gt_pose_dir, '*')))

    def __len__(self):
        return len(self.color_files)
    
    def get_cam_intrinsics(self):
        self.K = np.loadtxt(os.path.join(self.data_dir, 'cam_K.txt')).reshape(3, 3)
    
    def map_iter_to_index(self, iter):
        return self.start_frame + iter * self.step
    
    def get_frame_pkg(self, file_idx, use_gtD):
        rgb = self.get_color(file_idx)
        mask = self.get_mask(file_idx)
        if use_gtD:
            depth = self.get_gt_depth(file_idx)
        else:
            depth = self.get_depth(file_idx)
        normal_map = self.get_normal(file_idx)
        file_name = self.id_strs[file_idx]

        return rgb, mask, depth, normal_map, file_name
    
    def get_color(self, i):
        color = imageio.imread(self.color_files[i])
        if color.shape[2] > 3:
            color = color[:, :, :3]
        color = cv2.resize(color, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return color

    def get_mask(self, i):
        """read pre-processed object binary mask"""
        file_path = os.path.join(self.mask_dir, self.id_strs[i]+'.png')
        if not os.path.exists(file_path):
            logging.error(f"No mask image for frame {i} - {self.id_strs[i]}")
            return None
        mask = cv2.imread(file_path, -1)
        # if len(mask.shape) == 3:
        #     mask = (np.sum(mask, axis=-1) > 0).astype(np.uint8)
        if np.max(mask) > 1:
            mask = (mask / 255).astype(np.uint8)
        mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        # erode the mask
        # kernel = np.ones((3, 3), np.uint8)
        # mask = cv2.erode(mask, kernel, iterations=1)
        
        return mask

    def get_depth(self, i):
        """Read the monocular depth images"""
        file_path = os.path.join(self.depth_dir, self.id_strs[i]+'.png')
        depth = cv2.imread(file_path, -1) / 1000.0
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return depth
    
    def get_gt_depth(self, i):
        """read gt sensed depth images"""
        file_path = os.path.join(self.gt_depth_dir, self.id_strs[i]+'.png')
        if not os.path.exists(file_path):
            return None
        depth = cv2.imread(file_path, -1) / 1000.0
        # dialate the sensed depth
        # kernel = np.ones((3, 3), np.uint8)
        # depth = cv2.dilate(depth, kernel, iterations=1)
        # # process the sparse gt depth
        # depth = smooth_sensed_depth(depth, self.configs)
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return depth
    
    def get_normal(self, i):
        """read normal map in RGB format"""
        file_path = os.path.join(self.normal_dir, self.id_strs[i]+'.png')
        if not os.path.exists(file_path):
            return None
        normal = imageio.imread(file_path)[:,:,:3]
        normal = cv2.resize(normal, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        # process normal map
        valid_mask = np.sum(normal, axis=2) > 0
        normal = (normal / 255.0) * 2 - 1 # [0, 255] -> [-1, 1]
        normal = normal / (np.linalg.norm(normal, axis=2, keepdims=True) + 1e-10)
        normal[~valid_mask, :] = 0
        return normal
    
    def get_gt_pose(self, i):
        """ 
        Reference poses of the objects, convert from vertices in mesh to camera frame
        As we use this to draw bbox, which is defined in baseObj frame.
        So we append a model_offset, which is meshW_T_baseObj
        Return:
            Cam_T_baseObj
        """
        try:
            pose = np.loadtxt(self.gt_pose_files[i]).reshape(4, 4)
            return pose @ self.model_offset
        except:
            logging.info("GT pose not found, return None")
            return None
    





class YcbineoatReader(BaseReader):
    def __init__(
            self, data_dir, out_dir, 
            downscale=1, shorter_side=None, 
            start_frame=0, step=1,
            mono_type='depth_m3dv2', 
            normal_type='normal_m3dv2'
        ):

        super().__init__(data_dir, out_dir, downscale, shorter_side, 
                         start_frame, step, mono_type, normal_type)
        
        self.configs = {
            "erode": {"diff": 0.001, "radius": 1, "ratio": 0.8,},
            "zfar": 2.5,
            "bilateral_filter": {"radius": 2, "sigma_D": 2, "sigma_R": 100000,}
        }

        gt_pose_dir = os.path.join(self.data_dir, 'annotated_poses')
        self.gt_pose_files = sorted(glob.glob(os.path.join(gt_pose_dir, '*')))

        self.seqname_to_object = {
            'bleach0': "021_bleach_cleanser",
            'bleach_hard_00_03_chaitanya': "021_bleach_cleanser",
            'cracker_box_reorient': '003_cracker_box',
            'cracker_box_yalehand0': '003_cracker_box',
            'mustard0': '006_mustard_bottle',
            'mustard_easy_00_02': '006_mustard_bottle',
            'sugar_box1': '004_sugar_box',
            'sugar_box_yalehand0': '004_sugar_box',
            'tomato_soup_can_yalehand0': '005_tomato_soup_can',
        }

        obj_name = self.seqname_to_object[self.seq_name]
        self.mesh = trimesh.load(f'{YCB_MODEL_DIR}/{obj_name}/textured_simple.obj')
        to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
        self.model_offset = np.linalg.inv(to_origin)
    


class BEHAVEReader(BaseReader):
    def __init__(
            self, data_dir, out_dir,
            downscale=1, shorter_side=None, 
            start_frame=0, step=1,
            mono_type='depth_m3dv2', 
            normal_type='normal_m3dv2'
        ):

        super().__init__(data_dir, out_dir, downscale, shorter_side, 
                         start_frame, step, mono_type, normal_type)
        
        self.configs = {
            "erode": {"diff": 0.001, "radius": 1, "ratio": 0.8,},
            "zfar": 3.5,
            "bilateral_filter": {"radius": 2, "sigma_D": 2, "sigma_R": 100000,}
        }

        gt_pose_dir = os.path.join(self.out_dir, 'annotated_poses')
        self.gt_pose_files = sorted(glob.glob(os.path.join(gt_pose_dir, '*')))
        
        # to_origin: baseObj_T_W because pts of bbox in W-frame would be moved to the model frame
        # e.g.: to_origin @ vertices_inW -> pts in model frame
        obj_name = self.seq_name.split('_')[0]
        self.mesh = trimesh.load(f'{BEHAVE_MODEL_DIR}/{obj_name}/{obj_name}.obj')
        
        to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
        # self.bbox[:, 0] += 0.16
        # self.bbox[1, 2] -= 0.05
        self.model_offset[:3,:3] = np.linalg.inv(to_origin)[:3,:3]




class Ho3dReader(BaseReader):
    def __init__(
            self, data_dir, out_dir, 
            downscale=1, shorter_side=None, 
            start_frame=0, step=1,
            mono_type='depth_m3dv2', 
            normal_type='normal_m3dv2'
        ):

        super().__init__(data_dir, out_dir, downscale, shorter_side, 
                         start_frame, step, mono_type, normal_type)
        
        self.configs = {
            "erode": {"diff": 0.001, "radius": 1, "ratio": 0.8,},
            "zfar": 1.5,
            "bilateral_filter": {"radius": 2, "sigma_D": 2, "sigma_R": 100000,}
        }

        # self.gl_to_cv = np.array([
        #     [1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]
        # ])
        gt_poses_path = os.path.join(self.data_dir, 'ob_in_cams.txt')
        self.gt_poses = np.loadtxt(gt_poses_path).reshape((-1, 4, 4))
        
        self.seqname_to_object = {
            'AP': '019_pitcher_base',
            'MPM': '010_potted_meat_can',
            'SB': '021_bleach_cleanser',
            'SM': '006_mustard_bottle',
        }

        for k in self.seqname_to_object:
            if self.seq_name.startswith(k):
                obj_name = self.seqname_to_object[k]
                break
        self.mesh = trimesh.load(f'{YCB_MODEL_DIR}/{obj_name}/textured_simple.obj')
        to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
        self.model_offset = np.linalg.inv(to_origin)


    def get_cam_intrinsics(self):
        meta_file = glob.glob(os.path.join(self.data_dir, 'meta', '*'))[0]
        self.K = pickle.load(open(meta_file, 'rb'))['camMat'].reshape(3, 3)

    def get_gt_depth(self, i):
        file_path = os.path.join(self.gt_depth_dir, self.id_strs[i]+'.png')
        depth = cv2.imread(file_path, -1)
        # ************** from official code **************
        depth_scale = 0.00012498664727900177
        depth = (depth[:, :, 2] + depth[:, :, 1]* 256)* depth_scale
        # ************************************************
        # dialate the sensed depth
        # kernel = np.ones((5, 5), np.uint8)
        # depth = cv2.dilate(depth, kernel, iterations=1)
        # # process the sparse gt depth
        # depth = smooth_sensed_depth(depth, self.configs)
        depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return depth

    def get_gt_pose(self, i):
        """ 
        Reference poses of the objects, convert from vertices in mesh to openGL frame
        First convert it to in Camera frame
        As we use this to draw bbox, which is defined in baseObj frame.
        So we append a model_offset, which is meshW_T_baseObj
        Return:
            Cam_T_baseObj
        """
        pose = self.gt_poses[i]
        return pose @ self.model_offset












# ================================= depth processing ======================================

def smooth_sensed_depth(depth, configs):
    erode_radius = configs['erode']['radius']
    erode_ratio = configs['erode']['ratio']
    erode_diff = configs['erode']['diff']
    zfar = configs['zfar']

    bf_radius = configs['bilateral_filter']['radius']
    sigma_D = configs['bilateral_filter']['sigma_D']
    sigma_R = configs['bilateral_filter']['sigma_R']
    
    depth_eroded = erodeDepthMap(depth, erode_radius, erode_diff, erode_ratio, zfar)
    depth_filtered = gaussFilterDepthMap(depth_eroded, bf_radius, sigma_D, sigma_R, zfar)
    depth_filtered = gaussFilterDepthMap(depth_filtered, bf_radius, sigma_D, sigma_R, zfar)

    return depth_filtered


def erodeDepthMap(d_input, erode_radius, erode_diff, erode_ratio, zfar):
    """Only keep depth values in flat areas
    """
    d_output = np.zeros_like(d_input)
    valid_depth_mask = (d_input>0.05) & (d_input<=zfar)

    # pad and get neighbor blocks
    d_input_padded = np.pad(d_input, erode_radius, mode='constant', constant_values=0.0)
    win_size = (2*erode_radius + 1, 2*erode_radius + 1)
    local_blocks = np.lib.stride_tricks.sliding_window_view(d_input_padded, win_size)
    
    # Count invalid neighbors
    depth_diff = np.abs(local_blocks - d_input[:, :, None, None])
    valid_neighbors = (local_blocks<np.inf) & (local_blocks>=0.05) & (depth_diff<=erode_diff)
    valid_count = np.sum(valid_neighbors, axis=(-1, -2))

    # Check if the fraction of invalid neighbors exceeds the threshold
    erosion_mask = ((valid_count / (win_size[0]**2)) >= (1-erode_ratio)) & valid_depth_mask

    # Apply the mask to the valid depth areas
    d_output[erosion_mask] = d_input[erosion_mask]

    return d_output
    

def gaussFilterDepthMap(d_input, bf_radius, sigma_D, sigma_R, zfar):
    d_output = np.zeros_like(d_input)
    valid_depth_mask = (d_input>0.05) & (d_input<=zfar)

    # pad and get neighbor blocks
    d_input_padded = np.pad(d_input, bf_radius, mode='constant', constant_values=0)
    win_size = (2*bf_radius + 1, 2*bf_radius + 1)
    local_blocks = np.lib.stride_tricks.sliding_window_view(d_input_padded, win_size)

    # Compute the mean depth for each valid neighborhood
    valid_mask_neighbors = (local_blocks >= 0.05) & (local_blocks <= zfar)
    num_valid = np.sum(valid_mask_neighbors, axis=(-1, -2))
    valid_mask_mean_d = (num_valid > 0)

    # return 0 for center pixel without any valid neighbor
    mean_depth = np.zeros_like(d_input)
    mean_depth[valid_mask_mean_d] = np.sum(
        local_blocks * valid_mask_neighbors, axis=(-1, -2)
        )[valid_mask_mean_d] / num_valid[valid_mask_mean_d]

    # Gaussian spatial weights (distance-based)
    x, y = np.meshgrid(np.arange(-bf_radius, bf_radius + 1), np.arange(-bf_radius, bf_radius + 1))
    spatial_weights = np.exp(- (x**2 + y**2) / (2.0 * sigma_D**2))

    # Gaussian range weights (depth-based) calculated based on the mean depth
    depth_diff = local_blocks - d_input[:, :, None, None]
    diff_weights = np.exp(- depth_diff**2 / (2.0 * sigma_R**2))

    depth_mean_diff = np.abs(local_blocks - mean_depth[:, :, None, None])
    depth_mean_thres = 0.01
    valid_neighbors = (local_blocks>=0.05) & (local_blocks<=zfar) & (depth_mean_diff<depth_mean_thres)
    total_weights = np.where(
        valid_neighbors, 
        spatial_weights[None, None] * diff_weights, 0)
    
    sum_weighted_depth = np.sum(total_weights * local_blocks, axis=(-1, -2))
    sum_weights = np.sum(total_weights, axis=(-1, -2))

    valid_mask = valid_depth_mask & (sum_weights > 0.0) & (num_valid > 0) # / (win_size[0]**2)
    d_output[valid_mask] = sum_weighted_depth[valid_mask] / sum_weights[valid_mask]

    return d_output

