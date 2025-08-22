import os, sys, pickle, glob, json, tqdm

import numpy as np
from PIL import Image

sys.path.append("..")
# from src.kf_manager import *
from src.data_reader import *


def mask_out_img_with_transparency(img_path, mask_path, save_path):
    """
    Given an image and a mask, add the mask as an alpha channel to the image.
    """
    # print(f"Processing {img_path}")
    if not os.path.exists(img_path):
        logging.error(f"Image file {img_path} not found.")
        return
    if not os.path.exists(mask_path):
        logging.error(f"Mask file {mask_path} not found.")
    
    img = Image.open(img_path)
    mask = Image.open(mask_path)
    # if mask > 0, set alpha to 255, else 0
    mask = mask.convert("L")
    mask = mask.point(lambda p: 255 if p > 0 else 0)
    img.putalpha(mask)
    # if the alpha channel is 0, set RGB to 0 too
    img = img.convert("RGBA")
    img_data = np.array(img)
    # set RGB to 0 if alpha is 0
    img_data[img_data[:, :, 3] == 0] = 0
    img = Image.fromarray(img_data, "RGBA")
    # img = Image.fromarray(img_data[..., :3], "RGB")
    img.save(save_path)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)



def main(dataset, seq_name, start_frame, step):

    data_base_dir = '/media/zilong/Documents/MasterProject'
    out_base_dir = '/home/zilong/tmp'
    if dataset == 'YCB':
        data_dir = f"{data_base_dir}/YCBInEOAT/{seq_name}"
        out_dir = f"{out_base_dir}/ycbineoat/{seq_name}"
        reader = YcbineoatReader(data_dir=data_dir, out_dir=out_dir, 
            downscale=1.0, shorter_side=None, start_frame=start_frame, step=step, 
            mono_type='depth_any2', normal_type='normal_m3dv2')
        H, W = 480, 640
    elif dataset == 'BEHAVE':
        data_dir = f"{data_base_dir}/BEHAVE/Date02/{seq_name}"
        out_dir = f"{out_base_dir}/behave/date02/{seq_name}"
        reader = BEHAVEReader(data_dir=data_dir, out_dir=out_dir, 
            downscale=0.5, shorter_side=None, start_frame=start_frame, step=step, 
            mono_type='depth_any2', normal_type='normal_m3dv2')
        # H, W = 1536, 2048
        H, W = 768, 1024
    elif dataset == 'HO3D_v3':
        data_dir = f"{data_base_dir}/HO3D_v3/{seq_name}"
        out_dir = f"{out_base_dir}/ho3d/{seq_name}"
        reader = Ho3dReader(data_dir=data_dir, out_dir=out_dir, 
            downscale=1.0, shorter_side=None, start_frame=start_frame, step=step, 
            mono_type='depth_any2', normal_type='normal_m3dv2')
        H, W = 480, 640

    K = reader.K

    # select poses source
    # est_pose_dir = os.path.join(out_dir, 'bundlesdf', 'global_pose')
    est_pose_dir = os.path.join(out_dir, 'result_files', 'global_pose')
    pred_pose_list = sorted(glob.glob(est_pose_dir+'/*'))
    
    # also change the name here
    save_path = f'{log_dir}/BSDF_{seq_name}.json'

    log_dir = f'../logs/{seq_name}/nerf_data'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.makedirs(os.path.join(log_dir, 'rgb_masked_png'))
        os.makedirs(os.path.join(log_dir, 'image'))
        os.makedirs(os.path.join(log_dir, 'mask'))

    json_dict = {"camera_model": "OPENCV"}
    json_dict["fl_x"] = K[0, 0]
    json_dict["fl_y"] = K[1, 1]
    json_dict["cx"] = K[0, 2]
    json_dict["cy"] = K[1, 2]
    json_dict["w"] = W
    json_dict["h"] = H
    # set k1 k2 k3 k4 p1 p2 to 0
    json_dict["k1"] = 0
    json_dict["k2"] = 0
    json_dict["k3"] = 0
    json_dict["k4"] = 0
    json_dict["p1"] = 0
    json_dict["p2"] = 0

    json_dict["frames"] = []
    depth_type = 'depth_any2'
    mask_type = 'masks_net'

    num_frames = len(pred_pose_list)
    pose_offset = reader.get_gt_pose(reader.map_iter_to_index(0))
    model_offset = reader.model_offset

    for f_iter in tqdm.tqdm(range(num_frames)):
        file_idx = reader.map_iter_to_index(f_iter)
        frame_dict = {}
        file_name = reader.id_strs[file_idx]
        frame_dict['f_name'] = file_name

        # remove frames with incorrect gt pose data
        w2c_gt = reader.get_gt_pose(file_idx)
        if np.allclose(w2c_gt, model_offset):
            continue

        # np.savetxt(f'{log_dir}/{file_name}.txt', K)
        frame_dict["file_path"] = f"rgb_masked_png/{file_name}.png"
        
        save_pics = False
        if save_pics:
            rgb_path = f"{data_dir}/rgb/{file_name}.jpg"
            mask_path = f"{out_dir}/masks_net/{file_name}.png"
            out_path = f"{log_dir}/rgb_masked_png/{file_name}.png"
            mask_out_img_with_transparency(rgb_path, mask_path, out_path)

            img = cv2.imread(rgb_path)
            mask = cv2.imread(mask_path, -1)
            img = img * mask[..., None]
            img = cv2.imwrite(f"{log_dir}/image/{file_name}.jpg", img)
            # os.system(f"cp {rgb_path} {log_dir}/image/")
            # os.system(f"cp {mask_path} {log_dir}/mask/")

        w2c = np.loadtxt(pred_pose_list[f_iter]) @ pose_offset
        c2w = np.linalg.inv(w2c)
        # from OpenCV to OpenGL
        c2w_RUB = c2w @ np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        frame_dict["transform_matrix"] = c2w_RUB.tolist()

        # # *********** for PoRf ***********
        # frame_dict['w2c'] = w2c.tolist()
        # frame_dict['w2c_gt'] = w2c_gt.tolist()

        json_dict["frames"].append(frame_dict)

    with open(save_path, "w") as f:
        json.dump(json_dict, f, indent=2, cls=NumpyEncoder)
    print(f"Json file saved to {save_path}")



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

    # dataset = 'YCB'
    # dataset = 'BEHAVE'
    dataset = 'HO3D_v3'

    # seq_name = 'bleach0'
    # start_frame, step = 0, 3
    # main(dataset, seq_name, start_frame, step)

    for k, v in ho3d_list.items():
        main(dataset, k, v[0], v[1])
    
    

        
