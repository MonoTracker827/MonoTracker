import sys, os
import json
# import rerun as rr
import numpy as np
import scipy.optimize as opt
import imageio.v2 as imageio

data_path = "/media/Q/eulerBackUp/dataset/sugar_box_yalehand0/"
predicted_depth_folder = data_path + "depth_m3dv2"
gt_depth_folder = data_path + "depth"
K = np.loadtxt(data_path + "cam_K.txt")

def cal_scale_to_ref_frame(ref_frame, frame):
    """
    given a reference frame and a frame img path, calculate the scale and shift to make the frame img
    """
    print(f"ref_frame: {ref_frame}, frame: {frame}")
    frame_depth = imageio.imread(frame)
    ref_frame_depth = imageio.imread(ref_frame)
    # only consider the valid depth value in ref_frame
    frame_depth = frame_depth[ref_frame_depth != 0]
    ref_frame_depth = ref_frame_depth[ref_frame_depth != 0]
    # ref = a * frame + b, using huber loss to estimate a and b
    # define the loss function
    def loss_func(x):
        a, b = x
        sigma = 1
        # huber loss
        return np.sum(
            np.where(
                np.abs(a * frame_depth + b - ref_frame_depth) < sigma,
                0.5 * (a * frame_depth + b - ref_frame_depth) ** 2,
                sigma * np.abs(a * frame_depth + b - ref_frame_depth) - 0.5 * sigma ** 2)
            )
    # optimize
    res = opt.minimize(loss_func, [1, 0])
    return res.x
frames_name = os.listdir(predicted_depth_folder)
frames_name.sort()

####################################################
#region: align the predicted depth to the gt depth
# from tqdm import tqdm
# import multiprocessing as mp
# manager = mp.Manager()
# ab_dict = manager.dict()
# def process_frame(frame_name):
#     predicted_depth_file = os.path.join(predicted_depth_folder, frame_name)
#     gt_depth_file = os.path.join(gt_depth_folder, frame_name)
#     a, b = cal_scale_to_ref_frame(ref_frame = gt_depth_file,
#                                frame = predicted_depth_file)
#     print(f"{frame_name}: a={a}, b={b}")
#     ab_dict[frame_name] = (a, b)
# pool = mp.Pool(mp.cpu_count())
# for _ in tqdm(pool.imap_unordered(process_frame, frames_name), total=len(frames_name)):
#     pass
# pool.close()
# pool.join()
# # save the result
# json.dump(dict(ab_dict), open(data_path+"/m3dv2_scale_to_gt.json", "w"))
#endregion
###############END:align the predicted depth to the gt depth############

####################################################
#region: visualize the depth consistency
import rerun as rr
rr.init("depth_consistency", spawn=True)
ab_dict = json.load(open(data_path+"/m3dv2_scale_to_gt.json", "r"))
for i,frame_name in enumerate(frames_name):
    a = ab_dict[frame_name][0]
    b = ab_dict[frame_name][1] * 1e-3
    # print(f"processing {frame_name}")
    rr.set_time_sequence("seq",i)
    predicted_depth_file = os.path.join(predicted_depth_folder, frame_name)
    gt_depth_file = os.path.join(gt_depth_folder, frame_name)
    rgb_file = os.path.join(data_path, "rgb", frame_name)
    gt_depth = imageio.imread(gt_depth_file)
    pre_depth = imageio.imread(predicted_depth_file)
    rgb = imageio.imread(rgb_file)
    # remove the alpha channel if exists
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    # 3d points in predicted depth
    # homogeneous coordinates
    u, v = np.meshgrid(np.arange(gt_depth.shape[1]), np.arange(gt_depth.shape[0]))
    u = u.flatten()
    v = v.flatten()
    rgb = rgb.reshape(-1, 3)
    # map to camera coordinate
    z = pre_depth.flatten() * 1e-3 # convert to meter
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    # filter out invalid depth
    valid_idx = np.logical_and(z > 0, z < 1.2)
    x = x[valid_idx]
    y = y[valid_idx]
    z = z[valid_idx]
    rgb = rgb[valid_idx]
    x = x[::10]
    y = y[::10]
    z = z[::10]
    rgb = rgb[::10]
    # log the 3d points
    pre_points = np.stack([x, y, z], axis=1)
    recover_z = a * z + b
    recover_points = np.stack([x, y, recover_z], axis=1)
    # generate colors from depth use viridis colormap in matplotlib
    import matplotlib.cm as cm
    # pre_colors = cm.viridis(z.flatten()/z.max())
    # recover_colors = cm.viridis(recover_z.flatten()/recover_z.max())
    rr.log("world/pre_points", rr.Points3D(pre_points, colors=rgb))
    rr.log("world/recover_points", rr.Points3D(recover_points, colors=rgb))

    ## for gt depth
    # homogeneous coordinates
    u, v = np.meshgrid(np.arange(gt_depth.shape[1]), np.arange(gt_depth.shape[0]))
    u = u.flatten()
    v = v.flatten()
    # map to camera coordinate
    z = gt_depth.flatten() * 1e-3 # convert to meter
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    # filter out invalid depth
    valid_idx = np.logical_and(z > 0, z < 1.2)
    x = x[valid_idx]
    y = y[valid_idx]
    z = z[valid_idx]
    x = x[::10]
    y = y[::10]
    z = z[::10]
    # log the 3d points
    gt_points = np.stack([x, y, z], axis=1)
    # generate colors from depth use viridis colormap in matplotlib
    import matplotlib.cm as cm
    gt_colors = cm.viridis(z.flatten()/z.max())
    rr.log("world/gt_points", rr.Points3D(gt_points, colors=gt_colors))
    
    # rr.log(
    #     "world/gt_cam",
    #     rr.Pinhole(
    #         width=gt_depth.shape[1],
    #         height=gt_depth.shape[0],
    #         focal_length=200,
    #     )
    # )
    # rr.log("world/gt_cam/depth",
    #         rr.DepthImage(gt_depth,meter=10_000.0,colormap="viridis"))
    # rr.log(
    #     "world/pre_cam",
    #     rr.Pinhole(
    #         width=pre_depth.shape[1],
    #         height=pre_depth.shape[0],
    #         focal_length=200,
    #     )
    # )
    # rr.log("world/pre_cam/depth",
    #         rr.DepthImage(pre_depth,meter=10_000.0,colormap="viridis"))
#endregion    
    
