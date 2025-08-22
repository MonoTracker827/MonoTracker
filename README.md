<p align="center">
  <h1 align="center">MonoTracker: Monocular RGB-Only 6D Tracking of Unknown Objects</h1>
  <h2 align="center">BMVC 2025 (Oral)</h2>
  <p align="center">
    <a href="https://dzl666.github.io/">Zilong Deng</a><sup>1,2</sup></span>,
    <a href="https://scholar.google.com/citations?user=2ivpy00AAAAJ">Shaochang Tan</a><sup>1,2</sup>
    <br>
    <a href="https://zuriabauer.com/">Zuria Bauer</a><sup>2</sup>,
    <a href="https://people.inf.ethz.ch/pomarc/">Marc Pollefeys</a><sup>2,3</sup>,
    <a href="https://scholar.google.com/citations?user=U9-D8DYAAAAJ">Daniel Barath</a><sup>2,4</sup>,
    <br>
    <sup>1</sup>Univeristy of Zurich,
    <sup>2</sup>ETH Zurich,
    <sup>3</sup>Microsoft,
    <sup>4</sup>HUN-REN SZTAKI
  </p>
  <h3 align="center"><a href="">Paper</a> | <a href="https://monotracker827.github.io">Project Page</a> </h3>
  <div align="center"></div>
</p>

This is the official implementation of the BMVC2025 paper **MonoTracker**.

### BibTex

Coming soon...
<!-- <pre><code>
@inproceedings{engelmann2024opennerf,
   title={{OpenNerf: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views}},
   author={Engelmann, Francis and Manhardt, Fabian and Niemeyer, Michael and Tateno, Keisuke and Pollefeys, Marc and Tombari, Federico},
   booktitle={International Conference on Learning Representations (ICLR)},
   year={2024}
}</code></pre> -->

## Installation

### Environment setup

We suggest to use conda environment. Our code is tested under python 3.8.10.

```bash
# necessay building tools
sudo apt-get install cmake

# conda env
conda create -n monotracker python=3.8.10
conda activate monotracker

# select the version according to your CUDA driver
python -m pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

```

### Install third party tools

Download the checkpoint `outdoor_ds.ckpt` from LoFTR [Google Drive](https://drive.google.com/drive/folders/1xu2Pq6mZT5hmFgiYMBT9Zt8h1yO-3SIp) and put it under `third_party/LoFTR/weigths`

```bash
conda activate monotracker
mkdir third_party && cd third_party

git clone https://github.com/zju3dv/LoFTR.git
cd LoFTR && mkdir weights
cd ..

wget http://ceres-solver.org/ceres-solver-2.1.0.tar.gz
tar zxf ceres-solver-2.1.0.tar.gz
rm ceres-solver-2.1.0.tar.gz
mkdir ceres-bin && cd ceres-bin
cmake ../ceres-solver-2.1.0
make -j8
```

### Installation of DepthAnything-v2

Please follow the official instructions: [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)

### Installation of XMem

Please follow the official instructions: [XMem](https://github.com/hkchengrex/XMem)

## Data Preparation

### Download public datasets

- YCBInEOAT

- BEHAVE

### Prepare your files

Prepare your RGB video folder as below:

```text
[data_root]
  ├──rgb/       (RGB images)
  └──cam_K.txt  (3x3 intrinsic matrix, use space and enter to delimit)
```

You will need to create a initial object mask to specify the region you want to track, as stated in the paper.

Please run the monocular depth prdiction (Depth-Anything-v2) and mask prediction (XMem) beforehand and put them into the folder.
**NOTE: Their files names should be the same as the corresponding RGB images.**

```text
[output_folder]
  ├──pred_depth/  (PNG files, stored in mm, in uint16.)
  └──pred_mask/   (PNG files, including the initial one. 0: background, 1: ROI)
```

### Start to Track

```bash
cd src
python main_mp.py
```

## Future Improvements

- We tried to speed up the joint pose-scale optimiztion by using a CUDA version of Ceres. But we failed to make it run under our settings.

## Others

### Results

Output Folder: **[Your Folder]/[dataset_name]/[sequence_name]/result_files**

- Preidcted Poses: **[Output Folder]/global_pose**
- Projected Bounding Box with preidcted Poses: **[Output Folder]/global_pose_vis**
- Predicted Scale Factors: **[Output Folder]/scale_shift** and **[Output Folder]/Dscale.npy**
- Cache of the correspondences from the feature matcher: **[Output Folder]/corres_cache**

### Logs

Log Folder: **logs/[sequence_name]**

- Plot of the changes of scale factors: **[Log Folder]/scale_factors.png**
- Visualization during tracking: **[Log Folder]/opt_vis/**
- Comparison of beforeBA and afterBA: **[Log Folder]/BA_corres/**
- Pickel of the frame manager: **[Log Folder]/frame_manager.pkl**
-

### usage of helper_func/process_logs.py

Every time you run an optimization, a log file will be generated on `log_pickle` folder, which records all the frames scale shift and pose before and after optimization. You can visualize the log file by running the following command:

```bash
python log_viz.py -g # visualize the 3d point cloud before and after applying the grid scale and shift optimization
python log_viz.py -o # visualize the connection graph of the optimization, and the errors between 2 selected frames.
python log_viz.py -p # draw the pose bbox and export traj files

```
