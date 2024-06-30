# Stereo_Camera_HBVCAM
A project of stereo camera with HBVCAM-W202011HD V33

## Overview
![](./Trails/overview.png)

## Fuction
- get_frames: capture frames with camera
- calibration：singel camera calibration & stereo camera calibration
- realtime_depthmap.cpp: 1. feature point(by SIFT); 2. generate realtime disparity and depth map
- 3d_point_cloud_sparse.cpp: realtime sparse pointcloud
- 3d_point_cloud_dense.cpp: realtime dense pointcloud

## SIFT
### original frame
![](./Trails/SIFT特征点.png)
### matches before filtering
![](./Trails/goodmatches.png)
### matches after filtering
![](./Trails/过滤后的匹配.png)

## Disparity and Depth map
![](./Trails/深度图和视差图v2.png)

## Pointcloud

<table>
  <tr>
    <td>
      <img src="./Trails/Left_Camera.png" alt="Left Camera" width="400"/>
      <p align="center">Left Frame</p>
    </td>
    <td>
      <img src="./Trails/Right_Camera.png" alt="Right Camera" width="400"/>
      <p align="center">Right Frame</p>
    </td>
  </tr>
</table>

### sparse pointcloud
![sparse pointcloud](./Trails/稀疏点云图.png)

### dense pointcloud
![dense pointcloud](./Trails/Dense_pointcloud.gif)
