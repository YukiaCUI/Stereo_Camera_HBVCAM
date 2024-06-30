# Stereo_Camera_HBVCAM
A project of stereo camera with HBVCAM-W202011HD V33

## overview
![](./Trails/overview.png)

## fuction
- get_frames: capture frames with camera
- calibration：singel camera calibration & stereo camera calibration
- realtime_depthmap.cpp: 1. feature point(by SIFT); 2. generate realtime disparity and depth map
- 3d_point_cloud_sparse.cpp: realtime sparse pointcloud
- 3d_point_cloud_dense.cpp: realtime dense pointcloud

## example of dense pointcloud
![left frame](./Trails/Left_Camera.png)
![right frame](./Trails/Right_Camera.png)
![dense pointcloud](./Trails/稠密点云-动态.MP4)