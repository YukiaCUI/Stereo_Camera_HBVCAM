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

![dense pointcloud](./Trails/Dense_pointcloud.gif)
dense pointcloud