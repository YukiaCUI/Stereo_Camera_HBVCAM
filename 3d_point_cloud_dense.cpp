#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>
#include "./includes/show_pointcloud.hpp"

#define Img_width 2560
#define Img_height 720

using namespace std;
using namespace cv;

int main()
{
    // 从文件加载标定结果
    FileStorage fs("../calibration/calib_param.yml", FileStorage::READ);
    if (!fs.isOpened())
    {
        cerr << "Failed to open calibration file." << endl;
        return -1;
    }

    Mat cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T;
    fs["cameraMatrixL"] >> cameraMatrixL;
    fs["distCoeffsL"] >> distCoeffsL;
    fs["cameraMatrixR"] >> cameraMatrixR;
    fs["distCoeffsR"] >> distCoeffsR;
    fs["R"] >> R;
    fs["T"] >> T;

    fs.release();

    // 打开拼接的双目摄像头（根据自己的设备修改）
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "无法打开摄像头" << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, Img_width);
    cap.set(CAP_PROP_FRAME_HEIGHT, Img_height);

    // 图像大小
    Size imageSize(Img_width / 2, Img_height); // 根据你的图像实际大小设置

    // 立体校正
    Mat R1, R2, P1, P2, Q;
    stereoRectify(cameraMatrixL, distCoeffsL,
                  cameraMatrixR, distCoeffsR,
                  imageSize, R, -T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 0); // alpha = 0

    Mat map1x, map1y, map2x, map2y;
    initUndistortRectifyMap(cameraMatrixL, distCoeffsL, R1, P1, imageSize, CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(cameraMatrixR, distCoeffsR, R2, P2, imageSize, CV_32FC1, map2x, map2y);

    while (true)
    {
        Mat frame;
        cap >> frame; // 读取帧

        if (frame.empty())
        {
            cerr << "无法获取摄像头图像" << endl;
            break;
        }

        // 拆分图像
        Mat left_frame = frame(Rect(0, 0, Img_width / 2, Img_height));
        Mat right_frame = frame(Rect(Img_width / 2, 0, Img_width / 2, Img_height));

        // //测试图像
        // Mat left_frame = imread("../calibration/cone/imgL.png", IMREAD_GRAYSCALE);
        // Mat right_frame = imread("../calibration/cone/imgR.png", IMREAD_GRAYSCALE);

        // 应用映射生成校正后的图像
        Mat rectified_left, rectified_right;
        remap(left_frame, rectified_left, map1x, map1y, INTER_LINEAR);
        remap(right_frame, rectified_right, map2x, map2y, INTER_LINEAR);


        double fx = cameraMatrixL.at<double>(0, 0);
        double fy = cameraMatrixL.at<double>(1, 1);
        double cx = cameraMatrixL.at<double>(0, 2);
        double cy = cameraMatrixL.at<double>(1, 2);
        double b  = norm(T, NORM_L2)/1000.0;


        /*************************稠密点云*********************/
        Mat gray_left, gray_right, disparity, disparity_float;
        cvtColor(rectified_left, gray_left, COLOR_BGR2GRAY);
        cvtColor(rectified_right, gray_right, COLOR_BGR2GRAY);

        Ptr<StereoSGBM> sgbm = StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    
        Mat disparity_sgbm;
        sgbm->compute(gray_left, gray_right, disparity_sgbm);
        disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);
        
        
        
        
        vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud_dense;
        for (int v = 0; v < gray_left.rows; v++)
            for (int u = 0; u < gray_left.cols; u++) {
                if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) continue;

                Vector4d point(0, 0, 0, gray_left.at<uchar>(v, u) / 255.0); // 前三维为xyz,第四维为颜色

                // 根据双目模型计算 point 的位置
                double x = (u - cx) / fx;
                double y = (v - cy) / fy;
                double depth = fx * b / (disparity.at<float>(v, u));
                point[0] = x * depth;
                point[1] = y * depth;
                point[2] = depth;

                pointcloud_dense.push_back(point);
            }

        imshow("disparity", disparity / 96.0);
        waitKey(0);
        // 画出点云
        showPointCloud(pointcloud_dense);

        if (waitKey(15) >= 0)
        {
            break;
        }
    }

    return 0;
}
