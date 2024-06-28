#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>

#define Img_width 2560
#define Img_height 720


int main() {

    // 从文件加载标定结果
    cv::FileStorage fs("../calibration/calib_param.yml", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open calibration file." << std::endl;
        return -1;
    }

    cv::Mat cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T;
    fs["cameraMatrixL"] >> cameraMatrixL;
    fs["distCoeffsL"] >> distCoeffsL;
    fs["cameraMatrixR"] >> cameraMatrixR;
    fs["distCoeffsR"] >> distCoeffsR;
    fs["R"] >> R;
    fs["T"] >> T;

    fs.release();
    
    // 打开拼接的双目摄像头（根据自己的设备修改）
    cv::VideoCapture cap(2);

    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }

    // 图像大小
    cv::Size imageSize(Img_width / 2, Img_height);  // 根据你的图像实际大小设置


    /*
    alpha 参数说明
    alpha = -1：使用默认值，通常会保留所有有效像素，但可能会出现一些黑色边界。
    alpha = 0：裁剪到最小矩形以去除所有不需要的像素，从而最大化视差图的有效区域。
    alpha = 1：保留所有有效像素，视野最大化，但可能会包含更多的黑色边界。
    */
    // 立体校正
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(cameraMatrixL, distCoeffsL,
                    cameraMatrixR, distCoeffsR,
                    imageSize, R, T, R1, R2, P1, P2, Q,
                    cv::CALIB_ZERO_DISPARITY, 0);  // alpha = 0

    // std::cout << "R1: " << R1 << std::endl;
    // std::cout << "R2: " << R2 << std::endl;
    // std::cout << "P1: " << P1 << std::endl;
    // std::cout << "P2: " << P2 << std::endl;
    // std::cout << "Q: " << Q << std::endl;cv

    while (true) {
        cv::Mat frame;
        cap >> frame;  // 读取帧

        if (frame.empty()) {
            std::cerr << "无法获取摄像头图像" << std::endl;
            break;
        }


        // 拆分图像
        cv::Mat left_frame = frame(cv::Rect(0, 0, Img_width / 2, Img_height));
        cv::Mat right_frame = frame(cv::Rect(Img_width / 2, 0, Img_width / 2, Img_height));

        // 显示左右两个图像
        cv::imshow("Left Camera", left_frame);
        cv::imshow("Right Camera", right_frame);

        cv::Mat map1x, map1y, map2x, map2y;
        cv::initUndistortRectifyMap(cameraMatrixL, distCoeffsL, R1, P1, imageSize, CV_32FC1, map1x, map1y);
        cv::initUndistortRectifyMap(cameraMatrixR, distCoeffsR, R2, P2, imageSize, CV_32FC1, map2x, map2y);

        // 应用映射生成校正后的图像
        cv::Mat rectified_left, rectified_right;
        cv::remap(left_frame, rectified_left, map1x, map1y, cv::INTER_LINEAR);
        cv::remap(right_frame, rectified_right, map2x, map2y, cv::INTER_LINEAR);

        // // 显示校正后的图像
        // cv::imshow("Rectified Left Image", rectified_left);
        // cv::imshow("Rectified Right Image", rectified_right);
        


        // SIFT 特征提取
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;
        cv::Mat SIFT_left, SIFT_right;

        sift->detectAndCompute(rectified_left, cv::noArray(), keypoints1, descriptors1);
        sift->detectAndCompute(rectified_right, cv::noArray(), keypoints2, descriptors2);

        drawKeypoints(rectified_left, keypoints1, SIFT_left, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        drawKeypoints(rectified_right, keypoints2, SIFT_right, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        //显示SIFT特征的图像
        cv::imshow("SIFT L Camera", SIFT_left);
        cv::imshow("SIFT R Camera", SIFT_right);

        // // 检查提取的关键点和描述符数量
        // std::cout << "Number of keypoints in left image: " << keypoints1.size() << std::endl;
        // std::cout << "Number of keypoints in right image: " << keypoints2.size() << std::endl;
        // std::cout << "Number of descriptors in left image: " << descriptors1.rows << std::endl;
        // std::cout << "Number of descriptors in right image: " << descriptors2.rows << std::endl;

        



        if(cv::waitKey(15) >= 0)
        {
            break;
        }

    }

    return 0;
}
