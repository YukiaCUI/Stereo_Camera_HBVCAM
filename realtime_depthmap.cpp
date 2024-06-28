#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <filesystem>

void create_directories() {
    if (!std::filesystem::exists("left_frame")) {
        std::filesystem::create_directory("left_frame");
    }
    if (!std::filesystem::exists("right_frame")) {
        std::filesystem::create_directory("right_frame");
    }
}

int main() {
    // 从文件加载标定结果
    cv::FileStorage fs("./calibration/calib_param.yml", cv::FileStorage::READ);
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

    // 打开拼接的双目摄像头
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }

    create_directories();

    while (true) {
        cv::Mat frame;
        cap >> frame;  // 读取帧

        if (frame.empty()) {
            std::cerr << "无法获取摄像头图像" << std::endl;
            break;
        }

        // 根据实际分辨率调整
        int width = frame.cols;
        int height = frame.rows;

        // 拆分图像
        cv::Mat left_frame = frame(cv::Rect(0, 0, width / 2, height));
        cv::Mat right_frame = frame(cv::Rect(width / 2, 0, width / 2, height));

        // 显示左右两个图像
        cv::imshow("Left Camera", left_frame);
        cv::imshow("Right Camera", right_frame);

        // 图像大小
        cv::Size imageSize(width / 2, height);  // 根据你的图像实际大小设置

        // 立体校正
        cv::Mat R1, R2, P1, P2, Q;
        cv::stereoRectify(cameraMatrixL, distCoeffsL,
                        cameraMatrixR, distCoeffsR,
                        imageSize, R, T, R1, R2, P1, P2, Q);

        // std::cout << "R1: " << R1 << std::endl;
        // std::cout << "R2: " << R2 << std::endl;
        // std::cout << "P1: " << P1 << std::endl;
        // std::cout << "P2: " << P2 << std::endl;
        // std::cout << "Q: " << Q << std::endl;

        cv::Mat map1x, map1y, map2x, map2y;
        cv::initUndistortRectifyMap(cameraMatrixL, distCoeffsL, R1, P1, imageSize, CV_32FC1, map1x, map1y);
        cv::initUndistortRectifyMap(cameraMatrixR, distCoeffsR, R2, P2, imageSize, CV_32FC1, map2x, map2y);

        // 应用映射生成校正后的图像
        cv::Mat rectified_left, rectified_right;
        cv::remap(left_frame, rectified_left, map1x, map1y, cv::INTER_LINEAR);
        cv::remap(right_frame, rectified_right, map2x, map2y, cv::INTER_LINEAR);

        // 显示校正后的图像
        cv::imshow("Rectified Left Image", rectified_left);
        cv::imshow("Rectified Right Image", rectified_right);
    }

    return 0;
}
