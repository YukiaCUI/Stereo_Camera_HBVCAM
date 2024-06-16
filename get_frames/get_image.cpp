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
    // 打开拼接的双目摄像头
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }

    create_directories();

    int photo_count = 0;
    int total_photos = 20;
    int delay_seconds = 10;

    while (photo_count < total_photos) {
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

        // 倒计时显示
        for (int i = delay_seconds; i > 0; --i) {
            cap >> frame; // 继续读取新的帧
            if (frame.empty()) {
                std::cerr << "无法获取摄像头图像" << std::endl;
                break;
            }

            // 更新左右图像
            left_frame = frame(cv::Rect(0, 0, width / 2, height));
            right_frame = frame(cv::Rect(width / 2, 0, width / 2, height));

            cv::Mat countdown_frame = frame.clone();
            std::string countdown_text = "Taking photo in " + std::to_string(i) + " seconds";
            cv::putText(countdown_frame, countdown_text, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            cv::imshow("Left Camera", countdown_frame(cv::Rect(0, 0, width / 2, height)));
            cv::imshow("Right Camera", countdown_frame(cv::Rect(width / 2, 0, width / 2, height)));
            cv::waitKey(1000); // 延迟1秒
        }

        // 保存图像
        std::string left_filename = "left_frame/left_" + std::to_string(photo_count) + ".png";
        std::string right_filename = "right_frame/right_" + std::to_string(photo_count) + ".png";
        if (cv::imwrite(left_filename, left_frame)) {
            std::cout << "Saved left image: " << left_filename << std::endl;
        } else {
            std::cerr << "Failed to save left image: " << left_filename << std::endl;
        }
        if (cv::imwrite(right_filename, right_frame)) {
            std::cout << "Saved right image: " << right_filename << std::endl;
        } else {
            std::cerr << "Failed to save right image: " << right_filename << std::endl;
        }

        photo_count++;
    }

    // 释放摄像头资源
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
