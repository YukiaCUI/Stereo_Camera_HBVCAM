#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include "./includes/disparity.hpp"

#define Img_width 2560
#define Img_height 720

using namespace std;
using namespace cv;

int main() {

    // 从文件加载标定结果
    FileStorage fs("../calibration/calib_param.yml", FileStorage::READ);
    if (!fs.isOpened()) {
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
    VideoCapture cap(2);

    if (!cap.isOpened()) {
        cerr << "无法打开摄像头" << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, Img_width);
    cap.set(CAP_PROP_FRAME_HEIGHT, Img_height);

    // 图像大小
    Size imageSize(Img_width / 2, Img_height);  // 根据你的图像实际大小设置

    /*
    alpha 参数说明
    alpha = -1：使用默认值，通常会保留所有有效像素，但可能会出现一些黑色边界。
    alpha = 0：裁剪到最小矩形以去除所有不需要的像素，从而最大化视差图的有效区域。
    alpha = 1：保留所有有效像素，视野最大化，但可能会包含更多的黑色边界。
    */
    // 立体校正
    Mat R1, R2, P1, P2, Q;
    stereoRectify(cameraMatrixL, distCoeffsL,
                    cameraMatrixR, distCoeffsR,
                    imageSize, R, T, R1, R2, P1, P2, Q,
                    CALIB_ZERO_DISPARITY, 0);  // alpha = 0

    // cout << "R1: " << R1 << endl;
    // cout << "R2: " << R2 << endl;
    // cout << "P1: " << P1 << endl;
    // cout << "P2: " << P2 << endl;
    // cout << "Q: " << Q << endl;

    int photo_count = 0;

    while (true) {
        Mat frame;
        cap >> frame;  // 读取帧

        if (frame.empty()) {
            cerr << "无法获取摄像头图像" << endl;
            break;
        }

        // 拆分图像
        Mat left_frame = frame(Rect(0, 0, Img_width / 2, Img_height));
        Mat right_frame = frame(Rect(Img_width / 2, 0, Img_width / 2, Img_height));

        // 显示左右两个图像
        imshow("Left Camera", left_frame);
        imshow("Right Camera", right_frame);

        // 保存左右两个图像
        // imwrite("Left_Camera.png", left_frame);
        // imwrite("Right_Camera.png", right_frame);

        Mat map1x, map1y, map2x, map2y;
        initUndistortRectifyMap(cameraMatrixL, distCoeffsL, R1, P1, imageSize, CV_32FC1, map1x, map1y);
        initUndistortRectifyMap(cameraMatrixR, distCoeffsR, R2, P2, imageSize, CV_32FC1, map2x, map2y);

        // 应用映射生成校正后的图像
        Mat rectified_left, rectified_right;
        remap(left_frame, rectified_left, map1x, map1y, INTER_LINEAR);
        remap(right_frame, rectified_right, map2x, map2y, INTER_LINEAR);

        // 显示校正后的图像
        // imshow("Rectified Left Image", rectified_left);
        // imshow("Rectified Right Image", rectified_right);

        // SIFT 特征提取
        Ptr<SIFT> sift = SIFT::create();
        vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;
        Mat SIFT_left, SIFT_right;

        sift->detectAndCompute(rectified_left, noArray(), keypoints1, descriptors1);
        sift->detectAndCompute(rectified_right, noArray(), keypoints2, descriptors2);

        drawKeypoints(rectified_left, keypoints1, SIFT_left, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        drawKeypoints(rectified_right, keypoints2, SIFT_right, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        // 显示SIFT特征的图像
        // imshow("SIFT L Camera", SIFT_left);
        // imshow("SIFT R Camera", SIFT_right);

        // 检查提取的关键点和描述符数量
        // cout << "Number of keypoints in left image: " << keypoints1.size() << endl;
        // cout << "Number of keypoints in right image: " << keypoints2.size() << endl;
        // cout << "Number of descriptors in left image: " << descriptors1.rows << endl;
        // cout << "Number of descriptors in right image: " << descriptors2.rows << endl;

        // 特征点匹配
        BFMatcher matcher(NORM_L2);
        vector<vector<DMatch>> knn_matches;
        matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

        // 应用比率测试来过滤匹配
        const float ratio_thresh = 0.75f;
        vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        // 检查匹配索引是否有效
        for (const auto& match : good_matches) {
            if (match.queryIdx < 0 || match.queryIdx >= keypoints1.size() ||
                match.trainIdx < 0 || match.trainIdx >= keypoints2.size()) {
                cerr << "Invalid match index: " << match.queryIdx << ", " << match.trainIdx << endl;
                return -1;
            }
        }

        // 根据y坐标差值进行过滤
        std::vector<DMatch> y_filtered_matches;
        const float y_threshold = 10.0; // y坐标差值阈值，根据需要调整
        for (const auto& match : good_matches) {
            Point2f pt1 = keypoints1[match.queryIdx].pt;
            Point2f pt2 = keypoints2[match.trainIdx].pt;
            if (std::abs(pt1.y - pt2.y) < y_threshold) {
                y_filtered_matches.push_back(match);
            }
        }

        // 绘制匹配结果
        Mat img_matches;
        drawMatches(rectified_left, keypoints1, rectified_right, keypoints2, y_filtered_matches, img_matches);
        // imshow("Image Matches", img_matches);

        Mat gray_left, gray_right, disparity;
        cvtColor(rectified_left, gray_left, COLOR_BGR2GRAY);
        cvtColor(rectified_right, gray_right, COLOR_BGR2GRAY);

        
        stereoBM(gray_left,gray_right,disparity);
        stereoSGBM(gray_left,gray_right,disparity);

        // imwrite("gray_left.png",gray_left);
        // imwrite("gray_right.png",gray_right);
        
        // double fx = P1.at<double>(0, 0);
        // double baseline = norm(T, NORM_L2);

        // Mat depth(disparity.rows, disparity.cols, CV_16S);  //深度图
        // //视差图转深度图
        // for (int row = 0; row < depth.rows; row++){
        //     for (int col = 0; col < depth.cols; col++){
        //         short d = disparity.ptr<uchar>(row)[col];
        //         if (d == 0)
        //             continue;

        //         depth.ptr<short>(row)[col] = fx * baseline / d;
        //     }
        // }
        // cout << "depth: " << depth.type() << endl;
        // 提取深度信息并归一化
        
        Mat depth_map_3D;
        reprojectImageTo3D(disparity, depth_map_3D, Q, true); // 8UC1 -> 32FC3
        std::vector<Mat> channels(3);
        split(depth_map_3D, channels);
        Mat depth = channels[2]; // 32FC3 -> 32FC1

        depth.setTo(0, depth < 0);
        depth.setTo(4000, depth > 4000);

        Mat depth_normalized;
        normalize(depth, depth_normalized, 0, 255, NORM_MINMAX, CV_8UC1); // 32FC3 -> 8UC1
        
        imshow("depth_normalized",depth_normalized);

        // A waitkey() is necessary for camera
        if(waitKey(15) >= 0){
            break;
        }
    }

    return 0;
}