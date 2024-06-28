#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>

#define Img_width 2560
#define Img_height 720

using namespace std;
using namespace cv;

void draw_matches(const Mat &img1, const Mat &img2, const vector<pair<pair<float, float>, pair<float, float>>> &matches)
{
    // 将图像拼接成一张图
    Mat img_matches;
    hconcat(img1, img2, img_matches);
    
    for (auto &match : matches)
    {
        Point2f p1(match.first.first, match.first.second);
        Point2f p2(match.second.first + img1.cols, match.second.second);
        line(img_matches, p1, p2, Scalar(0, 0, 255), 2);
    }
    // resize
    resize(img_matches, img_matches, Size(img_matches.cols / 2, img_matches.rows / 2));
    // 显示匹配结果
    imshow("matches", img_matches);
    while (true)
    {
        if (waitKey(10) == 27)
        {
            destroyAllWindows();
            break;
        }
    }
}

void stereoBM(cv::Mat lpng,cv::Mat rpng,cv::Mat &disp)
{
    disp.create(lpng.rows,lpng.cols,CV_16S);
    cv::Mat disp1 = cv::Mat(lpng.rows,lpng.cols,CV_8UC1);
    cv::Size imgSize = lpng.size();
    cv::Rect roi1,roi2;
    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16,9);

    int nmDisparities = ((imgSize.width / 8) + 15) & -16;//视差搜索范围

    bm->setPreFilterType(CV_STEREO_BM_NORMALIZED_RESPONSE);//预处理滤波器类型
    bm->setPreFilterSize(9);//预处理滤波器窗口大小
    bm->setPreFilterCap(31);//预处理滤波器截断值
    bm->setBlockSize(9);//SAD窗口大小
    bm->setMinDisparity(0);//最小视差
    bm->setNumDisparities(nmDisparities);//视差搜索范围
    bm->setTextureThreshold(10);//低纹理区域的判断阈值
    bm->setUniquenessRatio(5);//视差唯一性百分比
    bm->setSpeckleWindowSize(100);//检查视差连通区域变化度窗口大小
    bm->setSpeckleRange(32);//视差变化阈值
    bm->setROI1(roi1);
    bm->setROI2(roi2);
    bm->setDisp12MaxDiff(1);//左右视差图最大容许差异
    bm->compute(lpng,rpng,disp);

    disp.convertTo(disp1,CV_8U,255 / (nmDisparities*16.));

    cv::imshow("disp_img",disp1);
}

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

    cap.set(cv::CAP_PROP_FRAME_WIDTH, Img_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, Img_height);

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
                    imageSize, R, -T, R1, R2, P1, P2, Q,
                    CALIB_ZERO_DISPARITY, 0);  // alpha = 0

    // cout << "R1: " << R1 << endl;
    // cout << "R2: " << R2 << endl;
    // cout << "P1: " << P1 << endl;
    // cout << "P2: " << P2 << endl;
    // cout << "Q: " << Q << endl;

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

        Mat map1x, map1y, map2x, map2y;
        initUndistortRectifyMap(cameraMatrixL, distCoeffsL, R1, P1, imageSize, CV_32FC1, map1x, map1y);
        initUndistortRectifyMap(cameraMatrixR, distCoeffsR, R2, P2, imageSize, CV_32FC1, map2x, map2y);

        // 应用映射生成校正后的图像
        Mat rectified_left, rectified_right;
        remap(left_frame, rectified_left, map1x, map1y, INTER_LINEAR);
        remap(right_frame, rectified_right, map2x, map2y, INTER_LINEAR);

        // // 显示校正后的图像
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

        // //显示SIFT特征的图像
        // imshow("SIFT L Camera", SIFT_left);
        // imshow("SIFT R Camera", SIFT_right);

        // // 检查提取的关键点和描述符数量
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
        std::vector<cv::DMatch> y_filtered_matches;
        const float y_threshold = 10.0; // y坐标差值阈值，根据需要调整
        for (const auto& match : good_matches) {
            cv::Point2f pt1 = keypoints1[match.queryIdx].pt;
            cv::Point2f pt2 = keypoints2[match.trainIdx].pt;
            if (std::abs(pt1.y - pt2.y) < y_threshold) {
                y_filtered_matches.push_back(match);
            }
        }

        // 绘制匹配结果
        cv::Mat img_matches;
        cv::drawMatches(rectified_left, keypoints1, rectified_right, keypoints2, y_filtered_matches, img_matches);
        // cv::imshow("Image Matches", img_matches);

        cv::Mat gray_left, gray_right, disparity;
        // cv::cvtColor(rectified_left, gray_left, cv::COLOR_BGR2GRAY);
        // cv::cvtColor(rectified_right, gray_right, cv::COLOR_BGR2GRAY);
        gray_left = cv::imread("../calibration/cone/dispL.png");
        gray_right = cv::imread("../calibration/cone/dispR.png");

        // // 创建StereoBM对象
        // int numDisparities = 16; // 视差搜索范围的数量
        // int blockSize = 9;       // 块匹配窗口的大小
        // cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(numDisparities, blockSize);

        // // 计算视差图
        // stereo->compute(gray_left, gray_right, disparity);

        // // 归一化视差图以便显示
        // cv::Mat disparity_normalized;
        // cv::normalize(disparity, disparity_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);

        stereoBM(gray_left,gray_right,disparity);

        // 计算深度图
        cv::Mat depth_map;
        cv::reprojectImageTo3D(disparity, depth_map, Q, true);
        
        // 提取深度信息并归一化
        cv::Mat depth_normalized;
        cv::normalize(depth_map, depth_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);

        // 显示视差图
        cv::imshow("Disparity", disparity_normalized);
        // 显示深度图
        cv::imshow("Depth Map", depth_map);

        if(waitKey(15) >= 0)
        {
            break;
        }

    }

    return 0;
}
