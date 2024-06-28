#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <iostream>
#include <chrono>
#include <thread>

#define Img_width 2560
#define Img_height 720

using namespace std;
using namespace cv;
using namespace Eigen;

void showPointCloud(const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> &pointcloud)
{   
    if (pointcloud.empty())
    {
        std::cerr << "Point cloud is empty!" << std::endl;
        return;
    }

    
    // 创建窗口
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        std::cerr << "Error initializing GLEW: " << glewGetErrorString(err) << std::endl;
        return;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 10000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0));

    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glPointSize(5);
        glBegin(GL_POINTS);
        for (auto &p : pointcloud)
        {
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    pangolin::DestroyWindow("Point Cloud Viewer");
}

void stereoSGBM(cv::Mat lpng,cv::Mat rpng,cv::Mat &disp)
{
    cv::GaussianBlur(lpng, lpng, cv::Size(5, 5), 1.5);
    cv::GaussianBlur(rpng, rpng, cv::Size(5, 5), 1.5);
    cv::pyrDown(lpng, lpng);
    cv::pyrDown(rpng, rpng);
    cv::GaussianBlur(lpng, lpng, cv::Size(5, 5), 1.5);
    cv::GaussianBlur(rpng, rpng, cv::Size(5, 5), 1.5);

    disp.create(lpng.rows,lpng.cols,CV_16S);
    cv::Mat disp1 = cv::Mat(lpng.rows,lpng.cols,CV_8UC1);
    cv::Size imgSize = lpng.size();
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create();
    cv::Mat disp_color;

    int nmDisparities = ((imgSize.width / 8) + 15) & -16;//视差搜索范围
    int pngChannels = lpng.channels();//获取左视图通道数
    int winSize = 9;

    sgbm->setPreFilterCap(31);//预处理滤波器截断值
    sgbm->setBlockSize(winSize);//SAD窗口大小
    sgbm->setP1(16*pngChannels*winSize*winSize);//控制视差平滑度第一参数
    sgbm->setP2(32*pngChannels*winSize*winSize);//控制视差平滑度第二参数
    sgbm->setMinDisparity(0);//最小视差
    sgbm->setNumDisparities(nmDisparities);//视差搜索范围
    sgbm->setUniquenessRatio(10);//视差唯一性百分比
    sgbm->setSpeckleWindowSize(200);//检查视差连通区域变化度的窗口大小
    sgbm->setSpeckleRange(32);//视差变化阈值
    sgbm->setDisp12MaxDiff(1);//左右视差图最大容许差异
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM);//采用全尺寸双通道动态编程算法
    sgbm->compute(lpng,rpng,disp);

    disp.convertTo(disp1,CV_8U,255 / (nmDisparities*16.));//转8位

    cv::applyColorMap(disp1,disp_color,cv::COLORMAP_JET);//转彩色图

    cv::imshow("SGBM_img",disp1);
    cv::imshow("SGBM_color",disp_color);
}

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
    VideoCapture cap(2);
    if (!cap.isOpened())
    {
        cerr << "无法打开摄像头" << endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, Img_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, Img_height);

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
        // Mat left_frame = cv::imread("../calibration/cone/imgL.png", cv::IMREAD_GRAYSCALE);
        // Mat right_frame = cv::imread("../calibration/cone/imgR.png", cv::IMREAD_GRAYSCALE);

        // 应用映射生成校正后的图像
        Mat rectified_left, rectified_right;
        remap(left_frame, rectified_left, map1x, map1y, INTER_LINEAR);
        remap(right_frame, rectified_right, map2x, map2y, INTER_LINEAR);

        // SIFT 特征提取
        Ptr<SIFT> sift = SIFT::create();
        vector<KeyPoint> keypoints1, keypoints2;
        Mat descriptors1, descriptors2;
        sift->detectAndCompute(rectified_left, noArray(), keypoints1, descriptors1);
        sift->detectAndCompute(rectified_right, noArray(), keypoints2, descriptors2);

        // 特征点匹配
        BFMatcher matcher(NORM_L2);
        vector<vector<DMatch>> knn_matches;
        matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

        // 应用比率测试来过滤匹配
        const float ratio_thresh = 0.75f;
        vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        // 检查匹配索引是否有效
        for (const auto &match : good_matches)
        {
            if (match.queryIdx < 0 || match.queryIdx >= keypoints1.size() ||
                match.trainIdx < 0 || match.trainIdx >= keypoints2.size())
            {
                cerr << "Invalid match index: " << match.queryIdx << ", " << match.trainIdx << endl;
                return -1;
            }
        }

        // 根据y坐标差值进行过滤
        std::vector<cv::DMatch> y_filtered_matches;
        const float y_threshold = 10.0; // y坐标差值阈值，根据需要调整
        for (const auto &match : good_matches)
        {
            cv::Point2f pt1 = keypoints1[match.queryIdx].pt;
            cv::Point2f pt2 = keypoints2[match.trainIdx].pt;
            if (std::abs(pt1.y - pt2.y) < y_threshold)
            {
                y_filtered_matches.push_back(match);
            }
        }

        // 绘制匹配结果
        cv::Mat img_matches;
        cv::drawMatches(rectified_left, keypoints1, rectified_right, keypoints2, y_filtered_matches, img_matches);
        cv::imshow("Image Matches", img_matches);

        double fx = cameraMatrixL.at<double>(0, 0);
        double fy = cameraMatrixL.at<double>(1, 1);
        double cx = cameraMatrixL.at<double>(0, 2);
        double cy = cameraMatrixL.at<double>(1, 2);
        double baseline = norm(T, cv::NORM_L2);


        /*************************稠密点云*********************/
        cv::Mat gray_left, gray_right, disparity;
        // cv::cvtColor(rectified_left, gray_left, cv::COLOR_BGR2GRAY);
        // cv::cvtColor(rectified_right, gray_right, cv::COLOR_BGR2GRAY);

        gray_left = cv::imread("../calibration/cone/dispL.png", cv::IMREAD_GRAYSCALE);
        gray_right = cv::imread("../calibration/cone/dispR.png", cv::IMREAD_GRAYSCALE);

        stereoSGBM(gray_left,gray_right,disparity);
        
        // 点云数组
        vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud_dense;
        // 便利像素点，利用disparity和相机内参计算点深度信息，将点存入数组中
        for (int v = 0; v < rectified_left.rows; v++)
            for (int u = 0; u < rectified_left.cols; u++)
            {
                if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0)
                    continue;
                Vector4d point(0, 0, 0, rectified_left.at<uchar>(v, u) / 255.0); // 前三维为xyz,第四维为颜色
                // 根据双目模型计算 point 的位置
                double x = (u - cx) / fx;
                double y = (v - cy) / fy;
                double depth = fx * baseline / (disparity.at<float>(v, u));
                point[0] = x * depth;
                point[1] = y * depth;
                point[2] = depth;
                pointcloud_dense.push_back(point);
            }

        // 画出点云
        showPointCloud(pointcloud_dense);

        if (waitKey(15) >= 0)
        {
            break;
        }
    }

    return 0;
}
