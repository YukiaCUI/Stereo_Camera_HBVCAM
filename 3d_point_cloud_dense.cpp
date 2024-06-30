#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <unistd.h>


#define Img_width 2560
#define Img_height 720

using namespace std;
using namespace cv;
using namespace Eigen;



void showPointCloud(const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    pangolin::DestroyWindow("Point Cloud Viewer");

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
    VideoCapture cap(0);
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


        double fx = cameraMatrixL.at<double>(0, 0);
        double fy = cameraMatrixL.at<double>(1, 1);
        double cx = cameraMatrixL.at<double>(0, 2);
        double cy = cameraMatrixL.at<double>(1, 2);
        double b  = norm(T, cv::NORM_L2)/1000.0;


        /*************************稠密点云*********************/
        cv::Mat gray_left, gray_right, disparity, disparity_float;
        cv::cvtColor(rectified_left, gray_left, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rectified_right, gray_right, cv::COLOR_BGR2GRAY);

        cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    
        cv::Mat disparity_sgbm;
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

        cv::imshow("disparity", disparity / 96.0);
        cv::waitKey(0);
        // 画出点云
        showPointCloud(pointcloud_dense);

        if (waitKey(15) >= 0)
        {
            break;
        }
    }

    return 0;
}
