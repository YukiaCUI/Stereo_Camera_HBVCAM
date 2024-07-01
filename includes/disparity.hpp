#ifndef DISPARITY_H
#define DISPARITY_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>

using namespace std;
using namespace cv;

void stereoBM(cv::Mat lpng,cv::Mat rpng,cv::Mat &disp)
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
    cv::Rect roi1,roi2;
    cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16,9);
    cv::Mat disp_color;

    int nmDisparities = ((imgSize.width / 8) + 15) & -16;//视差搜索范围

    bm->setPreFilterType(0);//预处理滤波器类型
    bm->setPreFilterSize(15);//预处理滤波器窗口大小
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

    cv::applyColorMap(disp1,disp_color,cv::COLORMAP_JET);//转彩色图

    cv::imshow("BM_img",disp1);
    cv::imshow("BM_color",disp_color);
}

void stereoSGBM(cv::Mat lpng,cv::Mat rpng,cv::Mat &disp_8u)
{
    cv::GaussianBlur(lpng, lpng, cv::Size(5, 5), 1.5);
    cv::GaussianBlur(rpng, rpng, cv::Size(5, 5), 1.5);
    cv::pyrDown(lpng, lpng);
    cv::pyrDown(rpng, rpng);
    cv::GaussianBlur(lpng, lpng, cv::Size(5, 5), 1.5);
    cv::GaussianBlur(rpng, rpng, cv::Size(5, 5), 1.5);

    cv::Mat disp = cv::Mat(lpng.rows,lpng.cols,CV_16S);
    disp_8u.create(lpng.rows,lpng.cols,CV_8UC1);
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

    disp.convertTo(disp_8u,CV_8U,255 / (nmDisparities*16.));//转8位
    cv::GaussianBlur(disp_8u, disp_8u, cv::Size(5, 5), 1.5);

    cv::applyColorMap(disp_8u,disp_color,cv::COLORMAP_JET);//转彩色图

    cv::imshow("SGBM_img",disp_8u);
    cv::imshow("SGBM_color",disp_color);
}

#endif
