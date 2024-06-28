cmake_minimum_required(VERSION 3.5.1)

# 项目名称
project(StereoCamera)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找库
find_package(OpenCV 3.4.16 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

# 包含头文件路径
include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${Pangolin_INCLUDE_DIRS})

# 设置OpenCV库路径
if (NOT OpenCV_LIBS)
  set(OpenCV_LIBS "opencv::opencv")
endif()

# 深度图
# add_executable(realtime_depthmap realtime_depthmap.cpp)
# target_link_libraries(realtime_depthmap ${OpenCV_LIBS})

# 稀疏点云图
# add_executable(3d_point_cloud 3d_point_cloud_sparse.cpp)
# target_link_libraries(3d_point_cloud ${OpenCV_LIBS} ${Pangolin_LIBRARIES} Eigen3::Eigen)

# 稠密点云图
add_executable(3d_point_cloud 3d_point_cloud_dense.cpp)
target_link_libraries(3d_point_cloud ${OpenCV_LIBS} ${Pangolin_LIBRARIES} Eigen3::Eigen)