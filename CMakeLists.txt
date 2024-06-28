set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_minimum_required(VERSION 3.5.1)

# 项目名称
project(StereoCamera)

# Add libraries.
find_package(OpenCV 3.4.16 REQUIRED)

if (NOT OpenCV_LIBS)
  set(OpenCV_LIBS "opencv::opencv")
endif()

# 添加可执行文件
add_executable(realtime_depthmap realtime_depthmap.cpp)

# 链接 OpenCV 库
target_link_libraries(realtime_depthmap ${OpenCV_LIBS})
