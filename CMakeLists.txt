cmake_minimum_required(VERSION 2.8.3)
project(stereo_reconstruction)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

#SET(OpenCV_DIR "/home/auv/software/opencv/build")
find_package(OpenCV REQUIRED)


set(INC_DIR ./include)
set(LIBELAS_SRC_DIR src/elas)
set(CMAKE_CXX_FLAGS "-msse3")
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.54.0 COMPONENTS system filesystem program_options REQUIRED)
find_package(PythonLibs 2.7)


#catkin
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  image_transport
  sensor_msgs
  dynamic_reconfigure
  geometry_msgs
)
# Dynamic reconfigure support
generate_dynamic_reconfigure_options(cfg/CamToRobotCalibParams.cfg)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs
  INCLUDE_DIRS include
  DEPENDS OpenCV
)


FILE(GLOB LIBELAS_SRC_FILES "src/elas/*.cpp")
set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} /usr/local/lib/python2.7/dist-packages/numpy/core/include)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${LIBELAS_SRC_DIR}
  ${Boost_INCLUDE_DIRS}
  ${INC_DIR} ${catkin_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS}
)
set (LIBS ${PYTHON_LIBRARIES} /usr/lib/x86_64-linux-gnu/libpython2.7.so)

#add_executable(dense_reconstruction src/dense_reconstruction.cpp src/utils.cpp ${LIBELAS_SRC_FILES})
#target_link_libraries(dense_reconstruction ${OpenCV_LIBS}  ${catkin_LIBRARIES}  ${Boost_GENERAL} "-lpopt")

add_executable(stereo_app src/stereo_app.cpp src/utils.cpp src/DepthReconstructor.cpp ${LIBELAS_SRC_FILES})
target_link_libraries(stereo_app ${OpenCV_LIBS}  ${catkin_LIBRARIES}  ${Boost_GENERAL} glog gflags)