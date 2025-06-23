cmake_minimum_required(VERSION 3.5 )
include_guard(GLOBAL)
project(my_segmentation_and_clustering CXX) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0") 
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")

find_package(ROS REQUIRED roscpp roscpp_serialization sensor_msgs message_filters rosbag tf2 cv_bridge)
find_package(PCL REQUIRED QUIET)
find_package(Eigen3 REQUIRED)
# find_package(OpenCV CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
# find_package(Boost 1.71.0 COMPONENTS filesystem system REQUIRED)

# find_package(catkin REQUIRED COMPONENTS
#     roscpp
#     roscpp_serialization
#     sensor_msgs
#     message_filters
#     rosbag
#     tf2
#     cv_bridge
# )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    # ${CMAKE_CURRENT_SOURCE_DIR}/src
                    # ${cv_bridge_INCLUDE_DIRS}
                    # ${catkin_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIRS}
                    ${PCL_INCLUDE_DIRS}
                    # ${OpenCV_INCLUDE_DIRS}
                    ) 

add_library(seg_cluster_lib  
    src/groundseg.cpp
    src/StaticClusterer.cpp
    src/comparing_clouds.cpp
)
target_link_libraries(seg_cluster_lib
     spdlog::spdlog
)


# add_executable(cluster src/test_cluster.cpp)
# target_link_libraries(cluster
#     my_segmentation_clustering_lib 
#     ${PCL_LIBRARIES}               
#     spdlog::spdlog                 
# )

add_executable(test_tracking src/test_tracking.cpp)


target_link_libraries(test_tracking
    # ${catkin_LIBRARIES} 
    seg_cluster_lib 
    ${PCL_LIBRARIES}               
    spdlog::spdlog     
    ros::roscpp
    ros::roscpp_serialization
    ros::message_filters
    ros::rosbag
#   ros::tf2      
)



# add_library(ekf STATIC 
#     src/EKFObjectTracker.cpp
# )
# add_executable(ekf_tracking_test src/ekf_tracking_test.cpp)
# target_link_libraries(ekf_tracking_test
#     ${catkin_LIBRARIES}
#     my_segmentation_clustering_lib
#     spdlog::spdlog
#     ${PCL_LIBRARIES}
#     ekf
# )