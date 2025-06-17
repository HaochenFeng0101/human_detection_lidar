cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(my_segmentation_and_clustering CXX) # Changed project name for clarity

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0") 
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  rosbag
  sensor_msgs
  pcl_ros
)

find_package(catkin REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${catkin_INCLUDE_DIRS}) 

# Find spdlog
find_package(spdlog REQUIRED)


# Find PCL
find_package(PCL 1.8 REQUIRED COMPONENTS common io filters features segmentation visualization kdtree)
if (PCL_FOUND)
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
endif()


add_library(my_segmentation_clustering_lib STATIC 
    src/groundseg.cpp
    src/StaticClusterer.cpp
)


# Libraries should also be linked to their dependencies
target_link_libraries(my_segmentation_clustering_lib
    ${PCL_LIBRARIES} # Link all PCL components found
    spdlog::spdlog    # Link spdlog
)


# add_executable(cluster src/test_cluster.cpp)
# target_link_libraries(cluster
#     my_segmentation_clustering_lib 
#     ${PCL_LIBRARIES}               
#     spdlog::spdlog                 
# )

add_executable(test_tracking src/test_tracking.cpp)
add_library(comparing_clouds_lib STATIC 
    src/comparing_clouds.cpp
)
target_link_libraries(test_tracking
    ${catkin_LIBRARIES} 
    my_segmentation_clustering_lib 
    comparing_clouds_lib
    ${PCL_LIBRARIES}               
    spdlog::spdlog                 
)
