#include "comparing_clouds.hpp"
#include "StaticClusterer.hpp"
#include "groundseg.hpp"

#include <iostream>
#include <algorithm>
#include <thread>
#include <chrono>
#include <spdlog/spdlog.h> // For logging
// PCL includes for point cloud processing and visualization
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h> // PCL Viewer
#include <pcl_conversions/pcl_conversions.h>  // For ROS to PCL conversion

// ROS includes for bag file reading
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h> // ROS PointCloud2 message type

// Typedef for clarity
typedef pcl::PointXYZI PointType;

// --- Main Function ---
int main(int argc, char *argv[])
{
    // Initialize ROS (even if not publishing, needed for rosbag)
    ros::init(argc, argv, "lidar_tracker_node");

    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <path_to_rosbag_file.bag>" << std::endl;
        return 1;
    }

    std::string bag_file_path = argv[1];

    // Open the ROS bag file
    rosbag::Bag bag;
    try
    {
        bag.open(bag_file_path, rosbag::bagmode::Read);
    }
    catch (const rosbag::BagException &e)
    {
        std::cerr << "Error opening bag file: " << e.what() << std::endl;
        return 1;
    }

    std::vector<std::string> topics;
    topics.push_back("/lidar_points");

    rosbag::View view(bag, rosbag::TopicQuery(topics));

    if (view.size()==0)
    {
        std::cerr << "ERROR: rosbag view is empty! No messages found for topics: ";
        for (const auto &t : topics)
        {
            std::cerr << t << " ";
        }
        std::cerr << std::endl;
        // Optionally, print all topics in the bag to help identify issues
        rosbag::View all_topics_view(bag);
        std::cerr << "Available topics in bag: " << std::endl;
      
        
        bag.close();
        ros::shutdown();
        return 1; // Exit early if no messages
    }
    else
    {
        std::cout << "Bag view contains " << view.size() << " messages on specified topics." << std::endl;
    }

    // --- Initialize Ground Segmenter, Static Clusterer, and Object Tracker ---
    LidarGroundSegmenter ground_segmenter;
    // Adjust parameters if needed: cluster_tolerance, min_cluster_size, max_cluster_size, voxel_leaf_size
    StaticClusterer clusterer(0.2, 30, 100, 0.15);
    // Adjust parameters for tracking/fall detection: max_hist_time, assoc_dist, fall_height_change, fall_duration, static_dist, min_static_frames
    comparing_clouds object_tracker(
        5.0, // max_hist_time (seconds) - keep history for 2 seconds
        0.5, // assoc_dist (meters) - max distance for object association between frames
        0.5, // min_fall_height_change (meters) - consider fall if min_z drops by M
        1.5, // min_fall_duration (seconds) - fall must happen within 0.5s
        0.3, // static_dist (meters) - max displacement for "static" object
        3    // min_static_frames - how many frames to confirm static
    );

    // --- PCL Viewer Setup ---
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("LiDAR Tracking Viewer"));
    viewer->setBackgroundColor(0, 0, 0); // Black background
    viewer->addCoordinateSystem(1.0);    // Add XYZ axis
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 10, 0, 1, 0, 0); // Initial camera position (x, y, z, vx, vy, vz, up_x, up_y, up_z)

    // --- Process ROS Bag Messages ---
    int frame_count = 0;
    for (rosbag::MessageInstance const m : view)
    {
        // Check if the viewer is still open
        if (viewer->wasStopped())
        {
            break; // Exit if viewer is closed
        }

        sensor_msgs::PointCloud2::ConstPtr cloud_msg = m.instantiate<sensor_msgs::PointCloud2>();
        if (cloud_msg != nullptr)
        {

            spdlog::info("cloud_msg size: {}, frame_count: {}", cloud_msg->data.size(), frame_count);
            frame_count++;
            std::cout << "\n--- Processing Frame " << frame_count << " ---" << std::endl;

            // Convert ROS PointCloud2 to PCL PointCloud<PointXYZI>
            pcl::PointCloud<PointType>::Ptr input_cloud_raw(new pcl::PointCloud<PointType>);
            pcl::fromROSMsg(*cloud_msg, *input_cloud_raw);

            double current_timestamp = cloud_msg->header.stamp.toSec();

            // 1. Ground Segmentation
            pcl::PointCloud<PointType>::Ptr ground_cloud(new pcl::PointCloud<PointType>);
            pcl::PointCloud<PointType>::Ptr non_ground_cloud(new pcl::PointCloud<PointType>);
            pcl::PointCloud<PointType>::Ptr leveled_input_cloud_for_viz(new pcl::PointCloud<PointType>);

            ground_segmenter.segmentGround(input_cloud_raw, ground_cloud, non_ground_cloud, leveled_input_cloud_for_viz);

            // 2. Object Clustering on Non-Ground Points
            std::vector<pcl::PointCloud<PointType>::Ptr> clustered_objects;
            clusterer.cluster(non_ground_cloud, clustered_objects);

            // 3. Object Tracking and Status Determination
            std::vector<int> falling_ids, static_ids, moved_ids;
            object_tracker.process_non_grounded_points(
                clustered_objects,
                current_timestamp,
                falling_ids,
                static_ids,
                moved_ids);

            // --- Visualization ---
            viewer->removeAllPointClouds(); // Clear previous clouds
            viewer->removeAllShapes();      // Clear previous shapes (if any)

            // Visualize Ground (White)
            if (!ground_cloud->empty())
            {
                pcl::visualization::PointCloudColorHandlerCustom<PointType> ground_color(ground_cloud, 255, 255, 255); // White
                viewer->addPointCloud<PointType>(ground_cloud, ground_color, "ground_cloud");
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "ground_cloud");
            }

            // Visualize Tracked Objects (colors based on status)
            const auto &all_tracked_objects = object_tracker.getAllTrackedObjects();
            for (const auto &tracked_obj : all_tracked_objects)
            {
                if (tracked_obj.history.empty())
                    continue; // Skip if no points

                pcl::PointCloud<PointType>::Ptr object_cloud = tracked_obj.history.back().cloud;
                std::string cloud_id_str = "object_" + std::to_string(tracked_obj.id);

                unsigned char r = 0, g = 0, b = 0; // Default to black
                if (tracked_obj.is_falling)
                {
                    r = 255;
                    g = 0;
                    b = 255; // Magenta for falling
                    std::cout << "  - Object ID " << tracked_obj.id << " is FALLING!" << std::endl;
                }
                else if (tracked_obj.is_static)
                {
                    r = 0;
                    g = 255;
                    b = 0; // Green for static
                    std::cout << "  - Object ID " << tracked_obj.id << " is STATIC." << std::endl;
                }
                else
                { // Moved or new
                    r = 255;
                    g = 0;
                    b = 0; // Red for moved/dynamic
                    std::cout << "  - Object ID " << tracked_obj.id << " is MOVED/DYNAMIC." << std::endl;
                }

                // // Add bounding box for tracked objects (optional but helpful)

                // pcl::PointCloud<PointXYZI>::ConstPtr min_pt, max_pt;
                // pcl::getMinMax3D(*object_cloud, min_pt, max_pt);
                // viewer->addCube(min_pt.x, max_pt.x, min_pt.y, max_pt.y, min_pt.z, max_pt.z, r/255.0, g/255.0, b/255.0, "bbox_" + cloud_id_str);
                // viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "bbox_" + cloud_id_str);

                pcl::visualization::PointCloudColorHandlerCustom<PointType> obj_color(object_cloud, r, g, b);
                viewer->addPointCloud<PointType>(object_cloud, obj_color, cloud_id_str);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloud_id_str);
            }

            viewer->spinOnce(100);                                      // Process PCL viewer events
            std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Small delay for visualization
        }
    }

    bag.close(); // Close the bag file

    std::cout << "Finished processing bag file. Viewer will remain open." << std::endl;
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    ros::shutdown(); // Shutdown ROS
    return 0;
}