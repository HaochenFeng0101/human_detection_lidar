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

    if (view.size() == 0)
    {
        std::cerr << "ERROR: rosbag view is empty! No messages found for topics: ";
        for (const auto &t : topics)
        {
            std::cerr << t << " ";
        }
        std::cerr << std::endl;
        // Print all topics in the bag to help identify issues
        rosbag::View all_topics_view(bag);
        std::cerr << "Available topics in bag: " << std::endl;
        for (const rosbag::ConnectionInfo *info : all_topics_view.getConnections())
        {
            std::cerr << "  - " << info->topic << " (" << info->datatype << ")" << std::endl;
        }

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
    StaticClusterer clusterer(0.2, 30, 100, 0.15); // 0.2 30 100 0.15
    // Adjust parameters for tracking/fall detection: max_hist_time, assoc_dist, fall_height_change, fall_duration, static_dist, min_static_frames
    comparing_clouds object_tracker(
        10.0, // max_hist_time (seconds) - keep history for 10 seconds
        0.5,  // assoc_dist (meters) - max distance for object association between frames
        0.45, // min_fall_height_change (meters) - consider fall if min_z drops by 0.45m
        15.5, // min_fall_duration (seconds) - fall must happen within 15.5s (NOTE: this is quite long for a fall. A typical fall is often <1s.)
        0.3,  // static_dist (meters) - max displacement for "static" object
        3  ,   // min_static_frames - how many frames to confirm static
        10.0 //time to move if not updated
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

            // Convert ROS PointCloud2 to PCL PointCloud<PointXYZI> PointType
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

            // 3. Object Tracking and Status Determination (now handled internally by comparing_clouds)
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
            // viewer->removePointCloud("ground_cloud");

            // Visualize Ground (White)
            if (!ground_cloud->empty())
            {
                pcl::visualization::PointCloudColorHandlerCustom<PointType> ground_color(ground_cloud, 255, 255, 255); // White
                viewer->addPointCloud<PointType>(ground_cloud, ground_color, "ground_cloud");
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "ground_cloud");
            }

            // Visualize Tracked Objects (colors based on status)
            // The flags (is_falling, is_static) are now set within each TrackedObject by process_non_grounded_points
            const auto &all_tracked_objects = object_tracker.getAllTrackedObjects();

            //two for loop might be inefficient
            for (const auto &tracked_obj : all_tracked_objects)
            {
                // Always use the latest cloud data for visualization
                for (const auto &current_clustered : clustered_objects)
                {
                    if (object_tracker.calculateCentroid(current_clustered) == tracked_obj.current_centroid)
                    {
                        // pcl::PointCloud<PointType>::Ptr object_cloud = tracked_obj.latest_cloud;
                        // tracked_obj.history.end
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
                        { // Moved or new (i.e., not falling and not static)
                            r = 255;
                            g = 0;
                            b = 0; // Red for moved/dynamic
                            std::cout << "  - Object ID " << tracked_obj.id << " is MOVED/DYNAMIC." << std::endl;
                        }

                        // print the tracked object details
                        spdlog::info("Tracked Object ID: {}, Type: {}, Centroid: ({:.2f}, {:.2f}, {:.2f}), Extents: ({:.2f}x{:.2f}x{:.2f}), Min Z: {:.2f}, Max Z: {:.2f}",
                                     tracked_obj.id,
                                     (tracked_obj.classified_type == ObjectType::HUMAN) ? "HUMAN" : "UNKNOWN",
                                     tracked_obj.current_centroid.x(),
                                     tracked_obj.current_centroid.y(),
                                     tracked_obj.current_centroid.z(),
                                     tracked_obj.current_extent_x,
                                     tracked_obj.current_extent_y,
                                     tracked_obj.current_extent_z,
                                     tracked_obj.current_min_z,
                                     tracked_obj.current_max_z);

                        pcl::visualization::PointCloudColorHandlerCustom<PointType> obj_color(current_clustered, r, g, b);
                        viewer->addPointCloud<PointType>(current_clustered, obj_color, cloud_id_str);
                        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, cloud_id_str);
                    }
                }
            }

            
            object_tracker.removeStaleTrackedObjects(current_timestamp);
            // removeStaleTrackedObjects
            viewer->spinOnce(100);                                      // Process PCL viewer events
            std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Small delay for visualization
        }
    }

    bag.close(); // Close the bag file

    std::cout << "Finished processing bag file. Viewer will remain open." << std::endl;
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    ros::shutdown(); // Shutdown ROS
    return 0;
}