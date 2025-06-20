#include "StaticClusterer.hpp" // Assuming this header exists and is correct
#include "groundseg.hpp"       // Assuming this header exists and is correct

// New include for EKFObjectTracker
#include "EKFObjectTracker.hpp"

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
#include <spdlog/spdlog.h> // For logging

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

    // --- Initialize Ground Segmenter, Static Clusterer, and EKF Object Tracker ---
    LidarGroundSegmenter ground_segmenter;
    // Adjust parameters if needed: cluster_tolerance, min_cluster_size, max_cluster_size, voxel_leaf_size
    StaticClusterer clusterer(0.2, 30, 100, 0.15); 

    // Initialize EKFObjectTracker with its parameters
    // These parameters will significantly affect tracking performance and fall/static detection.
    // Tuning is often required based on your sensor, environment, and desired behavior.
    EKFObjectTracker ekf_tracker(
        5.0,  // assoc_dist (Mahalanobis distance threshold) - tune carefully. Higher values allow more flexible associations.
              // Note: This is *not* a Euclidean distance, it's a statistical distance considering uncertainties.
              // A value of 5.0-15.0 is common for Mahalanobis^2
        5,    // max_missed_detections (frames) - Max consecutive frames an object can be unmatched before being deleted
        2.0,  // max_history_time (seconds) - How long to keep state and measurement history for each track
        0.5,  // min_fall_z_vel (m/s) - Minimum downward Z-velocity to be considered falling
        0.45, // min_fall_height_change (m) - Minimum height drop to detect a fall over 'fall_check_duration_sec'
        0.5,  // fall_check_duration_sec (s) - Time window for checking height drop for fall detection
        0.3,  // static_threshold_dist (m) - Maximum displacement (Euclidean) for an object to be considered static
        5     // min_static_frames - Minimum number of historical frames needed to confirm static status
    );

    // --- PCL Viewer Setup ---
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("LiDAR EKF Tracking Viewer"));
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
            std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> clustered_objects;
            clusterer.cluster(non_ground_cloud, clustered_objects);

            // 3. EKF Object Tracking: Process clusters with the EKF tracker
            ekf_tracker.processClusters(clustered_objects, current_timestamp);

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

            // Visualize Tracked Objects from EKF
            const auto &all_tracked_objects_ekf = ekf_tracker.getTrackedObjects();
            for (const auto &tracked_obj_ekf : all_tracked_objects_ekf)
            {
                // Optionally, only visualize objects classified as HUMAN
                if (tracked_obj_ekf.classified_type != ObjectType::HUMAN) {
                    continue; // Skip non-human objects for visualization
                }

                // Create a point cloud for visualization using the latest associated measurement
                // If no measurement history (e.g., just predicted or just initialized track),
                // create a single point at the EKF estimated position.
                pcl::PointCloud<PointType>::Ptr object_cloud_for_viz(new pcl::PointCloud<PointType>);
                if (!tracked_obj_ekf.measurement_history.empty()) {
                    object_cloud_for_viz = tracked_obj_ekf.measurement_history.back().cloud;
                } else {
                    // Fallback: Visualize EKF's current estimated centroid as a single point
                    PointType p_est;
                    p_est.x = tracked_obj_ekf.x(0);
                    p_est.y = tracked_obj_ekf.x(1);
                    p_est.z = tracked_obj_ekf.x(2);
                    p_est.intensity = 1.0; // Arbitrary intensity
                    object_cloud_for_viz->push_back(p_est);
                    object_cloud_for_viz->width = 1;
                    object_cloud_for_viz->height = 1;
                }

                std::string cloud_id_str = "ekf_object_" + std::to_string(tracked_obj_ekf.id);
                std::string text_id_str = "ekf_text_" + std::to_string(tracked_obj_ekf.id);

                unsigned char r = 255, g = 255, b = 255; // Default color
                std::string status_msg = "MOVING"; // Default status

                if (tracked_obj_ekf.is_falling)
                {
                    r = 255; g = 0; b = 255; // Magenta for falling
                    status_msg = "FALLING!";
                }
                else if (tracked_obj_ekf.is_static)
                {
                    r = 0; g = 255; b = 0; // Green for static
                    status_msg = "STATIC";
                }
                else // If not falling and not static, assume moving (red)
                {
                    r = 255; g = 0; b = 0; // Red for moving
                }

                // Add point cloud to viewer
                pcl::visualization::PointCloudColorHandlerCustom<PointType> obj_color(object_cloud_for_viz, r, g, b);
                viewer->addPointCloud<PointType>(object_cloud_for_viz, obj_color, cloud_id_str);
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, cloud_id_str); // Make tracked points larger

                // Add ID and status text at the EKF estimated centroid
                viewer->addText(
                    "ID: " + std::to_string(tracked_obj_ekf.id) + " (" + status_msg + ")",
                    tracked_obj_ekf.x(0), tracked_obj_ekf.x(1), tracked_obj_ekf.x(2) + 0.5, // Text slightly above CoM
                    1.0, 1.0, 1.0, // White color for text
                    text_id_str
                );

                // Optional: Visualize EKF covariance ellipsoid (for debugging filter behavior)
                // viewer->addEllipsoid(tracked_obj_ekf.P.block<3,3>(0,0), tracked_obj_ekf.x.head<3>().cast<float>(), 1.0, 0.0, 0.0, "ell_" + std::to_string(tracked_obj_ekf.id));
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