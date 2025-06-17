#include "groundseg.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>    
#include <pcl/features/normal_3d.h> // Though not directly used for RANSAC, good to keep if needed later
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <spdlog/spdlog.h>

#include <pcl/common/transforms.h>
#include <Eigen/Geometry>
#include <algorithm> 
#include <pcl/io/pcd_io.h> 

// Constructor implementation
LidarGroundSegmenter::LidarGroundSegmenter() {
    // Initialize parameters
    m_voxel_leaf_size = 0.1; // meters
    m_normal_search_radius = 0.3; // meters, for initial normal estimation if used
    m_max_iterations = 300;
    m_distance_threshold = 0.15; // meters, for RANSAC plane fitting
    m_min_plane_points = 200; // Minimum points for a valid plane
    m_max_ground_angle_rad = M_PI / 10; // 18 degrees, tolerance for a plane to be considered "horizontal ground"

    // Initialize the ground plane cloud member
    m_ground_plane_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>); 
    m_non_ground_plane_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
}

// Private helper to find a dominant plane and return its transform and coefficients
// This is now more general for internal use by segmentGround
bool LidarGroundSegmenter::alignCloudWithGround(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                                              pcl::PointCloud<pcl::PointXYZI>::Ptr& output_leveled_cloud,
                                              Eigen::Affine3f& transform_out, // Output: The calculated transform
                                              pcl::ModelCoefficients::Ptr& plane_coefficients_out) { // Output: The plane coeffs
    
    output_leveled_cloud->clear(); // Ensure output cloud is clean
    transform_out = Eigen::Affine3f::Identity(); // Reset transform

    if (input_cloud->empty()) {
        spdlog::warn("Input cloud for ground alignment is empty.");
        return false;
    }

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    
    pcl::SACSegmentation<pcl::PointXYZI> seg_align;
    seg_align.setOptimizeCoefficients(true);
    seg_align.setModelType(pcl::SACMODEL_PLANE);
    seg_align.setMethodType(pcl::SAC_RANSAC);
    seg_align.setMaxIterations(m_max_iterations); // Use m_max_iterations for consistency
    seg_align.setDistanceThreshold(m_distance_threshold);
    seg_align.setInputCloud(input_cloud);
    seg_align.segment(*inliers, *plane_coefficients_out); // Segment and get coefficients

    if (inliers->indices.empty() || inliers->indices.size() < m_min_plane_points / 2) {
        spdlog::warn("Failed to detect a dominant plane for ground alignment or plane is too small. Inliers: {}.", inliers->indices.size());
        *output_leveled_cloud = *input_cloud; // No transform applied
        return false;
    }

    Eigen::Vector3f plane_normal(plane_coefficients_out->values[0], 
                                 plane_coefficients_out->values[1], 
                                 plane_coefficients_out->values[2]);
    plane_normal.normalize();

    Eigen::Vector3f target_up_vector(0.0, 0.0, 1.0);
    
    // Ensure normal points upwards (or towards the target_up_vector direction)
    if (plane_normal.dot(target_up_vector) < 0) {
        plane_normal *= -1.0;
        plane_coefficients_out->values[0] *= -1.0f;
        plane_coefficients_out->values[1] *= -1.0f;
        plane_coefficients_out->values[2] *= -1.0f;
        plane_coefficients_out->values[3] *= -1.0f;
    }

    Eigen::Vector3f rotation_axis = plane_normal.cross(target_up_vector);
    float dot_product = plane_normal.dot(target_up_vector);
    
    if (rotation_axis.norm() < 1e-6) { // Plane is already aligned or anti-aligned with target_up_vector
        if (dot_product > 0.999) { // Already aligned
             spdlog::info("Cloud already aligned with ground.");
             *output_leveled_cloud = *input_cloud;
             return true;
        } else { // Anti-aligned (normal points down)
            // Should already be handled by the plane_normal.dot(target_up_vector) < 0 check
            // If still here, this implies a 180 deg rotation around X or Y axis is needed
            rotation_axis = Eigen::Vector3f::UnitX(); // Fallback axis for 180 deg flip
            dot_product = -1.0f;
        }
    }
    rotation_axis.normalize();

    float rotation_angle = std::acos(std::clamp(dot_product, -1.0f, 1.0f));

    transform_out.rotate(Eigen::AngleAxisf(rotation_angle, rotation_axis));

    pcl::transformPointCloud(*input_cloud, *output_leveled_cloud, transform_out);
    spdlog::info("Aligned cloud with ground. Normal: ({:.2f}, {:.2f}, {:.2f}), Rotation angle: {:.2f} rad around axis ({:.2f}, {:.2f}, {:.2f})",
              plane_normal.x(), plane_normal.y(), plane_normal.z(),
              rotation_angle, rotation_axis.x(), rotation_axis.y(), rotation_axis.z());
    
              
    return true;
}

// segmentGround function implementation
void LidarGroundSegmenter::segmentGround(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud_raw,
                                       pcl::PointCloud<pcl::PointXYZI>::Ptr& ground_cloud,
                                       pcl::PointCloud<pcl::PointXYZI>::Ptr& non_ground_cloud,
                                       pcl::PointCloud<pcl::PointXYZI>::Ptr& output_leveled_input_cloud) {
    // 0. Reset output clouds
    ground_cloud->clear();
    non_ground_cloud->clear();
    output_leveled_input_cloud->clear();

    if (input_cloud_raw->empty()) {
        std::cout << "Input cloud is empty, nothing to segment." << std::endl;
        return;
    }

    std::cout << "Starting ground segmentation for " << input_cloud_raw->points.size() << " points." << std::endl;

    // 1. Downsample the input point cloud (to make plane finding faster)
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    vg.setInputCloud(input_cloud_raw);
    vg.setLeafSize(m_voxel_leaf_size, m_voxel_leaf_size, m_voxel_leaf_size);
    vg.filter(*cloud_downsampled);
    std::cout << "Downsampled to " << cloud_downsampled->points.size() << " points." << std::endl;

    if (cloud_downsampled->empty()) {
        spdlog::error("Downsampled cloud is empty, cannot proceed with segmentation.");
        return;
    }

    // 2. Find the dominant plane in the downsampled cloud and get the transform to level it
    pcl::ModelCoefficients::Ptr dominant_plane_coefficients(new pcl::ModelCoefficients);
    Eigen::Affine3f leveling_transform = Eigen::Affine3f::Identity();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_leveled_from_dominant_plane(new pcl::PointCloud<pcl::PointXYZI>);

    bool plane_found_and_transformed = alignCloudWithGround(
        cloud_downsampled, // Use downsampled cloud to find plane efficiently
        cloud_leveled_from_dominant_plane, // The leveled version of the downsampled cloud
        leveling_transform, // Get the transform used
        dominant_plane_coefficients // Get the coefficients of the detected plane
    );

    if (!plane_found_and_transformed) {
        spdlog::warn("No suitable dominant plane found in downsampled cloud for segmentation. All points considered non-ground.");
        *non_ground_cloud = *input_cloud_raw; // Original input becomes non-ground
        *output_leveled_input_cloud = *input_cloud_raw; // No leveling applied
        return;
    }

    // Apply the *same* leveling transform to the *original raw input cloud*
    pcl::transformPointCloud(*input_cloud_raw, *output_leveled_input_cloud, leveling_transform);
    std::cout << "Original raw cloud aligned using the dominant plane transform." << std::endl;
    
    // Verify the normal of the dominant plane in the *leveled* coordinate system
    Eigen::Vector3f plane_normal_leveled(dominant_plane_coefficients->values[0],
                                         dominant_plane_coefficients->values[1],
                                         dominant_plane_coefficients->values[2]);
    plane_normal_leveled.normalize();
    Eigen::Vector3f up_vector(0.0, 0.0, 1.0); // Z is up in the *leveled* coordinate system

    // Ensure normal points upwards (after leveling)
    if (plane_normal_leveled.dot(up_vector) < 0) {
        plane_normal_leveled *= -1.0;
    }

    float normal_angle_with_z_rad = std::acos(std::clamp(plane_normal_leveled.dot(up_vector), -1.0f, 1.0f));
    float normal_angle_with_z_deg = normal_angle_with_z_rad * 180.0 / M_PI;
   
    std::cout << "Dominant plane normal in leveled space: (" << plane_normal_leveled.x() << ", " << plane_normal_leveled.y() << ", " << plane_normal_leveled.z() << "), Angle with Z: " << normal_angle_with_z_deg << " degrees." << std::endl;

    if (normal_angle_with_z_rad < m_max_ground_angle_rad) {
        // Use the original coefficients (which now represent a horizontal plane in the leveled frame)
        // to extract inliers from the *leveled original input cloud*.
        pcl::SACSegmentation<pcl::PointXYZI> seg_extract;
        seg_extract.setOptimizeCoefficients(true);
        seg_extract.setModelType(pcl::SACMODEL_PLANE);
        seg_extract.setMethodType(pcl::SAC_RANSAC);
        seg_extract.setMaxIterations(m_max_iterations);
        seg_extract.setDistanceThreshold(m_distance_threshold);
        seg_extract.setInputCloud(output_leveled_input_cloud); // Use the full leveled cloud
        
        pcl::PointIndices::Ptr ground_inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr ground_coefficients_final(new pcl::ModelCoefficients);

        // Re-segment on the *leveled_input_cloud* to get *its* ground plane.
        // This second RANSAC is needed because `alignCloudWithGround` used a downsampled cloud,
       
        
        seg_extract.segment(*ground_inliers, *ground_coefficients_final);

        if (ground_inliers->indices.empty() || ground_inliers->indices.size() < m_min_plane_points) {
            spdlog::warn("RANSAC on leveled full cloud found no ground or too small. All points considered non-ground.");
            *non_ground_cloud = *output_leveled_input_cloud;
            return;
        }

        pcl::ExtractIndices<pcl::PointXYZI> extract;
        extract.setInputCloud(output_leveled_input_cloud);
        extract.setIndices(ground_inliers);
        
        extract.setNegative(false); // Extract ground
        extract.filter(*ground_cloud);
        std::cout << "Identified ground plane with " << ground_cloud->points.size() << " points." << std::endl;

        m_ground_plane_cloud = ground_cloud; // Store the ground plane cloud

        pcl::io::savePCDFileASCII("aligned_ground.pcd", *m_ground_plane_cloud);
        std::cout << "Saved aligned ground cloud to aligned_ground.pcd" << std::endl;

        extract.setNegative(true); // Extract non-ground
        extract.filter(*non_ground_cloud);
        m_non_ground_plane_cloud = non_ground_cloud; 
        pcl::io::savePCDFileASCII("non_ground.pcd", *m_non_ground_plane_cloud);
      
    } else {
        spdlog::warn("Dominant plane found ({:.2f} deg from horizontal) but too steep to be considered ground in leveled frame. All points considered non-ground.", normal_angle_with_z_deg);
        *non_ground_cloud = *output_leveled_input_cloud; // All points are non-ground if the dominant plane is too steep
    }

    std::cout << "Ground Segmentation Complete." << std::endl;
    std::cout << "Final Ground points: " << ground_cloud->points.size() << std::endl;
    std::cout << "Final Non-ground points: " << non_ground_cloud->points.size() << std::endl;
}