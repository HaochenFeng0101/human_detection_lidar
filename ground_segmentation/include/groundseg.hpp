#ifndef GROUNDSEG_HPP
#define GROUNDSEG_HPP

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <vector>
#include <memory> 
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
// PCL visualization is commented out in your .cpp, keep it here just in case
// #include <pcl/visualization/pcl_visualizer.h>

class LidarGroundSegmenter {
public:
    LidarGroundSegmenter();

    /**
     * @brief Segments ground and non-ground points from an input point cloud.
     * @param input_cloud_raw The raw input point cloud.
     * @param ground_cloud Output cloud containing identified ground points.
     * @param non_ground_cloud Output cloud containing identified non-ground points.
     * @param output_leveled_input_cloud Output cloud of the input after any ground plane leveling transform.
     */
    void segmentGround(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud_raw,
                       pcl::PointCloud<pcl::PointXYZI>::Ptr& ground_cloud,
                       pcl::PointCloud<pcl::PointXYZI>::Ptr& non_ground_cloud,
                       pcl::PointCloud<pcl::PointXYZI>::Ptr& output_leveled_input_cloud);

    // Getters for stored clouds (useful for external access/visualization)
    pcl::PointCloud<pcl::PointXYZI>::Ptr getGroundPlaneCloud() const { return m_ground_plane_cloud; }
    pcl::PointCloud<pcl::PointXYZI>::Ptr getNonGroundPlaneCloud() const { return m_non_ground_plane_cloud; }

   
    

private:
    double m_voxel_leaf_size;
    double m_normal_search_radius; // Still useful if we reintroduce normal estimation
    int m_max_iterations;
    double m_distance_threshold;
    int m_min_plane_points;
    double m_max_ground_angle_rad;

    // Stored point clouds for internal use or external access
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_ground_plane_cloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_non_ground_plane_cloud;

    // This function will now be integrated/called within segmentGround more selectively
    bool alignCloudWithGround(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
                              pcl::PointCloud<pcl::PointXYZI>::Ptr& output_leveled_cloud,
                              Eigen::Affine3f& transform_out, // Added to return the transform
                              pcl::ModelCoefficients::Ptr& plane_coefficients_out); // Added to return coeffs

    // pcl::visualization::PCLVisualizer::Ptr m_viewer; // Commented out from your original
};

#endif // GROUNDSEG_HPP