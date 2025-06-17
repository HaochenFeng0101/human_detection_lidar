#ifndef STATIC_CLUSTERER_HPP
#define STATIC_CLUSTERER_HPP

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <vector>
#include <string>

// PCL clustering specific includes
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h> // Still needed for clustering
#include <pcl/filters/voxel_grid.h> // New include for Voxel Grid filter

class StaticClusterer {
public:
    // Constructor with parameters for clustering and denoising
    StaticClusterer(double cluster_tolerance = 0.2, int min_cluster_size = 10, int max_cluster_size = 2500,
                    double voxel_leaf_size = 0.1); // New parameter for denoising

    /**
     * @brief Performs static clustering on a non-ground point cloud, with an initial denoising step.
     * @param input_non_ground_cloud The input point cloud, assumed to be non-ground and ego-motion compensated.
     * @param clustered_objects_out A vector of point clouds, where each element is a detected cluster (object).
     */
    void cluster(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_non_ground_cloud,
                 std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clustered_objects_out);

private:
    double m_cluster_tolerance; // Corresponds to DBSCAN's 'eps'
    int m_min_cluster_size;
    int m_max_cluster_size;
    double m_voxel_leaf_size;   // Voxel grid resolution for denoising/downsampling
};

#endif // STATIC_CLUSTERER_HPP