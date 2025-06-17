#include "StaticClusterer.hpp"
#include <iostream>
#include <pcl/io/pcd_io.h> // Needed if you want to save denoised cloud or individual clusters

// Constructor implementation
StaticClusterer::StaticClusterer(double cluster_tolerance, int min_cluster_size, int max_cluster_size, double voxel_leaf_size)
    : m_cluster_tolerance(cluster_tolerance),
      m_min_cluster_size(min_cluster_size),
      m_max_cluster_size(max_cluster_size),
      m_voxel_leaf_size(voxel_leaf_size) { // Initialize new parameter
    std::cout << "StaticClusterer initialized with: "
              << "Cluster Tolerance=" << m_cluster_tolerance << "m, "
              << "MinClusterSize=" << m_min_cluster_size << ", "
              << "MaxClusterSize=" << m_max_cluster_size << ", "
              << "VoxelLeafSize=" << m_voxel_leaf_size << "m" << std::endl;
}

// Cluster function implementation
void StaticClusterer::cluster(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_non_ground_cloud,
                             std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clustered_objects_out) {
    
    // Clear any previous clusters
    clustered_objects_out.clear();

    if (input_non_ground_cloud->empty()) {
        std::cout << "Input non-ground cloud is empty, no clustering performed." << std::endl;
        return;
    }

    std::cout << "Starting processing for " << input_non_ground_cloud->points.size() << " non-ground points." << std::endl;

    // --- Denoising Step: Voxel Grid Filter ---
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    vg.setInputCloud(input_non_ground_cloud);
    vg.setLeafSize(m_voxel_leaf_size, m_voxel_leaf_size, m_voxel_leaf_size);
    vg.filter(*cloud_filtered);

    std::cout << "After VoxelGrid filter, cloud has " << cloud_filtered->points.size() << " points." << std::endl;

    if (cloud_filtered->empty()) {
        std::cout << "Filtered cloud is empty after denoising, no clustering performed." << std::endl;
        return;
    }

    // --- Clustering Step: Euclidean Cluster Extraction (DBSCAN-like) ---
    // Create a KdTree object for the search method of the extraction
    // This is essential for efficiently finding neighbors in DBSCAN
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(cloud_filtered); // Use the denoised cloud for clustering

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;

    ec.setClusterTolerance(m_cluster_tolerance); // Set the spatial tolerance (eps)
    ec.setMinClusterSize(m_min_cluster_size);   // Set the minimum number of points required to form a cluster (MinPts)
    ec.setMaxClusterSize(m_max_cluster_size);   // Set the maximum number of points for a cluster
    ec.setSearchMethod(tree);                   // Provide the KdTree for efficient neighborhood search
    ec.setInputCloud(cloud_filtered);           // Set the denoised input cloud
    ec.extract(cluster_indices);                // Perform the clustering

    std::cout << "Found " << cluster_indices.size() << " clusters." << std::endl;

    int j = 0;
    for (const auto& indices : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
        for (const int& idx : indices.indices) {
            cluster->points.push_back(cloud_filtered->points[idx]);
        }
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true; // All points are valid

        clustered_objects_out.push_back(cluster);

        // Optional: Save each cluster to a separate PCD file for inspection
        // std::string filename = "cluster_" + std::to_string(j++) + ".pcd";
        // pcl::io::savePCDFileASCII(filename, *cluster);
        // std::cout << "  Saved cluster " << j-1 << " with " << cluster->points.size() << " points to " << filename << std::endl;
        j++;
    }

    std::cout << "Clustering complete." << std::endl;
}