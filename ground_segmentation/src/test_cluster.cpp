#include <iostream>
#include <string>
#include <pcl/io/pcd_io.h>
// #include <pcl/visualization/pcl_visualizer.h> // For visualizing results
#include <thread> // For std::this_thread::sleep_for
#include <chrono> // For std::chrono::milliseconds

#include "StaticClusterer.hpp"
#include "groundseg.hpp"
// #include "DBSCAN_kdtree.h"
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>



int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_non_ground_pcd_file.pcd>" << std::endl;
        return 1;
    }

    std::string input_file_path = argv[1];
    pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(input_file_path, *input_cloud) == -1) {
        PCL_ERROR("Couldn't read file %s \n", input_file_path.c_str());
        return 1;
    }

    LidarGroundSegmenter segmenter;
    pcl::PointCloud<pcl::PointXYZI>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr leveled_input_for_viz(new pcl::PointCloud<pcl::PointXYZI>);

    segmenter.segmentGround(input_cloud, ground_cloud, non_ground_cloud, leveled_input_for_viz);


    // pcl::PointCloud<pcl::PointXYZI>::Ptr input_non_ground_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    std::cout << "Loaded " << non_ground_cloud->points.size() << " points from " << input_file_path << std::endl;

    // Create a StaticClusterer instance
    

    
    // cluster_tolerance: max distance between points to be in the same cluster (meters)
    // min_cluster_size: minimum points for a valid cluster
    // max_cluster_size: maximum points for a valid cluster
    StaticClusterer clusterer(0.2, 20, 200,0.15); 
    

    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> detected_objects;
    clusterer.cluster(non_ground_cloud, detected_objects);

    std::cout << "Total detected objects: " << detected_objects.size() << std::endl;

    for (size_t i = 0; i < detected_objects.size(); ++i) {
        std::cout << "Object " << i + 1 << ": " << detected_objects[i]->points.size() << " points." << std::endl;
        
        // Optionally save each detected object to a PCD file
        std::string output_filename = "/home/dc/dconstruct/dynamic_detection/ground_segmentation/build/res/detected_object_" + std::to_string(i + 1) + ".pcd";
        pcl::io::savePCDFileASCII(output_filename, *detected_objects[i]);
        std::cout << "Saved detected object to " << output_filename << std::endl;
    }
    // Visualize the clustered objects
    // visualizeClouds(non_ground_cloud, detected_objects);

    return 0;
}