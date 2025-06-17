#pragma once
#define SPDLOG_FMT_EXTERNAL
#include <deque>
#include <vector>
#include <memory> // For std::shared_ptr

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Dense> // For Eigen::Vector3f
#include <limits> // For std::numeric_limits

// Define an enum for object types
enum class ObjectType {
    UNKNOWN = 0, // Default or unclassified / non-human
    HUMAN        // Specifically identified as a human
};

// Structure to hold information about a cloud in a frame
struct CloudInfo {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    double timestamp;
    Eigen::Vector3f center_of_mass;
    float min_x, max_x;
    float min_y, max_y;
    float min_z, max_z;
    float extent_x, extent_y, extent_z;
    int id;
    ObjectType detected_type; // Type detected for this specific cloud info
};

// Structure to represent a tracked object
struct TrackedObject {
    int id;
    std::deque<CloudInfo> history; // History of cloud infos for this object
    Eigen::Vector3f current_centroid;
    float current_min_z;
    float current_max_z;
    float current_extent_x, current_extent_y, current_extent_z;
    double last_update_time;
    bool is_static;
    bool is_falling;
    ObjectType classified_type; // Stable classified type for the tracked object

    TrackedObject() : id(-1), current_centroid(Eigen::Vector3f::Zero()),
                      current_min_z(0.0f), current_max_z(0.0f),
                      current_extent_x(0.0f), current_extent_y(0.0f), current_extent_z(0.0f),
                      last_update_time(0.0), is_static(false), is_falling(false),
                      classified_type(ObjectType::UNKNOWN) {}
};

class comparing_clouds {
public:
    comparing_clouds(double max_hist_time, double assoc_dist,
                     double fall_height_change, double fall_duration,
                     double static_dist, int min_static_frames);
    ~comparing_clouds();

    void clearTrackedObjects();

    Eigen::Vector3f calculateCentroid(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud);

    void calculateBoundingBox(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                              float &min_x, float &max_x,
                              float &min_y, float &max_y,
                              float &min_z, float &max_z);

    // Modified to only classify for HUMAN
    ObjectType classifyCloud(float extent_x, float extent_y, float extent_z, size_t num_points) const;

    void process_non_grounded_points(
        const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& new_non_ground_objects_clusters,
        double current_timestamp,
        std::vector<int> &falling_objects_out,
        std::vector<int> &static_objects_out,
        std::vector<int> &moved_objects_out);

    std::vector<TrackedObject> getAllTrackedObjects() const;
    const TrackedObject* getTrackedObjectById(int object_id) const;

private:
    std::vector<TrackedObject> m_tracked_objects;
    double m_max_history_time;
    double m_association_dist_sq;
    double m_min_fall_height_change;
    double m_min_fall_duration;
    double m_static_threshold_dist_sq;
    int m_min_static_frames;
    int m_next_object_id;

    int findBestMatch(const CloudInfo& current_info,
                      const std::vector<TrackedObject>& tracked_objects_to_search,
                      const std::vector<bool>& original_tracked_object_matched_flags,
                      size_t search_limit) const;
};