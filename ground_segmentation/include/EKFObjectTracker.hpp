#pragma once
#define SPDLOG_FMT_EXTERNAL
#include <deque>
#include <vector>
#include <memory>
#include <Eigen/Dense> // For Eigen::VectorXd, Eigen::MatrixXd

// PCL includes for point cloud processing
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <limits> // For std::numeric_limits
#include <spdlog/spdlog.h> // For logging

// Define an enum for object types (reused from comparing_clouds)
enum class ObjectType {
    UNKNOWN = 0,
    HUMAN
};

// Struct to hold information about a clustered cloud in the current frame
struct CloudInfo {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
    double timestamp;
    Eigen::Vector3f center_of_mass;
    float min_x, max_x;
    float min_y, max_y;
    float min_z, max_z;
    float extent_x, extent_y, extent_z;
    int id; // Will be assigned by the tracker during association
    ObjectType detected_type; // Type detected for this specific cloud info
};

// Represents a tracked object using an EKF
struct TrackedObjectEKF {
    int id;
    Eigen::VectorXd x; // EKF State vector [x, y, z, vx, vy, vz]
    Eigen::MatrixXd P; // EKF Covariance matrix
    double last_measurement_timestamp;
    int missed_detections; // Counter for unmatched frames
    bool is_falling;
    bool is_static;
    ObjectType classified_type; // Stable classified type for the tracked object

    // History of estimated states and associated measurements for fall/static checks and visualization
    std::deque<Eigen::VectorXd> state_history;
    std::deque<double> timestamp_history;
    std::deque<CloudInfo> measurement_history; // Stores the raw cluster info that updated this track

    TrackedObjectEKF(int id_val, const Eigen::VectorXd& initial_x, const Eigen::MatrixXd& initial_P, double timestamp)
        : id(id_val), x(initial_x), P(initial_P), last_measurement_timestamp(timestamp),
          missed_detections(0), is_falling(false), is_static(false), classified_type(ObjectType::UNKNOWN)
    {
        state_history.push_back(initial_x);
        timestamp_history.push_back(timestamp);
    }
};

class EKFObjectTracker {
public:
    EKFObjectTracker(
        double assoc_dist,             // Max Mahalanobis distance for association
        int max_missed_detections,     // Max frames before track deletion
        double max_history_time,       // How long to keep state history (seconds)
        double min_fall_z_vel,         // Z velocity threshold for falling (m/s)
        double min_fall_height_change, // Min height drop over a short period (m)
        double fall_check_duration_sec,// Time window for fall height change check (seconds)
        double static_threshold_dist,  // Max displacement for static (m)
        int min_static_frames          // Min frames to consider for static status
    );

    // Main function to process incoming clusters for tracking
    void processClusters(
        const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& new_clusters,
        double current_timestamp);

    // Getter for visualization purposes
    const std::vector<TrackedObjectEKF>& getTrackedObjects() const { return m_tracked_objects; }

private:
    std::vector<TrackedObjectEKF> m_tracked_objects;
    int m_next_object_id;

    // EKF Parameters
    Eigen::MatrixXd m_Q; // Process noise covariance (Constant Velocity model)
    Eigen::MatrixXd m_R; // Measurement noise covariance (for [x, y, z] measurements)
    Eigen::MatrixXd m_P_initial; // Initial covariance for new tracks

    // Tracking Parameters
    double m_association_threshold_sq; // Squared Mahalanobis distance threshold for association
    int m_max_missed_detections;       // Maximum allowed consecutive missed detections before track removal
    double m_max_history_time;         // Maximum duration for storing track history

    // Fall Detection Parameters
    double m_min_fall_z_velocity;      // Minimum downward velocity to be considered falling
    double m_min_fall_height_change;   // Minimum height change over 'fall_check_duration_sec' for falling
    double m_fall_check_duration_sec;  // Time window for fall detection based on height change

    // Static Object Detection Parameters
    double m_static_threshold_dist_sq; // Maximum squared displacement to be considered static
    int m_min_static_frames;           // Minimum frames required to determine static status

    // --- Helper functions for cluster processing (reused from previous code) ---
    Eigen::Vector3f calculateCentroid(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud) const;
    void calculateBoundingBox(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                              float &min_x, float &max_x,
                              float &min_y, float &max_y,
                              float &min_z, float &max_z) const;
    ObjectType classifyCloud(float extent_x, float extent_y, float extent_z, size_t num_points) const;

    // --- EKF Core Logic ---
    Eigen::MatrixXd getF(double dt) const; // Motion model Jacobian for Constant Velocity
    Eigen::MatrixXd getH() const;         // Measurement model Jacobian for [x, y, z] measurements

    // Predicts the state and covariance of a single track
    void predictTrack(TrackedObjectEKF& track, double dt);

    // Updates the state and covariance of a single track with a measurement
    void updateTrack(TrackedObjectEKF& track, const CloudInfo& measurement);

    // --- Data Association ---
    void associate(
        const std::vector<CloudInfo>& current_clusters,
        std::vector<int>& cluster_to_track_map, // Output: maps cluster index to matched track index (-1 if unmatched)
        std::vector<bool>& track_matched_this_frame // Output: indicates if a track was matched in current frame
    );

    // --- Track Management ---
    void manageTrackLifecycle(double current_timestamp);

    // --- Object State Determination ---
    void determineObjectState(TrackedObjectEKF& track, double current_timestamp);
};