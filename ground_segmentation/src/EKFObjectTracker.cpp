#include "EKFObjectTracker.hpp"
#include <spdlog/spdlog.h> // For logging
#include <limits>           // For std::numeric_limits
#include <algorithm>        // For std::min, std::max, std::find_if
#include <cmath>            // For std::sqrt, std::pow

// Constructor: Initializes EKF parameters and tracking thresholds
EKFObjectTracker::EKFObjectTracker(
    double assoc_dist,
    int max_missed_detections_param,
    double max_history_time_param,
    double min_fall_z_vel,
    double min_fall_height_change,
    double fall_check_duration_sec,
    double static_threshold_dist,
    int min_static_frames_param)
    : m_next_object_id(0),
      m_association_threshold_sq(assoc_dist * assoc_dist), // Square the distance for comparison
      m_max_missed_detections(max_missed_detections_param),
      m_max_history_time(max_history_time_param),
      m_min_fall_z_velocity(min_fall_z_vel),
      m_min_fall_height_change(min_fall_height_change),
      m_fall_check_duration_sec(fall_check_duration_sec),
      m_static_threshold_dist_sq(static_threshold_dist * static_threshold_dist), // Square the distance
      m_min_static_frames(min_static_frames_param)
{
    // Initialize EKF Process Noise Covariance (Q) for Constant Velocity Model
    // State: [x, y, z, vx, vy, vz]
    m_Q.setIdentity(6, 6);
    // Tune these values based on expected system noise (how much the object can deviate from CV model)
    m_Q(0, 0) = 0.1; m_Q(1, 1) = 0.1; m_Q(2, 2) = 0.1; // Position variance
    m_Q(3, 3) = 0.5; m_Q(4, 4) = 0.5; m_Q(5, 5) = 0.5; // Velocity variance (often higher)

    // Initialize EKF Measurement Noise Covariance (R) for [x, y, z] measurements
    m_R.setIdentity(3, 3);
    // Tune these based on sensor noise and cluster centroid accuracy
    m_R(0, 0) = 0.1; m_R(1, 1) = 0.1; m_R(2, 2) = 0.1;

    // Initial Covariance (P) for new tracks (high uncertainty)
    m_P_initial.setIdentity(6, 6);
    m_P_initial *= 1000.0; // Very high initial uncertainty
    m_P_initial(3, 3) = 100.0; // Velocity uncertainties
    m_P_initial(4, 4) = 100.0;
    m_P_initial(5, 5) = 100.0;

    spdlog::info("EKFObjectTracker initialized with parameters.");
}

// Helper: Calculate centroid of a point cloud
Eigen::Vector3f EKFObjectTracker::calculateCentroid(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud) const
{
    if (cloud->empty()) {
        return Eigen::Vector3f::Zero();
    }
    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    for (const auto &p : cloud->points) {
        sum += p.getVector3fMap();
    }
    return sum / static_cast<float>(cloud->points.size());
}

// Helper: Calculate bounding box extents of a point cloud
void EKFObjectTracker::calculateBoundingBox(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                                            float &min_x, float &max_x,
                                            float &min_y, float &max_y,
                                            float &min_z, float &max_z) const
{
    min_x = min_y = min_z = std::numeric_limits<float>::max();
    max_x = max_y = max_z = std::numeric_limits<float>::lowest();

    if (cloud->empty()) {
        min_x = max_x = min_y = max_y = min_z = max_z = 0.0f;
        return;
    }

    for (const auto &p : cloud->points) {
        if (p.x < min_x) min_x = p.x;
        if (p.x > max_x) max_x = p.x;
        if (p.y < min_y) min_y = p.y;
        if (p.y > max_y) max_y = p.y;
        if (p.z < min_z) min_z = p.z;
        if (p.z > max_z) max_z = p.z;
    }
}

// Helper: Classify a cluster as UNKNOWN or HUMAN based on its dimensions and point count
ObjectType EKFObjectTracker::classifyCloud(float extent_x, float extent_y, float extent_z, size_t num_points) const
{
    if (num_points < 20) { // Filter out very small clusters likely to be noise
        return ObjectType::UNKNOWN;
    }

    const float HUMAN_MIN_HEIGHT = 0.3f;
    const float HUMAN_MAX_HEIGHT = 2.0f;
    const float HUMAN_MIN_HORIZONTAL = 0.2f;
    const float HUMAN_MAX_HORIZONTAL = 1.2f;
    const float med_height_and_horizon = 0.5f;

    if (extent_z >= HUMAN_MIN_HEIGHT && extent_z <= HUMAN_MAX_HEIGHT) {
        if (extent_x >= HUMAN_MIN_HORIZONTAL && extent_x <= HUMAN_MAX_HORIZONTAL &&
            extent_y >= HUMAN_MIN_HORIZONTAL && extent_y <= HUMAN_MAX_HORIZONTAL) {
            // Avoid classifying wide, flat objects as human
            if (!(extent_x > med_height_and_horizon && extent_y > med_height_and_horizon)) {
                float max_horizontal_extent = std::max(extent_x, extent_y);
                if (extent_z >= max_horizontal_extent * 0.7f) { // Height is sufficiently large compared to width/depth
                    return ObjectType::HUMAN;
                }
            }
        }
    }
    return ObjectType::UNKNOWN;
}

// EKF: Get Motion Model Jacobian (F) for Constant Velocity
Eigen::MatrixXd EKFObjectTracker::getF(double dt) const {
    Eigen::MatrixXd F(6, 6);
    F.setIdentity();
    F(0, 3) = dt; // x = x + vx*dt
    F(1, 4) = dt; // y = y + vy*dt
    F(2, 5) = dt; // z = z + vz*dt
    return F;
}

// EKF: Get Measurement Model Jacobian (H) for measuring [x, y, z]
Eigen::MatrixXd EKFObjectTracker::getH() const {
    Eigen::MatrixXd H(3, 6);
    H.setZero();
    H(0, 0) = 1; // Measure x
    H(1, 1) = 1; // Measure y
    H(2, 2) = 1; // Measure z
    return H;
}

// EKF: Predict step for a single tracked object
void EKFObjectTracker::predictTrack(TrackedObjectEKF& track, double dt) {
    Eigen::MatrixXd F = getF(dt);

    // State Prediction (Constant Velocity)
    track.x(0) += track.x(3) * dt;
    track.x(1) += track.x(4) * dt;
    track.x(2) += track.x(5) * dt;
    // Velocities (x(3), x(4), x(5)) remain unchanged in CV model

    // Covariance Prediction
    track.P = F * track.P * F.transpose() + m_Q;
}

// EKF: Update step for a single tracked object with a measurement
void EKFObjectTracker::updateTrack(TrackedObjectEKF& track, const CloudInfo& measurement) {
    Eigen::VectorXd z_meas(3);
    z_meas << measurement.center_of_mass.x(), measurement.center_of_mass.y(), measurement.center_of_mass.z();

    Eigen::MatrixXd H = getH(); // Measurement model Jacobian

    Eigen::VectorXd z_pred = H * track.x; // Predicted measurement from current state

    Eigen::VectorXd y = z_meas - z_pred; // Innovation (measurement residual)

    Eigen::MatrixXd S = H * track.P * H.transpose() + m_R; // Innovation covariance
    Eigen::MatrixXd K = track.P * H.transpose() * S.inverse(); // Kalman Gain

    // Update state and covariance
    track.x = track.x + K * y;
    track.P = (Eigen::MatrixXd::Identity(6, 6) - K * H) * track.P;

    // Update last measurement timestamp and reset missed detections counter
    track.last_measurement_timestamp = measurement.timestamp;
    track.missed_detections = 0;

    // Store state and measurement history
    track.state_history.push_back(track.x);
    track.timestamp_history.push_back(measurement.timestamp);
    track.measurement_history.push_back(measurement);

    // Trim history to adhere to max_history_time
    while (!track.state_history.empty() &&
           (measurement.timestamp - track.timestamp_history.front() > m_max_history_time))
    {
        track.state_history.pop_front();
        track.timestamp_history.pop_front();
        // Only pop measurement_history if it also aligns with the oldest state/timestamp
        // Assuming they are pushed/popped together for consistency.
        if (!track.measurement_history.empty()) {
            track.measurement_history.pop_front();
        }
    }
}

// Data Association: Matches current clusters to existing tracks
void EKFObjectTracker::associate(
    const std::vector<CloudInfo>& current_clusters,
    std::vector<int>& cluster_to_track_map,
    std::vector<bool>& track_matched_this_frame)
{
    cluster_to_track_map.assign(current_clusters.size(), -1); // Initialize all clusters as unmatched
    track_matched_this_frame.assign(m_tracked_objects.size(), false); // Initialize all tracks as unmatched

    if (current_clusters.empty() || m_tracked_objects.empty()) {
        return; // Nothing to associate if either list is empty
    }

    // Cost matrix based on Mahalanobis distance
    Eigen::MatrixXd costs(current_clusters.size(), m_tracked_objects.size());
    costs.setConstant(std::numeric_limits<double>::max()); // Initialize with very high cost

    Eigen::MatrixXd H = getH(); // Measurement model Jacobian

    for (size_t i = 0; i < current_clusters.size(); ++i) {
        const auto& cluster = current_clusters[i];
        Eigen::VectorXd z_meas(3);
        z_meas << cluster.center_of_mass.x(), cluster.center_of_mass.y(), cluster.center_of_mass.z();

        for (size_t j = 0; j < m_tracked_objects.size(); ++j) {
            const auto& track = m_tracked_objects[j];

            // Predicted measurement and innovation covariance
            Eigen::VectorXd z_pred = H * track.x;
            Eigen::VectorXd y = z_meas - z_pred; // Residual

            // Calculate innovation covariance S = H*P*H' + R
            // Ensure S is positive definite. Add a small epsilon if needed for numerical stability.
            Eigen::MatrixXd S = H * track.P * H.transpose() + m_R;
            
            // Mahalanobis distance squared: y' * S_inv * y
            // Using LLT for inverse for numerical stability with symmetric positive definite matrices
            double mahalanobis_sq = y.transpose() * S.llt().solve(y);

            if (mahalanobis_sq < m_association_threshold_sq) {
                costs(i, j) = mahalanobis_sq;
            }
        }
    }

    // Simple greedy assignment (can be replaced by Hungarian algorithm for optimal assignment)
    // Iteratively find the best match and remove it from consideration
    for (size_t k = 0; k < std::min(current_clusters.size(), m_tracked_objects.size()); ++k) {
        double min_cost = std::numeric_limits<double>::max();
        int best_cluster_idx = -1;
        int best_track_idx = -1;

        for (size_t i = 0; i < current_clusters.size(); ++i) {
            if (cluster_to_track_map[i] != -1) continue; // Cluster already assigned

            for (size_t j = 0; j < m_tracked_objects.size(); ++j) {
                if (track_matched_this_frame[j]) continue; // Track already assigned

                if (costs(i, j) < min_cost) {
                    min_cost = costs(i, j);
                    best_cluster_idx = i;
                    best_track_idx = j;
                }
            }
        }

        if (best_cluster_idx != -1) { // A valid match was found
            cluster_to_track_map[best_cluster_idx] = best_track_idx;
            track_matched_this_frame[best_track_idx] = true;
            // Invalidate costs for the matched cluster and track to ensure they are not reused
            costs.row(best_cluster_idx).setConstant(std::numeric_limits<double>::max());
            costs.col(best_track_idx).setConstant(std::numeric_limits<double>::max());
        } else {
            break; // No more valid matches possible
        }
    }
    spdlog::debug("Associated {} clusters to tracks.", std::count_if(cluster_to_track_map.begin(), cluster_to_track_map.end(), [](int i){ return i != -1; }));
}

// Manages the lifecycle of tracks (e.g., removal of old tracks)
void EKFObjectTracker::manageTrackLifecycle(double current_timestamp) {
    auto it = m_tracked_objects.begin();
    while (it != m_tracked_objects.end()) {
        // Remove track if it has missed too many consecutive detections
        if (it->missed_detections > m_max_missed_detections) {
            spdlog::info("Removing track ID {} due to {} missed detections (max allowed {}).",
                         it->id, it->missed_detections, m_max_missed_detections);
            it = m_tracked_objects.erase(it);
        } else {
            ++it;
        }
    }
}

// Determines if a tracked object is falling or static
void EKFObjectTracker::determineObjectState(TrackedObjectEKF& track, double current_timestamp) {
    track.is_falling = false;
    track.is_static = false;

    // A. Check for STATIC status
    // Requires sufficient history for reliable displacement calculation
    if (track.state_history.size() >= m_min_static_frames) {
        Eigen::Vector3d first_estimated_pos = track.state_history.front().head<3>().cast<double>();
        Eigen::Vector3d last_estimated_pos = track.x.head<3>().cast<double>(); // Current estimated position

        double displacement_sq = (last_estimated_pos - first_estimated_pos).squaredNorm();

        if (displacement_sq < m_static_threshold_dist_sq) {
            track.is_static = true;
        }
    }

    // B. Check for FALLING status
    // Condition 1: High downward Z-velocity
    double estimated_z_velocity = track.x(5); // Get vz component from EKF state
    if (estimated_z_velocity < -m_min_fall_z_velocity) { // Negative velocity indicates downward motion
        track.is_falling = true;
        spdlog::warn("Track ID {} detected falling (Z velocity: {:.2f} m/s)", track.id, estimated_z_velocity);
    }

    // Condition 2: Significant height drop over a short time window from history
    if (track.state_history.size() >= 2) { // Need at least two states to calculate height change
        auto it_end_ts = track.timestamp_history.rbegin(); // Latest timestamp
        auto it_start_ts = track.timestamp_history.rbegin();
        size_t end_idx = track.timestamp_history.size() - 1;

        // Find the oldest timestamp within the 'fall_check_duration_sec' window
        while(it_start_ts != track.timestamp_history.rend() &&
              (*it_end_ts - *it_start_ts) < m_fall_check_duration_sec) {
            ++it_start_ts;
        }

        if (it_start_ts != track.timestamp_history.rend()) { // Found a valid start point
            size_t start_idx = track.timestamp_history.size() - 1 - std::distance(track.timestamp_history.rbegin(), it_start_ts);

            double initial_z = track.state_history[start_idx](2); // Z position at start of window
            double final_z = track.state_history[end_idx](2);     // Z position at end of window

            double height_drop = initial_z - final_z; // Positive value if Z decreased

            if (height_drop > m_min_fall_height_change) {
                track.is_falling = true;
                spdlog::warn("Track ID {} detected falling (Height drop: {:.2f}m over {:.2f}s)",
                             track.id, height_drop, (*it_end_ts - *it_start_ts));
            }
        }
    }
    // If either condition above makes is_falling true, it remains true.
}

// Main processing loop for incoming clusters
void EKFObjectTracker::processClusters(
    const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& new_clusters,
    double current_timestamp)
{
    spdlog::info("EKF: Processing {} raw clusters at timestamp {}.", new_clusters.size(), current_timestamp);

    // 1. Convert raw clusters to `CloudInfo` objects, filtering for human-like shapes
    std::vector<CloudInfo> current_frame_infos;
    for (const auto& cluster_cloud : new_clusters) {
        if (cluster_cloud->empty()) continue;

        CloudInfo info;
        info.timestamp = current_timestamp;
        info.cloud = cluster_cloud;
        info.center_of_mass = calculateCentroid(cluster_cloud);
        calculateBoundingBox(cluster_cloud, info.min_x, info.max_x, info.min_y, info.max_y, info.min_z, info.max_z);
        info.extent_x = info.max_x - info.min_x;
        info.extent_y = info.max_y - info.min_y;
        info.extent_z = info.max_z - info.min_z;
        info.detected_type = classifyCloud(info.extent_x, info.extent_y, info.extent_z, cluster_cloud->points.size());

        if (info.detected_type == ObjectType::HUMAN) {
             current_frame_infos.push_back(info);
        } else {
             spdlog::debug("Discarding cluster ({} points) as non-human.", cluster_cloud->points.size());
        }
    }
    spdlog::info("EKF: Filtered {} human-like clusters for tracking.", current_frame_infos.size());


    // 2. Predict all existing tracks to the current timestamp
    for (auto& track : m_tracked_objects) {
        if (track.last_measurement_timestamp > 0) { // Only predict if track has previous measurement
            double dt = current_timestamp - track.last_measurement_timestamp;
            if (dt > 0) {
                 predictTrack(track, dt);
            }
        }
        // Increment missed detections for all tracks *before* association
        // This makes sure tracks that *don't* get matched in this frame count a miss.
        track.missed_detections++; // Will be reset to 0 in updateTrack if matched
    }

    // 3. Data Association: Match current clusters to predicted tracks
    std::vector<int> cluster_to_track_map; // Maps cluster index to track index (-1 if unmatched)
    std::vector<bool> track_matched_this_frame; // Flags if a track was matched

    associate(current_frame_infos, cluster_to_track_map, track_matched_this_frame);
    spdlog::debug("Completed data association.");

    // 4. Update existing tracks and create new ones
    for (size_t i = 0; i < current_frame_infos.size(); ++i) {
        CloudInfo& cluster = current_frame_infos[i];
        int matched_track_idx = cluster_to_track_map[i];

        if (matched_track_idx != -1) {
            // Found a match: Update the existing track with the current measurement
            updateTrack(m_tracked_objects[matched_track_idx], cluster);
            spdlog::debug("Updated track ID {} with cluster {}.", m_tracked_objects[matched_track_idx].id, i);
        } else {
            // No match for this cluster: Create a new track
            Eigen::VectorXd initial_x(6);
            // Initialize position from cluster centroid, velocities to zero
            initial_x << cluster.center_of_mass.x(), cluster.center_of_mass.y(), cluster.center_of_mass.z(),
                         0.0, 0.0, 0.0;

            TrackedObjectEKF new_track(m_next_object_id++, initial_x, m_P_initial, current_timestamp);
            new_track.classified_type = ObjectType::HUMAN; // New tracks are only created for HUMAN types
            new_track.measurement_history.push_back(cluster); // Store the first measurement
            new_track.last_measurement_timestamp = current_timestamp; // Set timestamp for new track
            new_track.missed_detections = 0; // It was just created, so 0 misses
            m_tracked_objects.push_back(new_track);
            spdlog::info("Created new track ID {} from unmatched cluster {}.", new_track.id, i);
        }
    }
    
    // 5. Determine object states (falling, static, etc.) for all active tracks
    for (auto& track : m_tracked_objects) {
        if (track.classified_type == ObjectType::HUMAN) {
            determineObjectState(track, current_timestamp);
        }
    }

    // 6. Manage track lifecycle (remove tracks that have missed too many detections)
    manageTrackLifecycle(current_timestamp);

    spdlog::info("EKF: Total active tracked objects: {}", m_tracked_objects.size());
}