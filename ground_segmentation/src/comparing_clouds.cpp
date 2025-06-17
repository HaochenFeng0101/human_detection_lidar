#include "comparing_clouds.hpp"
#include <iostream>
#include <algorithm> // For std::find_if, std::remove_if, std::max, std::min
#include <cmath>     // For std::pow, std::sqrt, std::abs
#include <spdlog/spdlog.h>

// Constructor
comparing_clouds::comparing_clouds(double max_hist_time, double assoc_dist,
                                   double fall_height_change, double fall_duration,
                                   double static_dist, int min_static_frames)
    : m_max_history_time(max_hist_time),
      m_association_dist_sq(assoc_dist * assoc_dist),
      m_min_fall_height_change(fall_height_change),
      m_min_fall_duration(fall_duration),
      m_static_threshold_dist_sq(static_dist * static_dist),
      m_min_static_frames(min_static_frames),
      m_next_object_id(0)
{
    std::cout << "ComparingClouds initialized with parameters:" << std::endl;
    std::cout << "  Max History Time: " << m_max_history_time << "s" << std::endl;
    std::cout << "  Association Distance: " << assoc_dist << "m" << std::endl;
    std::cout << "  Min Fall Height Change: " << m_min_fall_height_change << "m" << std::endl;
    std::cout << "  Min Fall Duration: " << m_min_fall_duration << "s" << std::endl;
    std::cout << "  Static Threshold Distance: " << static_dist << "m" << std::endl;
    std::cout << "  Min Static Frames: " << m_min_static_frames << std::endl;
}

// Destructor
comparing_clouds::~comparing_clouds()
{
    // Smart pointers and deques handle their own memory, no explicit cleanup needed.
}

// Clears all tracked objects and resets ID counter.
void comparing_clouds::clearTrackedObjects()
{
    m_tracked_objects.clear();
    m_next_object_id = 0; // Reset ID counter to start fresh
    std::cout << "All tracked objects cleared." << std::endl;
}

Eigen::Vector3f comparing_clouds::calculateCentroid(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud)
{
    if (cloud->empty())
    {
        return Eigen::Vector3f::Zero();
    }
    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    for (const auto &p : cloud->points)
    {
        sum += p.getVector3fMap();
    }
    return sum / static_cast<float>(cloud->points.size());
}

// Calculate full bounding box min/max for X, Y, Z
void comparing_clouds::calculateBoundingBox(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                                            float &min_x, float &max_x,
                                            float &min_y, float &max_y,
                                            float &min_z, float &max_z)
{
    min_x = min_y = min_z = std::numeric_limits<float>::max();
    max_x = max_y = max_z = std::numeric_limits<float>::lowest();

    if (cloud->empty())
    {
        min_x = max_x = min_y = max_y = min_z = max_z = 0.0f;
        return;
    }

    for (const auto &p : cloud->points)
    {
        if (p.x < min_x)
            min_x = p.x;
        if (p.x > max_x)
            max_x = p.x;
        if (p.y < min_y)
            min_y = p.y;
        if (p.y > max_y)
            max_y = p.y;
        if (p.z < min_z)
            min_z = p.z;
        if (p.z > max_z)
            max_z = p.z;
    }
}

// Modified: Classify object type only for Human
ObjectType comparing_clouds::classifyCloud(float extent_x, float extent_y, float extent_z, size_t num_points) const
{
    // Basic sanity check for point count. Adjust threshold as needed for noise.
    if (num_points < 20) // Increased point threshold for more robust detection
    {
        return ObjectType::UNKNOWN;
    }

    // Consider variations (crouching, lying down) might fall outside these simple AABB checks.
    const float HUMAN_MIN_HEIGHT = 1.0f;     // Min height for a standing human (m) - e.g., child or shorter adult
    const float HUMAN_MAX_HEIGHT = 2.0f;     // Max height for a tall human (m)
    const float HUMAN_MIN_HORIZONTAL = 0.2f; // Min horizontal (X/Y) extent for human body thickness (m)
    const float HUMAN_MAX_HORIZONTAL = 1.2f; // Max horizontal (X/Y) extent

    const float med_height_and_horizon = 0.5f;

    // Check for Human
    if (extent_z >= HUMAN_MIN_HEIGHT && extent_z <= HUMAN_MAX_HEIGHT)
    {
        // Both horizontal dimensions (X and Y) should be within human-like bounds, and not both too large
        if (extent_x >= HUMAN_MIN_HORIZONTAL && extent_x <= HUMAN_MAX_HORIZONTAL &&
            extent_y >= HUMAN_MIN_HORIZONTAL && extent_y <= HUMAN_MAX_HORIZONTAL && !(extent_x > med_height_and_horizon && extent_y > med_height_and_horizon))
        {

            // This can help differentiate from a wide, low object that might accidentally fit horizontal bounds.
            float max_horizontal_extent = std::max(extent_x, extent_y);
            if (extent_z / max_horizontal_extent > 1.0f) // Height is greater than max horizontal extent
            {
                return ObjectType::HUMAN;
            }
        }
    }

    // If it doesn't fit human criteria or is too small
    return ObjectType::UNKNOWN;
}

int comparing_clouds::findBestMatch(const CloudInfo &current_info,
                                    const std::vector<TrackedObject> &tracked_objects_to_search,
                                    const std::vector<bool> &original_tracked_object_matched_flags,
                                    size_t search_limit) const
{
    //find best match with distance
    double min_dist_sq = std::numeric_limits<double>::max();
    int best_match_idx = -1;

    for (size_t j = 0; j < search_limit; ++j)
    {
        if (original_tracked_object_matched_flags[j])
            continue;

        const auto &tracked_obj = tracked_objects_to_search[j];
        Eigen::Vector3f diff = current_info.center_of_mass - tracked_obj.current_centroid;
        double dist_sq = diff.squaredNorm();

        if (dist_sq < m_association_dist_sq && dist_sq < min_dist_sq)
        {
            min_dist_sq = dist_sq;
            best_match_idx = j;
        }
    }
    return best_match_idx;
}

void comparing_clouds::process_non_grounded_points(
    const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &new_non_ground_objects_clusters,
    double current_timestamp,
    std::vector<int> &falling_objects_out,
    std::vector<int> &static_objects_out,
    std::vector<int> &moved_objects_out)
{
    spdlog::info("Processing {} new non-ground object clusters at timestamp {}",
                 new_non_ground_objects_clusters.size(), current_timestamp);
    falling_objects_out.clear();
    static_objects_out.clear();
    moved_objects_out.clear();

    std::vector<CloudInfo> current_frame_infos;
    for (const auto &cluster_cloud : new_non_ground_objects_clusters)
    {
        if (!cluster_cloud->empty())
        {   
            if (cluster_cloud->points.size() < 55 || cluster_cloud->points.size() > 100)
            {
                spdlog::info("Skipping cluster with {} points (too few or too many).",
                             cluster_cloud->points.size());
                continue; // Skip clusters 
            }
           
            
            CloudInfo info;
            info.cloud = cluster_cloud;
            info.timestamp = current_timestamp;
            info.center_of_mass = calculateCentroid(cluster_cloud);

            calculateBoundingBox(cluster_cloud,
                                 info.min_x, info.max_x,
                                 info.min_y, info.max_y,
                                 info.min_z, info.max_z);
            // Calculate extents based on min max xyz 
            info.extent_x = info.max_x - info.min_x;
            info.extent_y = info.max_y - info.min_y;
            info.extent_z = info.max_z - info.min_z;

            // Classify the cloud based on its calculated extents
            info.detected_type = classifyCloud(info.extent_x, info.extent_y, info.extent_z, cluster_cloud->points.size());

            current_frame_infos.push_back(info);

            // Add classification info to log output for debugging
            std::string object_type_str;
            switch (info.detected_type)
            {
            case ObjectType::HUMAN:
                object_type_str = "Human";
                break;
            case ObjectType::UNKNOWN:
                object_type_str = "Unknown";
                break;
            }

            spdlog::info("Processed cluster (Points: {}), CoM ({:.2f}, {:.2f}, {:.2f}), Extents ({:.2f}x{:.2f}x{:.2f}), Type: {}",
                         cluster_cloud->points.size(),
                         current_frame_infos.back().center_of_mass.x(),
                         current_frame_infos.back().center_of_mass.y(),
                         current_frame_infos.back().center_of_mass.z(),
                         current_frame_infos.back().extent_x,
                         current_frame_infos.back().extent_y,
                         current_frame_infos.back().extent_z,
                         object_type_str);
        }
    }

    // --- 1. Associate current objects with existing tracked objects ---
    spdlog::info("Current frame clusters (CloudInfo) count: {}", current_frame_infos.size());

    size_t initial_tracked_objects_size = m_tracked_objects.size();
    spdlog::info("Initial tracked objects count: {}", initial_tracked_objects_size);

    std::vector<bool> original_tracked_object_matched_this_frame(initial_tracked_objects_size, false);

    for (size_t i = 0; i < current_frame_infos.size(); ++i)
    {
        auto &current_info = current_frame_infos[i];

        int best_match_idx = findBestMatch(current_info, m_tracked_objects,
                                           original_tracked_object_matched_this_frame,
                                           initial_tracked_objects_size);

        if (best_match_idx != -1)
        {
            // Match found! Update existing tracked object
            TrackedObject &matched_tracked_obj = m_tracked_objects[best_match_idx];

            matched_tracked_obj.history.push_back(current_info);

            // Keep history within max time
            while (!matched_tracked_obj.history.empty() &&
                   (current_timestamp - matched_tracked_obj.history.front().timestamp > m_max_history_time))
            {
                matched_tracked_obj.history.pop_front();
            }

            // Update current state of tracked object
            matched_tracked_obj.current_centroid = current_info.center_of_mass;
            matched_tracked_obj.current_min_z = current_info.min_z;
            matched_tracked_obj.current_max_z = current_info.max_z;
            matched_tracked_obj.current_extent_x = current_info.extent_x;
            matched_tracked_obj.current_extent_y = current_info.extent_y;
            matched_tracked_obj.current_extent_z = current_info.extent_z;
            matched_tracked_obj.last_update_time = current_timestamp;

            matched_tracked_obj.is_static = false;
            matched_tracked_obj.is_falling = false;

            // Only update classified_type if it's a HUMAN detection
            if (current_info.detected_type == ObjectType::HUMAN)
            {
                matched_tracked_obj.classified_type = ObjectType::HUMAN;
            }
            else
            {
                // If the current detection is not human, but it was previously tracked as human,
                
                
            }

            current_info.id = matched_tracked_obj.id;
            original_tracked_object_matched_this_frame[best_match_idx] = true;
        }
        else
        {
            // No match found for current object, it's a new object.
            // Only create a new tracked object if it's classified as HUMAN.
            if (current_info.detected_type == ObjectType::HUMAN)
            {
                TrackedObject new_tracked_obj;
                new_tracked_obj.id = m_next_object_id++;
                current_info.id = new_tracked_obj.id;
                new_tracked_obj.history.push_back(current_info);
                new_tracked_obj.current_centroid = current_info.center_of_mass;
                new_tracked_obj.current_min_z = current_info.min_z;
                new_tracked_obj.current_max_z = current_info.max_z;
                new_tracked_obj.current_extent_x = current_info.extent_x;
                new_tracked_obj.current_extent_y = current_info.extent_y;
                new_tracked_obj.current_extent_z = current_info.extent_z;
                new_tracked_obj.last_update_time = current_timestamp;
                new_tracked_obj.is_static = false;
                new_tracked_obj.is_falling = false;
                new_tracked_obj.classified_type = current_info.detected_type; // Should be HUMAN here

                m_tracked_objects.push_back(new_tracked_obj);
            }
            else
            {
                // If it's not a human and no match, just ignore it (don't track it)
                spdlog::debug("Cluster not classified as HUMAN and no match found. Skipping tracking.");
            }
        }
    }

    // --- 2. Update status and remove old or unmatched tracked objects ---
    std::vector<TrackedObject> next_tracked_objects;
    for (size_t j = 0; j < m_tracked_objects.size(); ++j)
    {
        TrackedObject &tracked_obj = m_tracked_objects[j];

        bool was_updated_this_frame;
        if (j < initial_tracked_objects_size)
        {
            was_updated_this_frame = original_tracked_object_matched_this_frame[j];
        }
        else
        {
            was_updated_this_frame = true;
        }

        // If an object was NOT updated in this frame and its last update is too old, remove it.
        // Or if its classified_type has somehow become UNKNOWN (implies it's no longer considered human)
        if (!was_updated_this_frame || tracked_obj.classified_type == ObjectType::UNKNOWN) // Added check for UNKNOWN type
        {
            if (current_timestamp - tracked_obj.last_update_time > m_max_history_time ||
                tracked_obj.classified_type == ObjectType::UNKNOWN) // Remove if type reverted
            {
                spdlog::info("Removing tracked object ID {} due to no update for too long or type changed to UNKNOWN.", tracked_obj.id);
                continue;
            }
            tracked_obj.is_falling = false;
        }

        // Check for static status (only relevant for HUMANs you are tracking)
        if (tracked_obj.classified_type == ObjectType::HUMAN && tracked_obj.history.size() >= m_min_static_frames)
        {
            Eigen::Vector3f first_centroid = tracked_obj.history.front().center_of_mass;
            Eigen::Vector3f last_centroid = tracked_obj.history.back().center_of_mass;
            double displacement_sq = (last_centroid - first_centroid).squaredNorm();

            if (displacement_sq < m_static_threshold_dist_sq)
            {
                tracked_obj.is_static = true;
                static_objects_out.push_back(tracked_obj.id);
            }
            else
            {
                tracked_obj.is_static = false;
            }
        }
        else
        {
            tracked_obj.is_static = false;
        }

        // Check for falling (only relevant for HUMANs you are tracking)
        if (!tracked_obj.is_static && tracked_obj.history.size() >= 2)
        {
            const CloudInfo &first_info_in_history = tracked_obj.history.front();
            const CloudInfo &last_info_in_history = tracked_obj.history.back();

            double time_span = last_info_in_history.timestamp - first_info_in_history.timestamp;
            float z_change = first_info_in_history.min_z - last_info_in_history.min_z;

            if (z_change > m_min_fall_height_change && time_span > 0 && time_span < m_min_fall_duration)
            {
                tracked_obj.is_falling = true;
                falling_objects_out.push_back(tracked_obj.id);
            }
            else
            {
                tracked_obj.is_falling = false;
            }
        }
        else
        {
            tracked_obj.is_falling = false;
        }

        // Report as 'moved' only if it's a HUMAN and not falling/static
        if (tracked_obj.classified_type == ObjectType::HUMAN && !tracked_obj.is_falling && !tracked_obj.is_static && was_updated_this_frame)
        {
            if (tracked_obj.history.size() > 1 || initial_tracked_objects_size > 0)
            {
                moved_objects_out.push_back(tracked_obj.id);
            }
        }

        // Only add to next_tracked_objects if it's still classified as HUMAN
        if (tracked_obj.classified_type == ObjectType::HUMAN)
        {
            next_tracked_objects.push_back(tracked_obj);
        }
        else
        {
            spdlog::debug("Not retaining tracked object ID {} because its type is no longer HUMAN.", tracked_obj.id);
        }
    }
    m_tracked_objects = next_tracked_objects;

    std::cout << "Tracked objects after processing: " << m_tracked_objects.size() << std::endl;
}

std::vector<TrackedObject> comparing_clouds::getAllTrackedObjects() const
{
    // Filter to only return HUMANS if that's the only interest
    std::vector<TrackedObject> humans;
    for (const auto &obj : m_tracked_objects)
    {
        if (obj.classified_type == ObjectType::HUMAN)
        {
            humans.push_back(obj);
        }
    }
    return humans;
}

const TrackedObject *comparing_clouds::getTrackedObjectById(int object_id) const
{
    auto it = std::find_if(m_tracked_objects.begin(), m_tracked_objects.end(),
                           [object_id](const TrackedObject &obj)
                           {
                               return obj.id == object_id && obj.classified_type == ObjectType::HUMAN; // Only return if human
                           });
    if (it != m_tracked_objects.end())
    {
        return &(*it);
    }
    return nullptr;
}