#include "comparing_clouds.hpp"
#include <iostream>
#include <algorithm> // For std::find_if, std::remove_if, std::max, std::min
#include <cmath>     // For std::pow, std::sqrt, std::abs
#include <spdlog/spdlog.h>
#include <pcl/io/pcd_io.h> // For saving PCD files
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

// Mark as const here as well
Eigen::Vector3f comparing_clouds::calculateCentroid(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud) const
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

// Mark as const here as well
void comparing_clouds::calculateBoundingBox(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                                            float &min_x, float &max_x,
                                            float &min_y, float &max_y,
                                            float &min_z, float &max_z) const
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
    // // Basic sanity check for point count. Adjust threshold as needed for noise.
    // if (num_points < 20) // Increased point threshold for more robust detection
    // {
    //     return ObjectType::UNKNOWN;
    // }

    // Consider variations (crouching, lying down) might fall outside these simple AABB checks.
    const float HUMAN_MIN_HEIGHT = 0.3f;     // Min height for a standing/falling human (m)
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

// === START OF NEWLY REFACTORED FUNCTIONS ===

// New function: Extract CloudInfo from raw clusters
std::vector<CloudInfo> comparing_clouds::processCurrentFrameClusters(
    const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &new_non_ground_objects_clusters,
    double current_timestamp) const
{
    std::vector<CloudInfo> current_frame_infos;
    for (const auto &cluster_cloud : new_non_ground_objects_clusters)
    {
        if (cluster_cloud->empty()) continue; // Skip empty clusters

        // Optional point count filtering (as commented out in original)
        // if (cluster_cloud->points.size() < 40 || cluster_cloud->points.size() > 100)
        // {
        //     spdlog::info("Skipping cluster with {} points (too few or too many) 40 100.",
        //                  cluster_cloud->points.size());
        //     continue;
        // }

        CloudInfo info;
        info.timestamp = current_timestamp;
        info.cloud = cluster_cloud; // Assign cloud to info

        // These calls are now valid because calculateCentroid and calculateBoundingBox are const
        info.center_of_mass = calculateCentroid(cluster_cloud);

        calculateBoundingBox(cluster_cloud,
                             info.min_x, info.max_x,
                             info.min_y, info.max_y,
                             info.min_z, info.max_z);

        info.extent_x = info.max_x - info.min_x;
        info.extent_y = info.max_y - info.min_y;
        info.extent_z = info.max_z - info.min_z;

        info.detected_type = classifyCloud(info.extent_x, info.extent_y, info.extent_z, cluster_cloud->points.size());
        current_frame_infos.push_back(info);

        // Add classification info to log output for debugging (uncomment for verbose logging)
        // std::string object_type_str;
        // switch (info.detected_type)
        // {
        // case ObjectType::HUMAN: object_type_str = "Human"; break;
        // case ObjectType::UNKNOWN: object_type_str = "Unknown"; break;
        // }
        // spdlog::info("Processed cluster (Points: {}), CoM ({:.2f}, {:.2f}, {:.2f}), Extents ({:.2f}x{:.2f}x{:.2f}), Type: {}",
        //              cluster_cloud->points.size(),
        //              current_frame_infos.back().center_of_mass.x(),
        //              current_frame_infos.back().center_of_mass.y(),
        //              current_frame_infos.back().center_of_mass.z(),
        //              current_frame_infos.back().extent_x,
        //              current_frame_infos.back().extent_y,
        //              current_frame_infos.back().extent_z,
        //              object_type_str);
    }
    return current_frame_infos;
}

// Handles association, history updates, and TrackedObject lifecycle
void comparing_clouds::associateAndUpdateTracks(
    std::vector<CloudInfo> &current_frame_infos, // current_frame_infos passed by reference to modify IDs
    double current_timestamp)
{
    spdlog::info("Associating and updating tracks. Current frame clusters: {}", current_frame_infos.size());

    size_t initial_tracked_objects_size = m_tracked_objects.size();
    std::vector<bool> original_tracked_object_matched_this_frame(initial_tracked_objects_size, false);

    for (size_t i = 0; i < current_frame_infos.size(); ++i)
    {
        auto &current_info = current_frame_infos[i];

        int best_match_idx = findBestMatch(current_info, m_tracked_objects,
                                           original_tracked_object_matched_this_frame,
                                           initial_tracked_objects_size);
                                           //find best match id with id 

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

            // Reset flags (state determination happens later)
            matched_tracked_obj.is_static = false;
            matched_tracked_obj.is_falling = false;

            // Only update classified_type if it's a HUMAN detection
            if (current_info.detected_type == ObjectType::HUMAN)
            {
                matched_tracked_obj.classified_type = ObjectType::HUMAN;
            }
            // else: If current is not human, but it was previously tracked as human,
            // we leave its classified_type as HUMAN for now. It will be removed later
            // if it stops being updated OR if the classification rules are refined to change its type.
            // The check for UNKNOWN type removal is done below.

            current_info.id = matched_tracked_obj.id; // Assign ID to current CloudInfo
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
                current_info.id = new_tracked_obj.id; // Assign ID to current CloudInfo
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
                spdlog::info("New tracked object created with ID {} at timestamp {:.2f}", new_tracked_obj.id, current_timestamp);
            }
            else
            {
                // If it's not a human and no match, just ignore it (don't track it)
                spdlog::debug("Cluster not classified as HUMAN and no match found. Skipping tracking.");
            }
        }
    }
    
    // This loop handles:
    // 1. Objects that haven't been updated for too long (disappeared).
    // 2. Objects whose `classified_type` somehow reverted to `UNKNOWN` (no longer looks like a human).
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
            // This case handles newly added objects in this frame (those pushed back into m_tracked_objects
            // within this `associateAndUpdateTracks` call). They were, by definition, updated this frame.
            was_updated_this_frame = true;
        }

        if (!was_updated_this_frame || tracked_obj.classified_type == ObjectType::UNKNOWN)
        {
            if (current_timestamp - tracked_obj.last_update_time > m_max_history_time ||
                tracked_obj.classified_type == ObjectType::UNKNOWN)
            {
                spdlog::info("Removing tracked object ID {} due to no update for too long or type changed to UNKNOWN.", tracked_obj.id);
                continue; // Skip adding to next_tracked_objects
            }
        }
        // If it was not updated but is still within history time, keep it but clear fall/static flags
        if (!was_updated_this_frame) {
            tracked_obj.is_falling = false;
            tracked_obj.is_static = false;
        }

        // Only add to next_tracked_objects if it's still classified as HUMAN
        if (tracked_obj.classified_type == ObjectType::HUMAN)
        {
            next_tracked_objects.push_back(tracked_obj);
        }
    }
    m_tracked_objects = next_tracked_objects; // Update the main list of tracked objects

    spdlog::info("Tracked objects after association and pruning: {}", m_tracked_objects.size());
}

// Determines static, falling, and moved states for *current* tracked objects
void comparing_clouds::determineObjectStates(
    double current_timestamp,
    std::vector<int> &falling_objects_out,
    std::vector<int> &static_objects_out,
    std::vector<int> &moved_objects_out)
{
    falling_objects_out.clear();
    static_objects_out.clear();
    moved_objects_out.clear();

    spdlog::info("Determining states for {} tracked objects.", m_tracked_objects.size());

    for (auto &tracked_obj : m_tracked_objects)
    {
        // Reset flags before re-evaluating (associateAndUpdateTracks already does this, but good to be explicit here too)
        tracked_obj.is_static = false;
        tracked_obj.is_falling = false;

        // Ensure it's a HUMAN and has enough history to check for static/falling
        if (tracked_obj.classified_type != ObjectType::HUMAN || tracked_obj.history.empty())
        {
            continue; // Only check human objects with history
        }

        // Check for static status
        if (tracked_obj.history.size() >= m_min_static_frames)
        {
            Eigen::Vector3f first_centroid = tracked_obj.history.front().center_of_mass;
            Eigen::Vector3f last_centroid = tracked_obj.history.back().center_of_mass;
            double displacement_sq = (last_centroid - first_centroid).squaredNorm();

            if (displacement_sq < m_static_threshold_dist_sq)
            {
                tracked_obj.is_static = true;
                static_objects_out.push_back(tracked_obj.id);
            }
        }

        // Check for falling (only if not static, and has enough history)
        if (!tracked_obj.is_static && tracked_obj.history.size() >= 5)
        {
            const CloudInfo &first_info_in_history = tracked_obj.history.front();
            const CloudInfo &last_info_in_history = tracked_obj.history.back();

            Eigen::Vector3f first_centroid = first_info_in_history.center_of_mass;
            Eigen::Vector3f last_centroid = last_info_in_history.center_of_mass;

            // Height change is typically based on Z-axis (up/down).
            // A fall implies a *decrease* in Z-coordinate, so (first_Z - last_Z) should be positive.
            double height_drop = first_centroid.z() - last_centroid.z();
            double time_span = last_info_in_history.timestamp - first_info_in_history.timestamp;

            spdlog::debug("Object ID {}: Height drop: {:.2f}m, Time span: {:.2f}s", tracked_obj.id, height_drop, time_span);

            // Condition for falling: significant height drop within a short duration
            if (height_drop > m_min_fall_height_change && time_span < m_min_fall_duration)
            {
                tracked_obj.is_falling = true;
                falling_objects_out.push_back(tracked_obj.id);
                spdlog::warn("DETECTED FALL for object ID {}! Height drop: {:.2f}m, Duration: {:.2f}s",
                             tracked_obj.id, height_drop, time_span);
            }
        }

        // Report as 'moved' if it's a HUMAN and not falling/static AND was updated this frame
        if (tracked_obj.classified_type == ObjectType::HUMAN && !tracked_obj.is_falling && !tracked_obj.is_static &&
            (tracked_obj.last_update_time == current_timestamp)) // Check if it was updated in the *current* frame
        {
            // Ensure there's enough history to distinguish from a new object that just appeared.
            // If it has more than just the current frame's info, it means it has moved.
            if (tracked_obj.history.size() > 1)
            {
                moved_objects_out.push_back(tracked_obj.id);
            }
        }
    }
}



// Orchestrates the overall processing flow
void comparing_clouds::process_non_grounded_points(
    const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &new_non_ground_objects_clusters,
    double current_timestamp,
    std::vector<int> &falling_objects_out,
    std::vector<int> &static_objects_out,
    std::vector<int> &moved_objects_out)
{
    spdlog::info("Processing {} new non-ground object clusters at timestamp {}",
                 new_non_ground_objects_clusters.size(), current_timestamp);

    // Step 1: Process raw clusters into CloudInfo objects for the current frame
    std::vector<CloudInfo> current_frame_infos = processCurrentFrameClusters(new_non_ground_objects_clusters, current_timestamp);
    spdlog::info("Generated {} CloudInfo objects for current frame.", current_frame_infos.size());

    // Step 2: Associate current frame objects with existing tracks, update them, and manage track lifecycle
    associateAndUpdateTracks(current_frame_infos, current_timestamp);
    spdlog::info("Finished association and track updates. {} tracks remaining.", m_tracked_objects.size());

    // Step 3: Determine the static, falling, and moved states of the current set of tracked objects
    determineObjectStates(current_timestamp, falling_objects_out, static_objects_out, moved_objects_out);
    spdlog::info("Identified {} falling, {} static, {} moved objects.",
                 falling_objects_out.size(), static_objects_out.size(), moved_objects_out.size());

    std::cout << "Tracked objects after full processing: " << m_tracked_objects.size() << std::endl;
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