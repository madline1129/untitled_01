#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

struct RuntimeCapacities {
    int max_agents = 0;
    int history_steps = 0;
    int max_future_steps = 0;
    int max_map_features = 0;
    int max_map_points = 0;
    int max_map_edges = 0;
    int max_traffic_lights = 0;
    int max_route_features = 0;

    bool operator==(const RuntimeCapacities &other) const;
};

struct RuntimeCounts {
    int agents = 0;
    int map_features = 0;
    int map_points = 0;
    int map_edges = 0;
    int traffic_lights = 0;
    int route_features = 0;
};

struct RuntimeScene {
    std::filesystem::path directory;
    std::string source_scenario_id;
    float dt = 0.1f;
    int episode_steps = 0;
    RuntimeCapacities capacities;
    RuntimeCounts counts;
    std::vector<std::string> agent_ids;

    std::vector<float> agent_initial_state;
    std::vector<std::uint8_t> agent_initial_valid;
    std::vector<std::int32_t> agent_type;
    std::vector<std::uint8_t> agent_is_ego;
    std::vector<std::uint8_t> agent_controllable;
    std::vector<float> agent_dimensions;
    std::vector<float> agent_goal;
    std::vector<std::uint8_t> agent_goal_valid;
    std::vector<float> reference_future;
    std::vector<std::uint8_t> reference_future_valid;

    std::vector<float> map_points;
    std::vector<std::uint8_t> map_point_valid;
    std::vector<std::int32_t> map_feature_type;
    std::vector<std::int32_t> map_geometry_type;
    std::vector<std::int32_t> map_feature_point_start;
    std::vector<std::int32_t> map_feature_point_count;
    std::vector<std::uint8_t> map_feature_valid;
    std::vector<float> map_speed_limit;
    std::vector<std::uint8_t> map_speed_limit_valid;
    std::vector<std::int32_t> map_edges;
    std::vector<std::uint8_t> map_edge_valid;

    std::vector<std::int32_t> traffic_light_feature_index;
    std::vector<std::uint8_t> traffic_light_state;
    std::vector<std::uint8_t> traffic_light_valid;
    std::vector<std::int32_t> route_feature_index;
    std::vector<std::uint8_t> route_feature_valid;
    std::vector<float> route_goal;
    std::vector<std::uint8_t> route_goal_valid;
};

RuntimeScene load_runtime_scene(const std::filesystem::path &directory);

std::vector<std::filesystem::path> discover_runtime_scenes(
    const std::filesystem::path &input);

std::vector<RuntimeScene> load_runtime_batch(
    const std::filesystem::path &input,
    int num_worlds);
