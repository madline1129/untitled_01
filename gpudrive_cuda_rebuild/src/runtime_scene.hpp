#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// 同一个 CUDA batch 中的所有场景必须使用相同容量。
struct RuntimeCapacities {
    int max_agents = 0;
    int history_steps = 0;
    int max_future_steps = 0;
    int max_map_features = 0;
    int max_map_points = 0;
    int max_map_edges = 0;
    int max_traffic_lights = 0;
    int max_route_features = 0;

    // 比较两个场景的全部张量容量是否完全一致。
    bool operator==(const RuntimeCapacities &other) const;
};

struct RuntimeCounts {
    // padding 之前的真实 Agent 数量。
    int agents = 0;
    int map_features = 0;
    int map_points = 0;
    int map_edges = 0;
    int traffic_lights = 0;
    int route_features = 0;
};

struct RuntimeScene {
    // 场景来源信息。
    std::filesystem::path directory;
    std::string source_scenario_id;

    // 当前 world 独立使用的时间间隔和 episode 长度。
    float dt = 0.1f;
    int episode_steps = 0;

    RuntimeCapacities capacities;
    RuntimeCounts counts;

    std::vector<std::string> agent_ids;

    // Agent reset 张量。
    std::vector<float> agent_initial_state;
    std::vector<std::uint8_t> agent_initial_valid;

    // Agent 元数据张量。
    std::vector<std::int32_t> agent_type;          // 形状：[max_agents]
    std::vector<std::uint8_t> agent_is_ego;        // 形状：[max_agents]
    std::vector<std::uint8_t> agent_controllable;  // 形状：[max_agents]

    // 形状：[max_agents, 3]，字段为 [length, width, height]。
    std::vector<float> agent_dimensions;

    // simulator 内部控制器使用的私有目标。
    std::vector<float> agent_goal;                 // 形状：[max_agents, 2]
    std::vector<std::uint8_t> agent_goal_valid;    // 形状：[max_agents]

    // 记录的未来轨迹属于 simulator 私有数据，不能进入策略输入。
    // 形状：[max_future_steps, max_agents, 5]。
    std::vector<float> reference_future;
    std::vector<std::uint8_t> reference_future_valid;

    // 展平后的地图几何。
    std::vector<float> map_points;                 // 形状：[max_map_points, 3]
    std::vector<std::uint8_t> map_point_valid;     // 形状：[max_map_points]

    // 地图 feature 元数据。
    std::vector<std::int32_t> map_feature_type;
    std::vector<std::int32_t> map_geometry_type;
    std::vector<std::int32_t> map_feature_point_start;
    std::vector<std::int32_t> map_feature_point_count;
    std::vector<std::uint8_t> map_feature_valid;

    // 可选的道路速度限制。
    std::vector<float> map_speed_limit;
    std::vector<std::uint8_t> map_speed_limit_valid;

    // 形状：[max_map_edges, 3]，字段为 [source, target, relation]。
    std::vector<std::int32_t> map_edges;
    std::vector<std::uint8_t> map_edge_valid;

    // 交通灯张量。
    std::vector<std::int32_t> traffic_light_feature_index;
    std::vector<std::uint8_t> traffic_light_state;
    std::vector<std::uint8_t> traffic_light_valid;

    // 有序路线数据。
    std::vector<std::int32_t> route_feature_index;
    std::vector<std::uint8_t> route_feature_valid;
    std::vector<float> route_goal;                 // [2] = [x, y]
    std::vector<std::uint8_t> route_goal_valid;    // [1]
};

RuntimeScene load_runtime_scene(
    const std::filesystem::path &directory
);

std::vector<std::filesystem::path> discover_runtime_scenes(
    const std::filesystem::path &input
);

std::vector<RuntimeScene> load_runtime_batch(
    const std::filesystem::path &input,
    int num_worlds
);
