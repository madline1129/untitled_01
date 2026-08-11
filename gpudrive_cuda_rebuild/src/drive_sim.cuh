#pragma once

#include "runtime_scene.hpp"

#include <cstdint>
#include <vector>

inline constexpr int kMaximumPartnerObservations = 16;
inline constexpr int kMaximumMapObservations = 64;

struct AgentState {
    float x;
    float y;
    float yaw;
    float vx;
    float vy;
};

struct AgentAction {
    float acceleration;
    float steering;
};

struct AgentDimensions {
    float length;
    float width;
    float height;
};

struct Point2 {
    float x;
    float y;
};

struct Point3 {
    float x;
    float y;
    float z;
};

struct AgentEvent {
    int collided_vehicle;
    int collided_road;
    int offroad;
    int reached_goal;
};

struct SelfObservation {
    float x;
    float y;
    float yaw;

    float vx;
    float vy;

    float speed;

    float length;
    float width;

    float goal_dx;
    float goal_dy;

    float steps_remaining;

    int valid;
};

struct PartnerObservation {
    float rel_x;
    float rel_y;
    float rel_vx;
    float rel_vy;
    float rel_yaw;

    float length;
    float width;
    float type;

    float distance;

    int valid;
};

struct MapObservation {
    float start_x;
    float start_y;

    // 地图线段终点在当前 Agent 坐标系中的位置。
    float end_x;
    float end_y;

    // 地图 feature 类型。
    float type;

    // 当前地图 feature 的速度限制。
    float speed_limit;

    // 当前 Agent 到该地图线段的距离。
    float distance;

    // 当前 map observation 是否有效。
    int valid;
};

// 当前时间步的交通灯观测。
struct SignalObservation {
    // 交通灯关联的地图 feature 索引。
    int feature_index;

    // 当前交通灯状态。
    int state;

    // 当前交通灯状态是否有效。
    int valid;
};

struct SimConfig {
    // 每个 Agent 保留最近的 partner 数量。
    int partner_observations = 16;

    // 每个 Agent 保留最近的地图 segment 数量。
    int map_observations = 64;

    // 自动控制器在参考未来轨迹中向前寻找多少步。
    int tracker_lookahead_steps = 3;

    // 运动学约束。
    float min_acceleration = -6.0f;
    float max_acceleration = 4.0f;
    float max_abs_steering = 0.6f;
    float max_speed = 40.0f;

    // Agent 与目标点距离小于该值时，记录 reached_goal。
    float goal_threshold = 2.0f;

    // 自动控制器开始为红灯减速的距离。
    float red_light_stop_distance = 20.0f;
};

// 从 GPU 拷贝到 CPU 的当前仿真结果。
struct SimSnapshot {
    // 形状：[num_worlds * max_agents]
    std::vector<AgentState> states;
    std::vector<AgentAction> applied_actions;
    std::vector<AgentEvent> events;
    std::vector<std::uint8_t> valid;
    std::vector<std::uint8_t> external_control;

    // 形状：[num_worlds]
    std::vector<int> world_steps;
    std::vector<int> world_done;
};

class DriveSim {
public:
    DriveSim(
        SimConfig config,
        std::vector<RuntimeScene> scenes
    );
    ~DriveSim();

    // DriveSim 内部持有原始的CUDA指针, 禁止复制, 防止两个指针都指向一个地方造成内存管理混乱
    DriveSim(const DriveSim &) = delete;
    DriveSim &operator=(const DriveSim &) = delete;
    DriveSim(DriveSim &&) = delete;
    DriveSim &operator=(DriveSim &&) = delete;

    // 重置所有world
    void reset();

    // 只重置指定的world, 其他world保持不变
    void reset_worlds(const std::vector<int> &world_ids);

    // 设置哪些Agent由外部动作接管
    // mask的形状[num_worlds * max_agents]
    void set_external_control_mask(
        const std::vector<std::uint8_t> &mask
    );

    // 设置所有Agent的外部动作
    // 只有external_control mask 为 1 的 Agent 会真正使用这些动作
    void set_actions(
        const std::vector<AgentAction> &actions);

    // 推进所有world一个时间步长
    void step();

    //GPU->CPU
    SimSnapshot copy_snapshot() const;
    std::vector<SelfObservation>
    copy_self_observations() const;
    std::vector<PartnerObservation>
    copy_partner_observations() const;
    // 从 GPU 拷贝 map observation。
    std::vector<MapObservation>
    copy_map_observations() const;

    // 从 GPU 拷贝当前交通灯 observation。
    std::vector<SignalObservation>
    copy_signal_observations() const;

    // 返回 simulator 中的 world 数量。
    int num_worlds() const
    {
        return num_worlds_;
    }

    // 返回每个 world 的固定 Agent 容量。
    int max_agents() const
    {
        return capacities_.max_agents;
    }

    // 返回每个 Agent 的 partner observation 数量。
    int partner_observation_count() const
    {
        return config_.partner_observations;
    }

    // 返回每个 Agent 的 map observation 数量。
    int map_observation_count() const
    {
        return config_.map_observations;
    }

private:
    // 根据当前状态重新生成所有策略观测。
    void build_observations();

    // 分配 GPU 内存，并上传所有 world 的只读场景数据。
    void allocate_and_upload(
        const std::vector<RuntimeScene> &scenes);

    // 释放当前对象拥有的全部 CUDA 显存。
    void release() noexcept;

    // simulator 配置和 batch 的固定容量。
    SimConfig config_;
    RuntimeCapacities capacities_;
    int num_worlds_ = 0;
    int total_agents_ = 0;

    // 以下是从 RuntimeScene 上传的只读 Agent 数据。
    AgentState *d_initial_states_ = nullptr;
    std::uint8_t *d_initial_valid_ = nullptr;
    std::int32_t *d_agent_type_ = nullptr;
    std::uint8_t *d_agent_is_ego_ = nullptr;
    std::uint8_t *d_agent_controllable_ = nullptr;
    AgentDimensions *d_agent_dimensions_ = nullptr;
    Point2 *d_agent_goals_ = nullptr;
    std::uint8_t *d_agent_goal_valid_ = nullptr;
    AgentState *d_reference_future_ = nullptr;
    std::uint8_t *d_reference_future_valid_ = nullptr;

    // 以下是从 RuntimeScene 上传的只读地图数据。
    Point3 *d_map_points_ = nullptr;
    std::int32_t *d_map_feature_type_ = nullptr;
    std::int32_t *d_map_geometry_type_ = nullptr;
    std::int32_t *d_map_feature_point_start_ = nullptr;
    std::int32_t *d_map_feature_point_count_ = nullptr;
    std::uint8_t *d_map_feature_valid_ = nullptr;
    float *d_map_speed_limit_ = nullptr;
    std::uint8_t *d_map_speed_limit_valid_ = nullptr;

    // 以下是从 RuntimeScene 上传的交通灯时间表。
    std::int32_t *d_traffic_light_feature_index_ = nullptr;
    std::uint8_t *d_traffic_light_state_ = nullptr;
    std::uint8_t *d_traffic_light_valid_ = nullptr;

    // 每个 world 独立使用的场景参数和有效数据数量。
    float *d_world_dt_ = nullptr;
    int *d_episode_steps_ = nullptr;
    int *d_map_feature_counts_ = nullptr;
    int *d_traffic_light_counts_ = nullptr;

    // 以下数据会在 reset 或每次 step 时发生变化。
    AgentState *d_states_ = nullptr;
    AgentAction *d_external_actions_ = nullptr;
    AgentAction *d_applied_actions_ = nullptr;
    std::uint8_t *d_external_control_ = nullptr;
    AgentEvent *d_events_ = nullptr;
    SelfObservation *d_self_observations_ = nullptr;
    PartnerObservation *d_partner_observations_ = nullptr;
    MapObservation *d_map_observations_ = nullptr;
    SignalObservation *d_signal_observations_ = nullptr;
    int *d_world_steps_ = nullptr;
    int *d_world_done_ = nullptr;
    int *d_world_reset_mask_ = nullptr;
};

// 这些结构会由 float 张量直接转换并上传，禁止出现额外内存 padding。
static_assert(
    sizeof(AgentState) == sizeof(float) * 5,
    "AgentState must match runtime state layout");
static_assert(
    sizeof(AgentAction) == sizeof(float) * 2,
    "AgentAction must be tightly packed");
static_assert(
    sizeof(AgentDimensions) == sizeof(float) * 3,
    "AgentDimensions must be tightly packed");
static_assert(
    sizeof(Point2) == sizeof(float) * 2,
    "Point2 must be tightly packed");
static_assert(
    sizeof(Point3) == sizeof(float) * 3,
    "Point3 must be tightly packed");
