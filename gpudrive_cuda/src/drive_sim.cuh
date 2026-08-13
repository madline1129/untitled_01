#pragma once

#include "runtime_scene.hpp"

#include <cuda_runtime_api.h>

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

enum class ControlMode : std::uint8_t {
    Auto = 0,
    Residual = 1,
    Direct = 2,
};

// 动态自行车模型的内部状态。速度分量使用车体坐标系。
struct VehicleDynamicsState {
    float longitudinal_velocity;
    float lateral_velocity;
    float yaw_rate;
    float steering_angle;
    float longitudinal_acceleration;
};

// 一组可配置的车辆动力学标定参数。
struct VehicleDynamicsPreset {
    float mass;
    float yaw_inertia;
    float front_cornering_stiffness;
    float rear_cornering_stiffness;
    float tire_friction;
};

// CPU 根据 Agent 类型和尺寸生成的逐车参数。
struct VehicleParameters {
    float mass;
    float yaw_inertia;
    float front_axle_distance;
    float rear_axle_distance;
    float front_cornering_stiffness;
    float rear_cornering_stiffness;
    float tire_friction;
    int use_dynamic_model;
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
    int collided_ego;
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
    float end_x;
    float end_y;
    float type;
    float speed_limit;
    float distance;
    int valid;
};

struct SignalObservation {
    int feature_index;
    int state;
    int valid;
};

struct SimConfig {
    int partner_observations = 16;
    int map_observations = 64;
    int tracker_lookahead_steps = 3;
    float min_acceleration = -6.0f;
    float max_acceleration = 4.0f;
    float max_abs_steering = 0.6f;
    float max_speed = 40.0f;
    float goal_threshold = 2.0f;
    float red_light_stop_distance = 20.0f;

    // 执行器响应和积分参数。
    int dynamics_substeps = 5;
    float acceleration_time_constant = 0.25f;
    float steering_time_constant = 0.15f;
    float max_jerk = 8.0f;
    float max_steering_rate = 0.8f;

    // 低速使用运动学模型，高速使用动态模型，中间平滑混合。
    float kinematic_speed_threshold = 1.5f;
    float dynamic_speed_threshold = 3.0f;
    float max_abs_slip_angle = 0.5f;
    float max_abs_yaw_rate = 2.5f;

    // 车辆尺寸分类阈值。
    float passenger_max_length = 6.5f;
    float passenger_max_width = 2.5f;

    VehicleDynamicsPreset passenger_vehicle{
        1500.0f, 2500.0f, 80000.0f, 80000.0f, 0.9f};
    VehicleDynamicsPreset large_vehicle{
        8000.0f, 35000.0f, 160000.0f, 200000.0f, 0.8f};
};

struct SimSnapshot {
    std::vector<AgentState> states;
    std::vector<AgentAction> applied_actions;
    std::vector<VehicleDynamicsState> dynamics_states;
    std::vector<AgentEvent> events;
    std::vector<std::uint8_t> valid;
    std::vector<std::uint8_t> external_control;
    std::vector<std::uint8_t> control_modes;
    std::vector<int> world_steps;
    std::vector<int> world_done;
};

// Torch 扩展通过这些稳定的设备指针构造 CUDA Tensor 视图。
// 指针只在 DriveSim 生命周期内有效，调用 reset/step 不会改变地址。
struct SimDeviceView {
    const AgentState *states;
    const VehicleDynamicsState *dynamics_states;
    const AgentAction *applied_actions;
    const AgentEvent *events;
    const SelfObservation *self_observations;
    const PartnerObservation *partner_observations;
    const MapObservation *map_observations;
    const std::uint8_t *valid;
    const std::int32_t *agent_type;
    const std::uint8_t *agent_is_ego;
    const std::uint8_t *agent_controllable;
    const std::uint8_t *control_modes;
    const int *world_steps;
    const int *world_done;
};

class DriveSim {
public:
    DriveSim(SimConfig config, std::vector<RuntimeScene> scenes);
    ~DriveSim();

    DriveSim(const DriveSim &) = delete;
    DriveSim &operator=(const DriveSim &) = delete;
    DriveSim(DriveSim &&) = delete;
    DriveSim &operator=(DriveSim &&) = delete;

    void reset();
    void reset_worlds(const std::vector<int> &world_ids);
    void reset_worlds_device(const std::uint8_t *reset_mask, cudaStream_t stream);
    void set_external_control_mask(const std::vector<std::uint8_t> &mask);
    void set_control_modes(const std::vector<std::uint8_t> &modes);
    void set_control_modes_device(const std::uint8_t *modes, cudaStream_t stream);
    void set_actions(const std::vector<AgentAction> &actions);
    void step();
    void step_device(const AgentAction *actions, cudaStream_t stream);

    SimSnapshot copy_snapshot() const;
    std::vector<SelfObservation> copy_self_observations() const;
    std::vector<PartnerObservation> copy_partner_observations() const;
    std::vector<MapObservation> copy_map_observations() const;
    std::vector<SignalObservation> copy_signal_observations() const;
    SimDeviceView device_view() const;

    int num_worlds() const { return num_worlds_; }
    int max_agents() const { return capacities_.max_agents; }
    int partner_observation_count() const { return config_.partner_observations; }
    int map_observation_count() const { return config_.map_observations; }

private:
    void reset_worlds_impl(const std::uint8_t *reset_mask, cudaStream_t stream);
    void step_impl(const AgentAction *actions, cudaStream_t stream);
    void build_observations(cudaStream_t stream);
    void allocate_and_upload(const std::vector<RuntimeScene> &scenes);
    void release() noexcept;

    SimConfig config_;
    RuntimeCapacities capacities_;
    int num_worlds_ = 0;
    int total_agents_ = 0;

    AgentState *d_initial_states_ = nullptr;
    std::uint8_t *d_initial_valid_ = nullptr;
    std::int32_t *d_agent_type_ = nullptr;
    std::uint8_t *d_agent_is_ego_ = nullptr;
    std::uint8_t *d_agent_controllable_ = nullptr;
    AgentDimensions *d_agent_dimensions_ = nullptr;
    VehicleParameters *d_vehicle_parameters_ = nullptr;
    Point2 *d_agent_goals_ = nullptr;
    std::uint8_t *d_agent_goal_valid_ = nullptr;
    AgentState *d_reference_future_ = nullptr;
    std::uint8_t *d_reference_future_valid_ = nullptr;

    Point3 *d_map_points_ = nullptr;
    std::int32_t *d_map_feature_type_ = nullptr;
    std::int32_t *d_map_geometry_type_ = nullptr;
    std::int32_t *d_map_feature_point_start_ = nullptr;
    std::int32_t *d_map_feature_point_count_ = nullptr;
    std::uint8_t *d_map_feature_valid_ = nullptr;
    float *d_map_speed_limit_ = nullptr;
    std::uint8_t *d_map_speed_limit_valid_ = nullptr;
    std::int32_t *d_traffic_light_feature_index_ = nullptr;
    std::uint8_t *d_traffic_light_state_ = nullptr;
    std::uint8_t *d_traffic_light_valid_ = nullptr;

    float *d_world_dt_ = nullptr;
    int *d_episode_steps_ = nullptr;
    int *d_map_feature_counts_ = nullptr;
    int *d_traffic_light_counts_ = nullptr;

    AgentState *d_states_ = nullptr;
    VehicleDynamicsState *d_dynamics_states_ = nullptr;
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
    std::uint8_t *d_world_reset_mask_ = nullptr;
};

static_assert(sizeof(AgentState) == sizeof(float) * 5, "AgentState must match runtime state layout");
static_assert(sizeof(AgentAction) == sizeof(float) * 2, "AgentAction must be tightly packed");
static_assert(sizeof(AgentEvent) == sizeof(int) * 5, "AgentEvent must be tightly packed");
static_assert(
    sizeof(VehicleDynamicsState) == sizeof(float) * 5,
    "VehicleDynamicsState must be tightly packed");
static_assert(sizeof(AgentDimensions) == sizeof(float) * 3, "AgentDimensions must be tightly packed");
static_assert(sizeof(Point2) == sizeof(float) * 2, "Point2 must be tightly packed");
static_assert(sizeof(Point3) == sizeof(float) * 3, "Point3 must be tightly packed");
