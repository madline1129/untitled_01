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
};

struct SimSnapshot {
    std::vector<AgentState> states;
    std::vector<AgentAction> applied_actions;
    std::vector<AgentEvent> events;
    std::vector<std::uint8_t> valid;
    std::vector<std::uint8_t> external_control;
    std::vector<int> world_steps;
    std::vector<int> world_done;
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
    void set_external_control_mask(const std::vector<std::uint8_t> &mask);
    void set_actions(const std::vector<AgentAction> &actions);
    void step();

    SimSnapshot copy_snapshot() const;
    std::vector<SelfObservation> copy_self_observations() const;
    std::vector<PartnerObservation> copy_partner_observations() const;
    std::vector<MapObservation> copy_map_observations() const;
    std::vector<SignalObservation> copy_signal_observations() const;

    int num_worlds() const { return num_worlds_; }
    int max_agents() const { return capacities_.max_agents; }
    int partner_observation_count() const { return config_.partner_observations; }
    int map_observation_count() const { return config_.map_observations; }

private:
    void build_observations();
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

static_assert(sizeof(AgentState) == sizeof(float) * 5, "AgentState must match runtime state layout");
static_assert(sizeof(AgentAction) == sizeof(float) * 2, "AgentAction must be tightly packed");
static_assert(sizeof(AgentDimensions) == sizeof(float) * 3, "AgentDimensions must be tightly packed");
static_assert(sizeof(Point2) == sizeof(float) * 2, "Point2 must be tightly packed");
static_assert(sizeof(Point3) == sizeof(float) * 3, "Point3 must be tightly packed");
