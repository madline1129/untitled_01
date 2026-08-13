#include "drive_sim.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kGravity = 9.81f;
constexpr int kAgentTypeVehicle = 1;
constexpr int kMapTypeRoadEdge = 4;
constexpr int kMapTypeRoadblock = 8;
constexpr int kMapTypeRoadblockConnector = 9;
constexpr int kGeometryPolygon = 3;
constexpr int kTrafficLightStop = 1;
constexpr int kTrafficLightCaution = 2;
constexpr std::uint8_t kControlAuto = static_cast<std::uint8_t>(ControlMode::Auto);
constexpr std::uint8_t kControlResidual = static_cast<std::uint8_t>(ControlMode::Residual);
constexpr std::uint8_t kControlDirect = static_cast<std::uint8_t>(ControlMode::Direct);

void cuda_check(cudaError_t result, const char *expression, const char *file, int line)
{
    if (result == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << "CUDA error for " << expression << ": " << cudaGetErrorString(result)
            << " at " << file << ':' << line;
    throw std::runtime_error(message.str());
}

#define CUDA_CHECK(expression) cuda_check((expression), #expression, __FILE__, __LINE__)

void check_kernel_launch()
{
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void allocate_device(T **pointer, std::size_t count)
{
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(pointer), count * sizeof(T)));
}

template <typename T>
void upload_device(T **pointer, const std::vector<T> &values)
{
    allocate_device(pointer, values.size());
    CUDA_CHECK(cudaMemcpy(
        *pointer,
        values.data(),
        values.size() * sizeof(T),
        cudaMemcpyHostToDevice));
}

template <typename T>
std::vector<T> download_device(const T *pointer, std::size_t count)
{
    std::vector<T> values(count);
    CUDA_CHECK(cudaMemcpy(
        values.data(),
        pointer,
        count * sizeof(T),
        cudaMemcpyDeviceToHost));
    return values;
}

template <typename T>
void append_values(std::vector<T> &target, const std::vector<T> &source)
{
    target.insert(target.end(), source.begin(), source.end());
}

template <typename T>
std::vector<T> packed_structs(const std::vector<float> &values, int width)
{
    if (values.size() % static_cast<std::size_t>(width) != 0 ||
        sizeof(T) != static_cast<std::size_t>(width) * sizeof(float)) {
        throw std::runtime_error("runtime float tensor cannot be packed into CUDA struct");
    }
    std::vector<T> result(values.size() / static_cast<std::size_t>(width));
    std::memcpy(result.data(), values.data(), values.size() * sizeof(float));
    return result;
}

bool valid_dynamics_preset(const VehicleDynamicsPreset &preset)
{
    return preset.mass > 0.0f && preset.yaw_inertia > 0.0f &&
           preset.front_cornering_stiffness > 0.0f &&
           preset.rear_cornering_stiffness > 0.0f &&
           preset.tire_friction > 0.0f;
}

VehicleParameters make_vehicle_parameters(
    std::int32_t agent_type,
    const AgentDimensions &dimensions,
    const SimConfig &config)
{
    const bool passenger = dimensions.length < config.passenger_max_length &&
                           dimensions.width < config.passenger_max_width;
    const VehicleDynamicsPreset &preset =
        passenger ? config.passenger_vehicle : config.large_vehicle;
    const float length = std::max(0.1f, dimensions.length);
    const float wheelbase = std::clamp(
        0.6f * length,
        1.0f,
        std::max(1.0f, 0.9f * length));
    return VehicleParameters{
        preset.mass,
        preset.yaw_inertia,
        0.45f * wheelbase,
        0.55f * wheelbase,
        preset.front_cornering_stiffness,
        preset.rear_cornering_stiffness,
        preset.tire_friction,
        agent_type == kAgentTypeVehicle ? 1 : 0,
    };
}

__device__ float clamp_value(float value, float lower, float upper)
{
    return fminf(upper, fmaxf(lower, value));
}

__device__ float smoothstep_value(float lower, float upper, float value)
{
    const float normalized = clamp_value((value - lower) / (upper - lower), 0.0f, 1.0f);
    return normalized * normalized * (3.0f - 2.0f * normalized);
}

__device__ float update_actuator(
    float current,
    float target,
    float time_constant,
    float max_rate,
    float dt)
{
    const float remaining = target - current;
    const float desired_change = remaining * dt / time_constant;
    const float maximum_change = max_rate * dt;
    const float rate_limited = clamp_value(desired_change, -maximum_change, maximum_change);
    const float change = remaining >= 0.0f
        ? clamp_value(rate_limited, 0.0f, remaining)
        : clamp_value(rate_limited, remaining, 0.0f);
    return current + change;
}

__device__ int min_int(int left, int right)
{
    return left < right ? left : right;
}

__device__ int max_int(int left, int right)
{
    return left > right ? left : right;
}

__device__ float wrap_angle(float angle)
{
    while (angle > kPi) {
        angle -= 2.0f * kPi;
    }
    while (angle < -kPi) {
        angle += 2.0f * kPi;
    }
    return angle;
}

__device__ Point2 global_to_local(const AgentState &state, float x, float y)
{
    const float dx = x - state.x;
    const float dy = y - state.y;
    const float cosine = cosf(state.yaw);
    const float sine = sinf(state.yaw);
    Point2 result;
    result.x = cosine * dx + sine * dy;
    result.y = -sine * dx + cosine * dy;
    return result;
}

__device__ float point_segment_distance_squared(
    float px,
    float py,
    float ax,
    float ay,
    float bx,
    float by)
{
    const float dx = bx - ax;
    const float dy = by - ay;
    const float length_squared = dx * dx + dy * dy;
    if (length_squared < 1e-8f) {
        const float ex = px - ax;
        const float ey = py - ay;
        return ex * ex + ey * ey;
    }
    const float t = clamp_value(((px - ax) * dx + (py - ay) * dy) / length_squared, 0.0f, 1.0f);
    const float qx = ax + t * dx;
    const float qy = ay + t * dy;
    const float ex = px - qx;
    const float ey = py - qy;
    return ex * ex + ey * ey;
}

__device__ bool axis_separates(
    float axis_x,
    float axis_y,
    float delta_x,
    float delta_y,
    const AgentState &left,
    const AgentDimensions &left_dimensions,
    const AgentState &right,
    const AgentDimensions &right_dimensions)
{
    const float left_hx = cosf(left.yaw);
    const float left_hy = sinf(left.yaw);
    const float left_sx = -left_hy;
    const float left_sy = left_hx;
    const float right_hx = cosf(right.yaw);
    const float right_hy = sinf(right.yaw);
    const float right_sx = -right_hy;
    const float right_sy = right_hx;
    const float left_radius =
        0.5f * left_dimensions.length * fabsf(axis_x * left_hx + axis_y * left_hy) +
        0.5f * left_dimensions.width * fabsf(axis_x * left_sx + axis_y * left_sy);
    const float right_radius =
        0.5f * right_dimensions.length * fabsf(axis_x * right_hx + axis_y * right_hy) +
        0.5f * right_dimensions.width * fabsf(axis_x * right_sx + axis_y * right_sy);
    const float center_distance = fabsf(axis_x * delta_x + axis_y * delta_y);
    return center_distance > left_radius + right_radius;
}

__device__ bool obb_overlap(
    const AgentState &left,
    const AgentDimensions &left_dimensions,
    const AgentState &right,
    const AgentDimensions &right_dimensions)
{
    const float delta_x = right.x - left.x;
    const float delta_y = right.y - left.y;
    const float left_cosine = cosf(left.yaw);
    const float left_sine = sinf(left.yaw);
    const float right_cosine = cosf(right.yaw);
    const float right_sine = sinf(right.yaw);
    return !axis_separates(left_cosine, left_sine, delta_x, delta_y,
                           left, left_dimensions, right, right_dimensions) &&
           !axis_separates(-left_sine, left_cosine, delta_x, delta_y,
                           left, left_dimensions, right, right_dimensions) &&
           !axis_separates(right_cosine, right_sine, delta_x, delta_y,
                           left, left_dimensions, right, right_dimensions) &&
           !axis_separates(-right_sine, right_cosine, delta_x, delta_y,
                           left, left_dimensions, right, right_dimensions);
}

__device__ bool segment_intersects_local_box(
    Point2 start,
    Point2 end,
    float half_length,
    float half_width)
{
    float t_min = 0.0f;
    float t_max = 1.0f;
    const float delta[2] = {end.x - start.x, end.y - start.y};
    const float origin[2] = {start.x, start.y};
    const float lower[2] = {-half_length, -half_width};
    const float upper[2] = {half_length, half_width};
    for (int axis = 0; axis < 2; ++axis) {
        if (fabsf(delta[axis]) < 1e-8f) {
            if (origin[axis] < lower[axis] || origin[axis] > upper[axis]) {
                return false;
            }
            continue;
        }
        float enter = (lower[axis] - origin[axis]) / delta[axis];
        float exit = (upper[axis] - origin[axis]) / delta[axis];
        if (enter > exit) {
            const float temporary = enter;
            enter = exit;
            exit = temporary;
        }
        t_min = fmaxf(t_min, enter);
        t_max = fminf(t_max, exit);
        if (t_min > t_max) {
            return false;
        }
    }
    return true;
}

__device__ bool point_in_polygon(
    float x,
    float y,
    const Point3 *points,
    int start,
    int count)
{
    bool inside = false;
    for (int current = 0, previous = count - 1; current < count; previous = current++) {
        const Point3 &a = points[start + current];
        const Point3 &b = points[start + previous];
        const bool crosses = ((a.y > y) != (b.y > y)) &&
            (x < (b.x - a.x) * (y - a.y) / ((b.y - a.y) + 1e-12f) + a.x);
        if (crosses) {
            inside = !inside;
        }
    }
    return inside;
}

__global__ void reset_world_kernel(
    int *world_steps,
    int *world_done,
    const std::uint8_t *reset_mask,
    int num_worlds)
{
    const int world = blockIdx.x * blockDim.x + threadIdx.x;
    if (world >= num_worlds || reset_mask[world] == 0) {
        return;
    }
    world_steps[world] = 0;
    world_done[world] = 0;
}

__global__ void reset_agent_kernel(
    const AgentState *initial_states,
    const std::uint8_t *initial_valid,
    AgentState *states,
    VehicleDynamicsState *dynamics_states,
    AgentAction *external_actions,
    AgentAction *applied_actions,
    std::uint8_t *control_modes,
    AgentEvent *events,
    const std::uint8_t *reset_mask,
    int total_agents,
    int max_agents)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_agents || reset_mask[index / max_agents] == 0) {
        return;
    }
    states[index] = initial_states[index];
    const AgentState initial = initial_states[index];
    const float cosine = cosf(initial.yaw);
    const float sine = sinf(initial.yaw);
    dynamics_states[index] = VehicleDynamicsState{
        fmaxf(0.0f, cosine * initial.vx + sine * initial.vy),
        -sine * initial.vx + cosine * initial.vy,
        0.0f,
        0.0f,
        0.0f,
    };
    external_actions[index] = AgentAction{0.0f, 0.0f};
    applied_actions[index] = AgentAction{0.0f, 0.0f};
    control_modes[index] = kControlAuto;
    events[index] = AgentEvent{0, 0, 0, 0, 0};
    if (initial_valid[index] == 0) {
        states[index] = AgentState{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        dynamics_states[index] = VehicleDynamicsState{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }
}

__global__ void sanitize_control_modes_kernel(
    std::uint8_t *control_modes,
    const std::uint8_t *initial_valid,
    const std::uint8_t *controllable,
    int total_agents)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_agents) {
        return;
    }
    const std::uint8_t mode = control_modes[index];
    const bool valid_mode = mode == kControlAuto || mode == kControlResidual ||
                            mode == kControlDirect;
    control_modes[index] = static_cast<std::uint8_t>(
        valid_mode && initial_valid[index] != 0 && controllable[index] != 0
            ? mode
            : kControlAuto);
}

__device__ AgentAction merge_control_action(
    AgentAction automatic_action,
    AgentAction external_action,
    std::uint8_t control_mode,
    float min_acceleration,
    float max_acceleration,
    float max_steering)
{
    AgentAction result = automatic_action;
    if (control_mode == kControlResidual) {
        result.acceleration += external_action.acceleration;
        result.steering += external_action.steering;
    } else if (control_mode == kControlDirect) {
        result = external_action;
    }
    result.acceleration = clamp_value(
        result.acceleration, min_acceleration, max_acceleration);
    result.steering = clamp_value(result.steering, -max_steering, max_steering);
    return result;
}

__global__ void controller_kernel(
    const AgentState *states,
    const std::uint8_t *valid,
    const AgentDimensions *dimensions,
    const Point2 *goals,
    const std::uint8_t *goal_valid,
    const AgentState *reference_future,
    const std::uint8_t *reference_valid,
    const Point3 *map_points,
    const std::int32_t *map_feature_point_start,
    const std::uint8_t *map_feature_valid,
    const std::int32_t *traffic_feature_index,
    const std::uint8_t *traffic_state,
    const std::uint8_t *traffic_valid,
    const int *traffic_counts,
    const int *world_steps,
    const int *world_done,
    const AgentAction *external_actions,
    const std::uint8_t *control_modes,
    AgentAction *applied_actions,
    int total_agents,
    int max_agents,
    int max_future_steps,
    int max_map_features,
    int max_map_points,
    int max_lights,
    int lookahead_steps,
    float min_acceleration,
    float max_acceleration,
    float max_steering,
    float red_stop_distance)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_agents) {
        return;
    }
    const int world = index / max_agents;
    const int agent = index % max_agents;
    if (valid[index] == 0 || world_done[world] != 0) {
        applied_actions[index] = AgentAction{0.0f, 0.0f};
        return;
    }

    const AgentState state = states[index];
    const int step = world_steps[world];
    int target_step = min_int(max_future_steps - 1, step + max_int(0, lookahead_steps - 1));
    bool has_target = false;
    AgentState target{0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int offset = 0; offset <= lookahead_steps && target_step + offset < max_future_steps; ++offset) {
        const int future_index = (world * max_future_steps + target_step + offset) * max_agents + agent;
        if (reference_valid[future_index] != 0) {
            target = reference_future[future_index];
            has_target = true;
            break;
        }
    }
    if (!has_target && goal_valid[index] != 0) {
        target.x = goals[index].x;
        target.y = goals[index].y;
        target.yaw = atan2f(target.y - state.y, target.x - state.x);
        target.vx = 8.0f * cosf(target.yaw);
        target.vy = 8.0f * sinf(target.yaw);
        has_target = true;
    }
    if (!has_target) {
        applied_actions[index] = merge_control_action(
            AgentAction{0.0f, 0.0f},
            external_actions[index],
            control_modes[index],
            min_acceleration,
            max_acceleration,
            max_steering);
        return;
    }

    const float dx = target.x - state.x;
    const float dy = target.y - state.y;
    const float distance = sqrtf(dx * dx + dy * dy);
    const float speed = sqrtf(state.vx * state.vx + state.vy * state.vy);
    float desired_speed = sqrtf(target.vx * target.vx + target.vy * target.vy);
    if (desired_speed < 0.2f && distance > 2.0f) {
        desired_speed = fminf(8.0f, distance);
    }

    const int signal_step = min_int(max_future_steps, max_int(0, step));
    const int signal_base = (world * (max_future_steps + 1) + signal_step) * max_lights;
    const int map_base = world * max_map_points;
    const int feature_base = world * max_map_features;
    for (int light = 0; light < traffic_counts[world]; ++light) {
        if (traffic_valid[signal_base + light] == 0) {
            continue;
        }
        const int signal = traffic_state[signal_base + light];
        if (signal != kTrafficLightStop && signal != kTrafficLightCaution) {
            continue;
        }
        const int feature = traffic_feature_index[world * max_lights + light];
        if (feature < 0 || feature >= max_map_features ||
            map_feature_valid[feature_base + feature] == 0) {
            continue;
        }
        const int point = map_feature_point_start[feature_base + feature];
        if (point < 0 || point >= max_map_points) {
            continue;
        }
        const Point3 stop_point = map_points[map_base + point];
        const Point2 local = global_to_local(state, stop_point.x, stop_point.y);
        if (local.x > 0.0f && local.x < red_stop_distance && fabsf(local.y) < 4.0f) {
            desired_speed = signal == kTrafficLightStop ? 0.0f : fminf(desired_speed, 2.0f);
        }
    }

    const float desired_heading = distance > 0.5f ? atan2f(dy, dx) : target.yaw;
    const float heading_error = wrap_angle(desired_heading - state.yaw);
    const float wheelbase = fmaxf(1.0f, 0.6f * dimensions[index].length);
    const float steering = atan2f(
        2.0f * wheelbase * sinf(heading_error),
        fmaxf(1.0f, distance));
    AgentAction action;
    action.acceleration = clamp_value(2.0f * (desired_speed - speed), min_acceleration, max_acceleration);
    action.steering = clamp_value(steering, -max_steering, max_steering);
    applied_actions[index] = merge_control_action(
        action,
        external_actions[index],
        control_modes[index],
        min_acceleration,
        max_acceleration,
        max_steering);
}

__global__ void dynamics_kernel(
    AgentState *states,
    VehicleDynamicsState *dynamics_states,
    const AgentAction *actions,
    const VehicleParameters *vehicle_parameters,
    const std::uint8_t *valid,
    const float *world_dt,
    const int *world_done,
    int total_agents,
    int max_agents,
    float min_acceleration,
    float max_acceleration,
    float max_steering,
    float max_speed,
    int substeps,
    float acceleration_time_constant,
    float steering_time_constant,
    float max_jerk,
    float max_steering_rate,
    float kinematic_speed_threshold,
    float dynamic_speed_threshold,
    float max_slip_angle,
    float max_yaw_rate)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_agents || valid[index] == 0) {
        return;
    }
    const int world = index / max_agents;
    if (world_done[world] != 0) {
        return;
    }

    AgentState state = states[index];
    VehicleDynamicsState dynamics = dynamics_states[index];
    const VehicleParameters parameters = vehicle_parameters[index];
    AgentAction action = actions[index];
    action.acceleration = clamp_value(action.acceleration, min_acceleration, max_acceleration);
    action.steering = clamp_value(action.steering, -max_steering, max_steering);
    const float dt = world_dt[world] / static_cast<float>(substeps);
    const float wheelbase = parameters.front_axle_distance + parameters.rear_axle_distance;

    for (int substep = 0; substep < substeps; ++substep) {
        const AgentState previous_state = state;
        const VehicleDynamicsState previous_dynamics = dynamics;

        dynamics.longitudinal_acceleration = clamp_value(
            update_actuator(
                dynamics.longitudinal_acceleration,
                action.acceleration,
                acceleration_time_constant,
                max_jerk,
                dt),
            min_acceleration,
            max_acceleration);
        dynamics.steering_angle = clamp_value(
            update_actuator(
                dynamics.steering_angle,
                action.steering,
                steering_time_constant,
                max_steering_rate,
                dt),
            -max_steering,
            max_steering);

        const float body_speed = hypotf(
            dynamics.longitudinal_velocity,
            dynamics.lateral_velocity);
        const float next_kinematic_speed = clamp_value(
            body_speed + dynamics.longitudinal_acceleration * dt,
            0.0f,
            max_speed);
        const float average_kinematic_speed =
            0.5f * (body_speed + next_kinematic_speed);
        const float beta = atanf(
            parameters.rear_axle_distance / wheelbase *
            tanf(dynamics.steering_angle));
        const float kinematic_u = next_kinematic_speed * cosf(beta);
        const float kinematic_v = next_kinematic_speed * sinf(beta);
        const float kinematic_yaw_rate = clamp_value(
            average_kinematic_speed * cosf(beta) *
                tanf(dynamics.steering_angle) / wheelbase,
            -max_yaw_rate,
            max_yaw_rate);
        const float kinematic_yaw = wrap_angle(
            state.yaw + kinematic_yaw_rate * dt);
        const float kinematic_x = state.x +
            average_kinematic_speed * cosf(state.yaw + beta) * dt;
        const float kinematic_y = state.y +
            average_kinematic_speed * sinf(state.yaw + beta) * dt;

        const float u = fmaxf(0.0f, dynamics.longitudinal_velocity);
        const float v = dynamics.lateral_velocity;
        const float yaw_rate = dynamics.yaw_rate;
        const float safe_u = fmaxf(0.5f, u);
        const float front_slip = clamp_value(
            dynamics.steering_angle -
                atan2f(v + parameters.front_axle_distance * yaw_rate, safe_u),
            -max_slip_angle,
            max_slip_angle);
        const float rear_slip = clamp_value(
            -atan2f(v - parameters.rear_axle_distance * yaw_rate, safe_u),
            -max_slip_angle,
            max_slip_angle);

        const float front_normal_force =
            parameters.mass * kGravity * parameters.rear_axle_distance / wheelbase;
        const float rear_normal_force =
            parameters.mass * kGravity * parameters.front_axle_distance / wheelbase;
        const float front_lateral_force = clamp_value(
            parameters.front_cornering_stiffness * front_slip,
            -parameters.tire_friction * front_normal_force,
            parameters.tire_friction * front_normal_force);
        const float rear_lateral_force = clamp_value(
            parameters.rear_cornering_stiffness * rear_slip,
            -parameters.tire_friction * rear_normal_force,
            parameters.tire_friction * rear_normal_force);

        const float steering_cosine = cosf(dynamics.steering_angle);
        const float steering_sine = sinf(dynamics.steering_angle);
        const float longitudinal_derivative =
            dynamics.longitudinal_acceleration + v * yaw_rate -
            front_lateral_force * steering_sine / parameters.mass;
        const float lateral_derivative =
            (front_lateral_force * steering_cosine + rear_lateral_force) /
                parameters.mass -
            u * yaw_rate;
        const float yaw_rate_derivative =
            (parameters.front_axle_distance * front_lateral_force * steering_cosine -
             parameters.rear_axle_distance * rear_lateral_force) /
            parameters.yaw_inertia;

        float dynamic_u = clamp_value(
            u + longitudinal_derivative * dt,
            0.0f,
            max_speed);
        float dynamic_v = clamp_value(
            v + lateral_derivative * dt,
            -max_speed,
            max_speed);
        const float dynamic_speed = hypotf(dynamic_u, dynamic_v);
        if (dynamic_speed > max_speed) {
            const float scale = max_speed / dynamic_speed;
            dynamic_u *= scale;
            dynamic_v *= scale;
        }
        const float dynamic_yaw_rate = clamp_value(
            yaw_rate + yaw_rate_derivative * dt,
            -max_yaw_rate,
            max_yaw_rate);
        const float dynamic_yaw = wrap_angle(
            state.yaw + dynamic_yaw_rate * dt);
        const float dynamic_x = state.x +
            (dynamic_u * cosf(dynamic_yaw) - dynamic_v * sinf(dynamic_yaw)) * dt;
        const float dynamic_y = state.y +
            (dynamic_u * sinf(dynamic_yaw) + dynamic_v * cosf(dynamic_yaw)) * dt;

        const float blend = parameters.use_dynamic_model != 0
            ? smoothstep_value(
                  kinematic_speed_threshold,
                  dynamic_speed_threshold,
                  body_speed)
            : 0.0f;
        state.x = kinematic_x + blend * (dynamic_x - kinematic_x);
        state.y = kinematic_y + blend * (dynamic_y - kinematic_y);
        state.yaw = wrap_angle(
            kinematic_yaw + blend * wrap_angle(dynamic_yaw - kinematic_yaw));
        dynamics.longitudinal_velocity =
            kinematic_u + blend * (dynamic_u - kinematic_u);
        dynamics.lateral_velocity =
            kinematic_v + blend * (dynamic_v - kinematic_v);
        dynamics.yaw_rate =
            kinematic_yaw_rate + blend * (dynamic_yaw_rate - kinematic_yaw_rate);
        state.vx = dynamics.longitudinal_velocity * cosf(state.yaw) -
                   dynamics.lateral_velocity * sinf(state.yaw);
        state.vy = dynamics.longitudinal_velocity * sinf(state.yaw) +
                   dynamics.lateral_velocity * cosf(state.yaw);

        if (!isfinite(state.x) || !isfinite(state.y) || !isfinite(state.yaw) ||
            !isfinite(state.vx) || !isfinite(state.vy) ||
            !isfinite(dynamics.longitudinal_velocity) ||
            !isfinite(dynamics.lateral_velocity) || !isfinite(dynamics.yaw_rate)) {
            state = previous_state;
            dynamics = previous_dynamics;
            break;
        }
    }

    states[index] = state;
    dynamics_states[index] = dynamics;
}

__global__ void collision_kernel(
    const AgentState *states,
    const AgentDimensions *dimensions,
    const std::uint8_t *valid,
    const std::uint8_t *is_ego,
    AgentEvent *events,
    int num_worlds,
    int max_agents)
{
    const int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pairs_per_world = max_agents * max_agents;
    const int total_pairs = num_worlds * pairs_per_world;
    if (pair_index >= total_pairs) {
        return;
    }
    const int world = pair_index / pairs_per_world;
    const int local = pair_index % pairs_per_world;
    const int left_agent = local / max_agents;
    const int right_agent = local % max_agents;
    if (left_agent >= right_agent) {
        return;
    }
    const int left = world * max_agents + left_agent;
    const int right = world * max_agents + right_agent;
    if (valid[left] == 0 || valid[right] == 0) {
        return;
    }
    if (obb_overlap(states[left], dimensions[left], states[right], dimensions[right])) {
        atomicExch(&events[left].collided_vehicle, 1);
        atomicExch(&events[right].collided_vehicle, 1);
        if (is_ego[left] != 0 || is_ego[right] != 0) {
            atomicExch(&events[left].collided_ego, 1);
            atomicExch(&events[right].collided_ego, 1);
        }
    }
}

__global__ void road_and_goal_kernel(
    const AgentState *states,
    const std::uint8_t *valid,
    const AgentDimensions *dimensions,
    const Point2 *goals,
    const std::uint8_t *goal_valid,
    const Point3 *map_points,
    const std::int32_t *feature_type,
    const std::int32_t *geometry_type,
    const std::int32_t *feature_start,
    const std::int32_t *feature_count,
    const std::uint8_t *feature_valid,
    const int *map_feature_counts,
    AgentEvent *events,
    int total_agents,
    int max_agents,
    int max_map_features,
    int max_map_points,
    float goal_threshold)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_agents || valid[index] == 0) {
        return;
    }
    const int world = index / max_agents;
    const int feature_base = world * max_map_features;
    const int point_base = world * max_map_points;
    const AgentState state = states[index];
    const AgentDimensions size = dimensions[index];
    bool has_drivable_polygon = false;
    bool inside_drivable_polygon = false;

    for (int feature = 0; feature < map_feature_counts[world]; ++feature) {
        const int feature_index = feature_base + feature;
        if (feature_valid[feature_index] == 0) {
            continue;
        }
        const int start = feature_start[feature_index];
        const int count = feature_count[feature_index];
        if (start < 0 || count <= 0 || start + count > max_map_points) {
            continue;
        }
        const int type = feature_type[feature_index];
        if ((type == kMapTypeRoadblock || type == kMapTypeRoadblockConnector) &&
            geometry_type[feature_index] == kGeometryPolygon && count >= 3) {
            has_drivable_polygon = true;
            if (point_in_polygon(state.x, state.y, map_points + point_base, start, count)) {
                inside_drivable_polygon = true;
            }
        }
        if (type != kMapTypeRoadEdge || count < 2) {
            continue;
        }
        for (int point = 0; point + 1 < count; ++point) {
            const Point3 a = map_points[point_base + start + point];
            const Point3 b = map_points[point_base + start + point + 1];
            const Point2 local_a = global_to_local(state, a.x, a.y);
            const Point2 local_b = global_to_local(state, b.x, b.y);
            if (segment_intersects_local_box(
                    local_a, local_b, 0.5f * size.length, 0.5f * size.width)) {
                events[index].collided_road = 1;
                break;
            }
        }
    }
    events[index].offroad = has_drivable_polygon && !inside_drivable_polygon ? 1 : 0;
    if (goal_valid[index] != 0) {
        const float dx = goals[index].x - state.x;
        const float dy = goals[index].y - state.y;
        events[index].reached_goal = dx * dx + dy * dy <= goal_threshold * goal_threshold ? 1 : 0;
    }
}

__global__ void advance_world_kernel(
    int *world_steps,
    int *world_done,
    const int *episode_steps,
    int num_worlds)
{
    const int world = blockIdx.x * blockDim.x + threadIdx.x;
    if (world >= num_worlds || world_done[world] != 0) {
        return;
    }
    world_steps[world] += 1;
    if (world_steps[world] >= episode_steps[world]) {
        world_done[world] = 1;
    }
}

__global__ void self_observation_kernel(
    const AgentState *states,
    const std::uint8_t *valid,
    const AgentDimensions *dimensions,
    const Point2 *goals,
    const std::uint8_t *goal_valid,
    const int *world_steps,
    const int *episode_steps,
    SelfObservation *observations,
    int total_agents,
    int max_agents)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_agents) {
        return;
    }
    if (valid[index] == 0) {
        observations[index] = SelfObservation{0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0};
        return;
    }
    const int world = index / max_agents;
    const AgentState state = states[index];
    Point2 goal_local{0.0f, 0.0f};
    if (goal_valid[index] != 0) {
        goal_local = global_to_local(state, goals[index].x, goals[index].y);
    }
    observations[index] = SelfObservation{
        state.x,
        state.y,
        state.yaw,
        state.vx,
        state.vy,
        sqrtf(state.vx * state.vx + state.vy * state.vy),
        dimensions[index].length,
        dimensions[index].width,
        goal_local.x,
        goal_local.y,
        static_cast<float>(max_int(0, episode_steps[world] - world_steps[world])),
        1,
    };
}

__global__ void partner_observation_kernel(
    const AgentState *states,
    const std::uint8_t *valid,
    const AgentDimensions *dimensions,
    const std::int32_t *agent_type,
    PartnerObservation *observations,
    int total_agents,
    int max_agents,
    int observation_count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_agents) {
        return;
    }
    PartnerObservation *output = observations + index * observation_count;
    for (int slot = 0; slot < observation_count; ++slot) {
        output[slot] = PartnerObservation{0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0};
    }
    if (valid[index] == 0) {
        return;
    }
    const int world = index / max_agents;
    const int self_agent = index % max_agents;
    const AgentState self = states[index];
    for (int other_agent = 0; other_agent < max_agents; ++other_agent) {
        if (other_agent == self_agent) {
            continue;
        }
        const int other = world * max_agents + other_agent;
        if (valid[other] == 0) {
            continue;
        }
        const AgentState partner = states[other];
        const Point2 relative_position = global_to_local(self, partner.x, partner.y);
        const float cosine = cosf(self.yaw);
        const float sine = sinf(self.yaw);
        const float delta_vx = partner.vx - self.vx;
        const float delta_vy = partner.vy - self.vy;
        PartnerObservation candidate;
        candidate.rel_x = relative_position.x;
        candidate.rel_y = relative_position.y;
        candidate.rel_vx = cosine * delta_vx + sine * delta_vy;
        candidate.rel_vy = -sine * delta_vx + cosine * delta_vy;
        candidate.rel_yaw = wrap_angle(partner.yaw - self.yaw);
        candidate.length = dimensions[other].length;
        candidate.width = dimensions[other].width;
        candidate.type = static_cast<float>(agent_type[other]);
        candidate.distance = sqrtf(
            relative_position.x * relative_position.x + relative_position.y * relative_position.y);
        candidate.valid = 1;

        int insert_at = observation_count;
        for (int slot = 0; slot < observation_count; ++slot) {
            if (output[slot].valid == 0 || candidate.distance < output[slot].distance) {
                insert_at = slot;
                break;
            }
        }
        if (insert_at == observation_count) {
            continue;
        }
        for (int slot = observation_count - 1; slot > insert_at; --slot) {
            output[slot] = output[slot - 1];
        }
        output[insert_at] = candidate;
    }
}

__global__ void map_observation_kernel(
    const AgentState *states,
    const std::uint8_t *agent_valid,
    const Point3 *map_points,
    const std::int32_t *feature_type,
    const std::int32_t *feature_start,
    const std::int32_t *feature_count,
    const std::uint8_t *feature_valid,
    const float *speed_limit,
    const std::uint8_t *speed_limit_valid,
    const int *map_feature_counts,
    MapObservation *observations,
    int total_agents,
    int max_agents,
    int max_map_features,
    int max_map_points,
    int observation_count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_agents) {
        return;
    }
    MapObservation *output = observations + index * observation_count;
    for (int slot = 0; slot < observation_count; ++slot) {
        output[slot] = MapObservation{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0};
    }
    if (agent_valid[index] == 0) {
        return;
    }
    const int world = index / max_agents;
    const int feature_base = world * max_map_features;
    const int point_base = world * max_map_points;
    const AgentState state = states[index];
    for (int feature = 0; feature < map_feature_counts[world]; ++feature) {
        const int feature_index = feature_base + feature;
        if (feature_valid[feature_index] == 0) {
            continue;
        }
        const int start = feature_start[feature_index];
        const int count = feature_count[feature_index];
        if (start < 0 || count < 2 || start + count > max_map_points) {
            continue;
        }
        for (int point = 0; point + 1 < count; ++point) {
            const Point3 a = map_points[point_base + start + point];
            const Point3 b = map_points[point_base + start + point + 1];
            MapObservation candidate;
            const Point2 local_a = global_to_local(state, a.x, a.y);
            const Point2 local_b = global_to_local(state, b.x, b.y);
            candidate.start_x = local_a.x;
            candidate.start_y = local_a.y;
            candidate.end_x = local_b.x;
            candidate.end_y = local_b.y;
            candidate.type = static_cast<float>(feature_type[feature_index]);
            candidate.speed_limit = speed_limit_valid[feature_index] != 0
                ? speed_limit[feature_index]
                : 0.0f;
            candidate.distance = sqrtf(point_segment_distance_squared(
                0.0f, 0.0f, local_a.x, local_a.y, local_b.x, local_b.y));
            candidate.valid = 1;

            int insert_at = observation_count;
            for (int slot = 0; slot < observation_count; ++slot) {
                if (output[slot].valid == 0 || candidate.distance < output[slot].distance) {
                    insert_at = slot;
                    break;
                }
            }
            if (insert_at == observation_count) {
                continue;
            }
            for (int slot = observation_count - 1; slot > insert_at; --slot) {
                output[slot] = output[slot - 1];
            }
            output[insert_at] = candidate;
        }
    }
}

__global__ void signal_observation_kernel(
    const std::int32_t *feature_index,
    const std::uint8_t *signal_state,
    const std::uint8_t *signal_valid,
    const int *world_steps,
    const int *traffic_counts,
    SignalObservation *observations,
    int num_worlds,
    int max_future_steps,
    int max_lights)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_worlds * max_lights;
    if (index >= total) {
        return;
    }
    const int world = index / max_lights;
    const int light = index % max_lights;
    if (light >= traffic_counts[world]) {
        observations[index] = SignalObservation{-1, 0, 0};
        return;
    }
    const int step = min_int(max_future_steps, max_int(0, world_steps[world]));
    const int schedule_index = (world * (max_future_steps + 1) + step) * max_lights + light;
    observations[index] = SignalObservation{
        feature_index[index],
        static_cast<int>(signal_state[schedule_index]),
        static_cast<int>(signal_valid[schedule_index]),
    };
}

}  // namespace

DriveSim::DriveSim(SimConfig config, std::vector<RuntimeScene> scenes)
    : config_(config),
      capacities_(scenes.empty() ? RuntimeCapacities{} : scenes.front().capacities),
      num_worlds_(static_cast<int>(scenes.size())),
      total_agents_(num_worlds_ * capacities_.max_agents)
{
    if (scenes.empty()) {
        throw std::runtime_error("DriveSim requires at least one runtime scene");
    }
    if (config_.partner_observations <= 0 ||
        config_.partner_observations > kMaximumPartnerObservations) {
        throw std::runtime_error("partner_observations must be in [1, 16]");
    }
    if (config_.map_observations <= 0 || config_.map_observations > kMaximumMapObservations) {
        throw std::runtime_error("map_observations must be in [1, 64]");
    }
    if (config_.tracker_lookahead_steps <= 0 ||
        config_.min_acceleration >= config_.max_acceleration ||
        config_.max_abs_steering <= 0.0f || config_.max_speed <= 0.0f ||
        config_.goal_threshold <= 0.0f || config_.red_light_stop_distance <= 0.0f) {
        throw std::runtime_error("invalid simulator configuration");
    }
    if (config_.dynamics_substeps <= 0 || config_.dynamics_substeps > 64 ||
        config_.acceleration_time_constant <= 0.0f ||
        config_.steering_time_constant <= 0.0f || config_.max_jerk <= 0.0f ||
        config_.max_steering_rate <= 0.0f ||
        config_.kinematic_speed_threshold < 0.0f ||
        config_.dynamic_speed_threshold <= config_.kinematic_speed_threshold ||
        config_.max_abs_slip_angle <= 0.0f || config_.max_abs_yaw_rate <= 0.0f ||
        config_.passenger_max_length <= 0.0f || config_.passenger_max_width <= 0.0f ||
        !valid_dynamics_preset(config_.passenger_vehicle) ||
        !valid_dynamics_preset(config_.large_vehicle)) {
        throw std::runtime_error("invalid vehicle dynamics configuration");
    }
    for (const RuntimeScene &scene : scenes) {
        if (!(scene.capacities == capacities_)) {
            throw std::runtime_error("all DriveSim scenes must use identical capacities");
        }
    }
    try {
        allocate_and_upload(scenes);
        reset();
    } catch (...) {
        release();
        throw;
    }
}

DriveSim::~DriveSim()
{
    release();
}

void DriveSim::allocate_and_upload(const std::vector<RuntimeScene> &scenes)
{
    std::vector<AgentState> initial_states;
    std::vector<std::uint8_t> initial_valid;
    std::vector<std::int32_t> agent_type;
    std::vector<std::uint8_t> agent_is_ego;
    std::vector<std::uint8_t> agent_controllable;
    std::vector<AgentDimensions> dimensions;
    std::vector<VehicleParameters> vehicle_parameters;
    std::vector<Point2> goals;
    std::vector<std::uint8_t> goal_valid;
    std::vector<AgentState> future;
    std::vector<std::uint8_t> future_valid;
    std::vector<Point3> map_points;
    std::vector<std::int32_t> map_feature_type;
    std::vector<std::int32_t> map_geometry_type;
    std::vector<std::int32_t> map_feature_start;
    std::vector<std::int32_t> map_feature_count;
    std::vector<std::uint8_t> map_feature_valid;
    std::vector<float> map_speed_limit;
    std::vector<std::uint8_t> map_speed_limit_valid;
    std::vector<std::int32_t> traffic_feature_index;
    std::vector<std::uint8_t> traffic_state;
    std::vector<std::uint8_t> traffic_valid;
    std::vector<float> world_dt;
    std::vector<int> episode_steps;
    std::vector<int> map_feature_counts;
    std::vector<int> traffic_counts;

    for (const RuntimeScene &scene : scenes) {
        const std::vector<AgentDimensions> scene_dimensions =
            packed_structs<AgentDimensions>(scene.agent_dimensions, 3);
        append_values(initial_states, packed_structs<AgentState>(scene.agent_initial_state, 5));
        append_values(initial_valid, scene.agent_initial_valid);
        append_values(agent_type, scene.agent_type);
        append_values(agent_is_ego, scene.agent_is_ego);
        append_values(agent_controllable, scene.agent_controllable);
        append_values(dimensions, scene_dimensions);
        for (int agent = 0; agent < scene.capacities.max_agents; ++agent) {
            vehicle_parameters.push_back(make_vehicle_parameters(
                scene.agent_type[static_cast<std::size_t>(agent)],
                scene_dimensions[static_cast<std::size_t>(agent)],
                config_));
        }
        append_values(goals, packed_structs<Point2>(scene.agent_goal, 2));
        append_values(goal_valid, scene.agent_goal_valid);
        append_values(future, packed_structs<AgentState>(scene.reference_future, 5));
        append_values(future_valid, scene.reference_future_valid);
        append_values(map_points, packed_structs<Point3>(scene.map_points, 3));
        append_values(map_feature_type, scene.map_feature_type);
        append_values(map_geometry_type, scene.map_geometry_type);
        append_values(map_feature_start, scene.map_feature_point_start);
        append_values(map_feature_count, scene.map_feature_point_count);
        append_values(map_feature_valid, scene.map_feature_valid);
        append_values(map_speed_limit, scene.map_speed_limit);
        append_values(map_speed_limit_valid, scene.map_speed_limit_valid);
        append_values(traffic_feature_index, scene.traffic_light_feature_index);
        append_values(traffic_state, scene.traffic_light_state);
        append_values(traffic_valid, scene.traffic_light_valid);
        world_dt.push_back(scene.dt);
        episode_steps.push_back(scene.episode_steps);
        map_feature_counts.push_back(scene.counts.map_features);
        traffic_counts.push_back(scene.counts.traffic_lights);
    }

    upload_device(&d_initial_states_, initial_states);
    upload_device(&d_initial_valid_, initial_valid);
    upload_device(&d_agent_type_, agent_type);
    upload_device(&d_agent_is_ego_, agent_is_ego);
    upload_device(&d_agent_controllable_, agent_controllable);
    upload_device(&d_agent_dimensions_, dimensions);
    upload_device(&d_vehicle_parameters_, vehicle_parameters);
    upload_device(&d_agent_goals_, goals);
    upload_device(&d_agent_goal_valid_, goal_valid);
    upload_device(&d_reference_future_, future);
    upload_device(&d_reference_future_valid_, future_valid);
    upload_device(&d_map_points_, map_points);
    upload_device(&d_map_feature_type_, map_feature_type);
    upload_device(&d_map_geometry_type_, map_geometry_type);
    upload_device(&d_map_feature_point_start_, map_feature_start);
    upload_device(&d_map_feature_point_count_, map_feature_count);
    upload_device(&d_map_feature_valid_, map_feature_valid);
    upload_device(&d_map_speed_limit_, map_speed_limit);
    upload_device(&d_map_speed_limit_valid_, map_speed_limit_valid);
    upload_device(&d_traffic_light_feature_index_, traffic_feature_index);
    upload_device(&d_traffic_light_state_, traffic_state);
    upload_device(&d_traffic_light_valid_, traffic_valid);
    upload_device(&d_world_dt_, world_dt);
    upload_device(&d_episode_steps_, episode_steps);
    upload_device(&d_map_feature_counts_, map_feature_counts);
    upload_device(&d_traffic_light_counts_, traffic_counts);

    allocate_device(&d_states_, static_cast<std::size_t>(total_agents_));
    allocate_device(&d_dynamics_states_, static_cast<std::size_t>(total_agents_));
    allocate_device(&d_external_actions_, static_cast<std::size_t>(total_agents_));
    allocate_device(&d_applied_actions_, static_cast<std::size_t>(total_agents_));
    allocate_device(&d_external_control_, static_cast<std::size_t>(total_agents_));
    allocate_device(&d_events_, static_cast<std::size_t>(total_agents_));
    allocate_device(&d_self_observations_, static_cast<std::size_t>(total_agents_));
    allocate_device(
        &d_partner_observations_,
        static_cast<std::size_t>(total_agents_) * config_.partner_observations);
    allocate_device(
        &d_map_observations_,
        static_cast<std::size_t>(total_agents_) * config_.map_observations);
    allocate_device(
        &d_signal_observations_,
        static_cast<std::size_t>(num_worlds_) * capacities_.max_traffic_lights);
    allocate_device(&d_world_steps_, static_cast<std::size_t>(num_worlds_));
    allocate_device(&d_world_done_, static_cast<std::size_t>(num_worlds_));
    allocate_device(&d_world_reset_mask_, static_cast<std::size_t>(num_worlds_));
}

void DriveSim::release() noexcept
{
#define CUDA_FREE(pointer) do { if ((pointer) != nullptr) { cudaFree(pointer); (pointer) = nullptr; } } while (0)
    CUDA_FREE(d_initial_states_);
    CUDA_FREE(d_initial_valid_);
    CUDA_FREE(d_agent_type_);
    CUDA_FREE(d_agent_is_ego_);
    CUDA_FREE(d_agent_controllable_);
    CUDA_FREE(d_agent_dimensions_);
    CUDA_FREE(d_vehicle_parameters_);
    CUDA_FREE(d_agent_goals_);
    CUDA_FREE(d_agent_goal_valid_);
    CUDA_FREE(d_reference_future_);
    CUDA_FREE(d_reference_future_valid_);
    CUDA_FREE(d_map_points_);
    CUDA_FREE(d_map_feature_type_);
    CUDA_FREE(d_map_geometry_type_);
    CUDA_FREE(d_map_feature_point_start_);
    CUDA_FREE(d_map_feature_point_count_);
    CUDA_FREE(d_map_feature_valid_);
    CUDA_FREE(d_map_speed_limit_);
    CUDA_FREE(d_map_speed_limit_valid_);
    CUDA_FREE(d_traffic_light_feature_index_);
    CUDA_FREE(d_traffic_light_state_);
    CUDA_FREE(d_traffic_light_valid_);
    CUDA_FREE(d_world_dt_);
    CUDA_FREE(d_episode_steps_);
    CUDA_FREE(d_map_feature_counts_);
    CUDA_FREE(d_traffic_light_counts_);
    CUDA_FREE(d_states_);
    CUDA_FREE(d_dynamics_states_);
    CUDA_FREE(d_external_actions_);
    CUDA_FREE(d_applied_actions_);
    CUDA_FREE(d_external_control_);
    CUDA_FREE(d_events_);
    CUDA_FREE(d_self_observations_);
    CUDA_FREE(d_partner_observations_);
    CUDA_FREE(d_map_observations_);
    CUDA_FREE(d_signal_observations_);
    CUDA_FREE(d_world_steps_);
    CUDA_FREE(d_world_done_);
    CUDA_FREE(d_world_reset_mask_);
#undef CUDA_FREE
}

void DriveSim::reset()
{
    std::vector<int> worlds(static_cast<std::size_t>(num_worlds_));
    for (int world = 0; world < num_worlds_; ++world) {
        worlds[static_cast<std::size_t>(world)] = world;
    }
    reset_worlds(worlds);
}

void DriveSim::reset_worlds(const std::vector<int> &world_ids)
{
    std::vector<std::uint8_t> reset_mask(static_cast<std::size_t>(num_worlds_), 0);
    for (int world : world_ids) {
        if (world < 0 || world >= num_worlds_) {
            throw std::runtime_error("reset_worlds received an invalid world index");
        }
        reset_mask[static_cast<std::size_t>(world)] = 1;
    }
    CUDA_CHECK(cudaMemcpy(
        d_world_reset_mask_,
        reset_mask.data(),
        reset_mask.size() * sizeof(std::uint8_t),
        cudaMemcpyHostToDevice));
    reset_worlds_impl(d_world_reset_mask_, 0);
    CUDA_CHECK(cudaStreamSynchronize(0));
}

void DriveSim::reset_worlds_device(const std::uint8_t *reset_mask, cudaStream_t stream)
{
    if (reset_mask == nullptr) {
        throw std::runtime_error("reset_worlds_device received a null reset mask");
    }
    reset_worlds_impl(reset_mask, stream);
}

void DriveSim::reset_worlds_impl(const std::uint8_t *reset_mask, cudaStream_t stream)
{
    constexpr int threads = 128;
    reset_world_kernel<<<(num_worlds_ + threads - 1) / threads, threads, 0, stream>>>(
        d_world_steps_, d_world_done_, reset_mask, num_worlds_);
    check_kernel_launch();
    reset_agent_kernel<<<(total_agents_ + threads - 1) / threads, threads, 0, stream>>>(
        d_initial_states_,
        d_initial_valid_,
        d_states_,
        d_dynamics_states_,
        d_external_actions_,
        d_applied_actions_,
        d_external_control_,
        d_events_,
        reset_mask,
        total_agents_,
        capacities_.max_agents);
    check_kernel_launch();
    build_observations(stream);
}

void DriveSim::set_external_control_mask(const std::vector<std::uint8_t> &mask)
{
    if (mask.size() != static_cast<std::size_t>(total_agents_)) {
        throw std::runtime_error("external control mask size does not match [worlds, agents]");
    }
    std::vector<std::uint8_t> modes(mask.size(), kControlAuto);
    for (std::size_t index = 0; index < mask.size(); ++index) {
        modes[index] = mask[index] != 0 ? kControlDirect : kControlAuto;
    }
    set_control_modes(modes);
}

void DriveSim::set_control_modes(const std::vector<std::uint8_t> &modes)
{
    if (modes.size() != static_cast<std::size_t>(total_agents_)) {
        throw std::runtime_error("control mode size does not match [worlds, agents]");
    }
    CUDA_CHECK(cudaMemcpy(
        d_external_control_,
        modes.data(),
        modes.size() * sizeof(std::uint8_t),
        cudaMemcpyHostToDevice));
    constexpr int threads = 128;
    sanitize_control_modes_kernel<<<(total_agents_ + threads - 1) / threads, threads>>>(
        d_external_control_, d_initial_valid_, d_agent_controllable_, total_agents_);
    check_kernel_launch();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void DriveSim::set_control_modes_device(const std::uint8_t *modes, cudaStream_t stream)
{
    if (modes == nullptr) {
        throw std::runtime_error("set_control_modes_device received a null mode tensor");
    }
    CUDA_CHECK(cudaMemcpyAsync(
        d_external_control_,
        modes,
        static_cast<std::size_t>(total_agents_) * sizeof(std::uint8_t),
        cudaMemcpyDeviceToDevice,
        stream));
    constexpr int threads = 128;
    sanitize_control_modes_kernel<<<(total_agents_ + threads - 1) / threads, threads, 0, stream>>>(
        d_external_control_, d_initial_valid_, d_agent_controllable_, total_agents_);
    check_kernel_launch();
}

void DriveSim::set_actions(const std::vector<AgentAction> &actions)
{
    if (actions.size() != static_cast<std::size_t>(total_agents_)) {
        throw std::runtime_error("action size does not match [worlds, agents]");
    }
    CUDA_CHECK(cudaMemcpy(
        d_external_actions_,
        actions.data(),
        actions.size() * sizeof(AgentAction),
        cudaMemcpyHostToDevice));
}

void DriveSim::step()
{
    step_impl(d_external_actions_, 0);
    CUDA_CHECK(cudaStreamSynchronize(0));
}

void DriveSim::step_device(const AgentAction *actions, cudaStream_t stream)
{
    if (actions == nullptr) {
        throw std::runtime_error("step_device received a null action tensor");
    }
    step_impl(actions, stream);
}

void DriveSim::step_impl(const AgentAction *actions, cudaStream_t stream)
{
    constexpr int threads = 128;
    CUDA_CHECK(cudaMemsetAsync(
        d_events_,
        0,
        static_cast<std::size_t>(total_agents_) * sizeof(AgentEvent),
        stream));
    controller_kernel<<<(total_agents_ + threads - 1) / threads, threads, 0, stream>>>(
        d_states_,
        d_initial_valid_,
        d_agent_dimensions_,
        d_agent_goals_,
        d_agent_goal_valid_,
        d_reference_future_,
        d_reference_future_valid_,
        d_map_points_,
        d_map_feature_point_start_,
        d_map_feature_valid_,
        d_traffic_light_feature_index_,
        d_traffic_light_state_,
        d_traffic_light_valid_,
        d_traffic_light_counts_,
        d_world_steps_,
        d_world_done_,
        actions,
        d_external_control_,
        d_applied_actions_,
        total_agents_,
        capacities_.max_agents,
        capacities_.max_future_steps,
        capacities_.max_map_features,
        capacities_.max_map_points,
        capacities_.max_traffic_lights,
        config_.tracker_lookahead_steps,
        config_.min_acceleration,
        config_.max_acceleration,
        config_.max_abs_steering,
        config_.red_light_stop_distance);
    check_kernel_launch();
    dynamics_kernel<<<(total_agents_ + threads - 1) / threads, threads, 0, stream>>>(
        d_states_,
        d_dynamics_states_,
        d_applied_actions_,
        d_vehicle_parameters_,
        d_initial_valid_,
        d_world_dt_,
        d_world_done_,
        total_agents_,
        capacities_.max_agents,
        config_.min_acceleration,
        config_.max_acceleration,
        config_.max_abs_steering,
        config_.max_speed,
        config_.dynamics_substeps,
        config_.acceleration_time_constant,
        config_.steering_time_constant,
        config_.max_jerk,
        config_.max_steering_rate,
        config_.kinematic_speed_threshold,
        config_.dynamic_speed_threshold,
        config_.max_abs_slip_angle,
        config_.max_abs_yaw_rate);
    check_kernel_launch();

    const int total_pairs = num_worlds_ * capacities_.max_agents * capacities_.max_agents;
    collision_kernel<<<(total_pairs + threads - 1) / threads, threads, 0, stream>>>(
        d_states_,
        d_agent_dimensions_,
        d_initial_valid_,
        d_agent_is_ego_,
        d_events_,
        num_worlds_,
        capacities_.max_agents);
    check_kernel_launch();
    road_and_goal_kernel<<<(total_agents_ + threads - 1) / threads, threads, 0, stream>>>(
        d_states_,
        d_initial_valid_,
        d_agent_dimensions_,
        d_agent_goals_,
        d_agent_goal_valid_,
        d_map_points_,
        d_map_feature_type_,
        d_map_geometry_type_,
        d_map_feature_point_start_,
        d_map_feature_point_count_,
        d_map_feature_valid_,
        d_map_feature_counts_,
        d_events_,
        total_agents_,
        capacities_.max_agents,
        capacities_.max_map_features,
        capacities_.max_map_points,
        config_.goal_threshold);
    check_kernel_launch();
    advance_world_kernel<<<(num_worlds_ + threads - 1) / threads, threads, 0, stream>>>(
        d_world_steps_, d_world_done_, d_episode_steps_, num_worlds_);
    check_kernel_launch();
    build_observations(stream);
}

void DriveSim::build_observations(cudaStream_t stream)
{
    constexpr int threads = 128;
    self_observation_kernel<<<(total_agents_ + threads - 1) / threads, threads, 0, stream>>>(
        d_states_,
        d_initial_valid_,
        d_agent_dimensions_,
        d_agent_goals_,
        d_agent_goal_valid_,
        d_world_steps_,
        d_episode_steps_,
        d_self_observations_,
        total_agents_,
        capacities_.max_agents);
    check_kernel_launch();
    partner_observation_kernel<<<(total_agents_ + threads - 1) / threads, threads, 0, stream>>>(
        d_states_,
        d_initial_valid_,
        d_agent_dimensions_,
        d_agent_type_,
        d_partner_observations_,
        total_agents_,
        capacities_.max_agents,
        config_.partner_observations);
    check_kernel_launch();
    map_observation_kernel<<<(total_agents_ + threads - 1) / threads, threads, 0, stream>>>(
        d_states_,
        d_initial_valid_,
        d_map_points_,
        d_map_feature_type_,
        d_map_feature_point_start_,
        d_map_feature_point_count_,
        d_map_feature_valid_,
        d_map_speed_limit_,
        d_map_speed_limit_valid_,
        d_map_feature_counts_,
        d_map_observations_,
        total_agents_,
        capacities_.max_agents,
        capacities_.max_map_features,
        capacities_.max_map_points,
        config_.map_observations);
    check_kernel_launch();
    const int total_signals = num_worlds_ * capacities_.max_traffic_lights;
    signal_observation_kernel<<<(total_signals + threads - 1) / threads, threads, 0, stream>>>(
        d_traffic_light_feature_index_,
        d_traffic_light_state_,
        d_traffic_light_valid_,
        d_world_steps_,
        d_traffic_light_counts_,
        d_signal_observations_,
        num_worlds_,
        capacities_.max_future_steps,
        capacities_.max_traffic_lights);
    check_kernel_launch();
}

SimSnapshot DriveSim::copy_snapshot() const
{
    CUDA_CHECK(cudaDeviceSynchronize());
    SimSnapshot snapshot;
    snapshot.states = download_device(d_states_, static_cast<std::size_t>(total_agents_));
    snapshot.applied_actions = download_device(
        d_applied_actions_, static_cast<std::size_t>(total_agents_));
    snapshot.dynamics_states = download_device(
        d_dynamics_states_, static_cast<std::size_t>(total_agents_));
    snapshot.events = download_device(d_events_, static_cast<std::size_t>(total_agents_));
    snapshot.valid = download_device(d_initial_valid_, static_cast<std::size_t>(total_agents_));
    snapshot.control_modes = download_device(
        d_external_control_, static_cast<std::size_t>(total_agents_));
    snapshot.external_control.resize(snapshot.control_modes.size(), 0);
    for (std::size_t index = 0; index < snapshot.control_modes.size(); ++index) {
        snapshot.external_control[index] = static_cast<std::uint8_t>(
            snapshot.control_modes[index] != kControlAuto);
    }
    snapshot.world_steps = download_device(d_world_steps_, static_cast<std::size_t>(num_worlds_));
    snapshot.world_done = download_device(d_world_done_, static_cast<std::size_t>(num_worlds_));
    return snapshot;
}

std::vector<SelfObservation> DriveSim::copy_self_observations() const
{
    CUDA_CHECK(cudaDeviceSynchronize());
    return download_device(d_self_observations_, static_cast<std::size_t>(total_agents_));
}

std::vector<PartnerObservation> DriveSim::copy_partner_observations() const
{
    CUDA_CHECK(cudaDeviceSynchronize());
    return download_device(
        d_partner_observations_,
        static_cast<std::size_t>(total_agents_) * config_.partner_observations);
}

std::vector<MapObservation> DriveSim::copy_map_observations() const
{
    CUDA_CHECK(cudaDeviceSynchronize());
    return download_device(
        d_map_observations_,
        static_cast<std::size_t>(total_agents_) * config_.map_observations);
}

std::vector<SignalObservation> DriveSim::copy_signal_observations() const
{
    CUDA_CHECK(cudaDeviceSynchronize());
    return download_device(
        d_signal_observations_,
        static_cast<std::size_t>(num_worlds_) * capacities_.max_traffic_lights);
}

SimDeviceView DriveSim::device_view() const
{
    return SimDeviceView{
        d_states_,
        d_dynamics_states_,
        d_applied_actions_,
        d_events_,
        d_self_observations_,
        d_partner_observations_,
        d_map_observations_,
        d_initial_valid_,
        d_agent_type_,
        d_agent_is_ego_,
        d_agent_controllable_,
        d_external_control_,
        d_world_steps_,
        d_world_done_,
    };
}
