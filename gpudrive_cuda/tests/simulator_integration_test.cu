#include "drive_sim.cuh"
#include "runtime_scene.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool finite_state(const AgentState &state)
{
    return std::isfinite(state.x) && std::isfinite(state.y) && std::isfinite(state.yaw) &&
           std::isfinite(state.vx) && std::isfinite(state.vy);
}

bool finite_dynamics(const VehicleDynamicsState &state)
{
    return std::isfinite(state.longitudinal_velocity) &&
           std::isfinite(state.lateral_velocity) && std::isfinite(state.yaw_rate) &&
           std::isfinite(state.steering_angle) &&
           std::isfinite(state.longitudinal_acceleration);
}

void set_state(RuntimeScene &scene, int agent, float x, float y, float yaw, float vx, float vy)
{
    const int offset = agent * 5;
    scene.agent_initial_state[static_cast<std::size_t>(offset)] = x;
    scene.agent_initial_state[static_cast<std::size_t>(offset + 1)] = y;
    scene.agent_initial_state[static_cast<std::size_t>(offset + 2)] = yaw;
    scene.agent_initial_state[static_cast<std::size_t>(offset + 3)] = vx;
    scene.agent_initial_state[static_cast<std::size_t>(offset + 4)] = vy;
}

void set_map_point(RuntimeScene &scene, int index, float x, float y)
{
    scene.map_points[static_cast<std::size_t>(index * 3)] = x;
    scene.map_points[static_cast<std::size_t>(index * 3 + 1)] = y;
    scene.map_points[static_cast<std::size_t>(index * 3 + 2)] = 0.0f;
    scene.map_point_valid[static_cast<std::size_t>(index)] = 1;
}

RuntimeScene isolated_agent_scene(
    const RuntimeScene &source,
    int agent_type,
    float length,
    float width,
    float speed)
{
    RuntimeScene scene = source;
    scene.counts.agents = 1;
    scene.episode_steps = 100;
    std::fill(scene.agent_initial_valid.begin(), scene.agent_initial_valid.end(), 0);
    std::fill(scene.agent_controllable.begin(), scene.agent_controllable.end(), 0);
    std::fill(scene.agent_is_ego.begin(), scene.agent_is_ego.end(), 0);
    std::fill(scene.agent_goal_valid.begin(), scene.agent_goal_valid.end(), 0);
    std::fill(scene.reference_future_valid.begin(), scene.reference_future_valid.end(), 0);
    scene.agent_initial_valid[0] = 1;
    scene.agent_controllable[0] = 1;
    scene.agent_is_ego[0] = 1;
    scene.agent_type[0] = agent_type;
    scene.agent_dimensions[0] = length;
    scene.agent_dimensions[1] = width;
    scene.agent_dimensions[2] = 1.5f;
    set_state(scene, 0, 0.0f, 0.0f, 0.0f, speed, 0.0f);
    return scene;
}

void test_rollout_and_reset(const RuntimeScene &source)
{
    DriveSim simulator(SimConfig{}, {source, source});
    const SimSnapshot initial = simulator.copy_snapshot();
    require(initial.world_steps[0] == 0 && initial.world_steps[1] == 0, "reset step is not zero");
    require(initial.valid[0] == 1, "ego is invalid after reset");
    require(initial.dynamics_states.size() == initial.states.size(),
            "dynamics snapshot shape is wrong");
    require(finite_dynamics(initial.dynamics_states[0]),
            "initial dynamics state is not finite");

    simulator.step();
    const SimSnapshot stepped = simulator.copy_snapshot();
    require(stepped.world_steps[0] == 1 && stepped.world_steps[1] == 1, "world did not advance");
    require(finite_state(stepped.states[0]), "ego state is not finite");
    const float displacement = std::hypot(
        stepped.states[0].x - initial.states[0].x,
        stepped.states[0].y - initial.states[0].y);
    require(displacement > 0.01f, "automatic controller did not move ego");

    const int padding = source.counts.agents;
    if (padding < source.capacities.max_agents) {
        require(stepped.valid[static_cast<std::size_t>(padding)] == 0, "padding slot became valid");
        require(stepped.states[static_cast<std::size_t>(padding)].x == 0.0f, "padding slot moved");
    }

    simulator.reset_worlds({0});
    const SimSnapshot reset = simulator.copy_snapshot();
    require(reset.world_steps[0] == 0 && reset.world_steps[1] == 1, "selective reset changed wrong worlds");
    require(std::fabs(reset.states[0].x - initial.states[0].x) < 1e-6f, "reset did not restore state");
    require(std::fabs(reset.dynamics_states[0].steering_angle) < 1e-6f &&
                std::fabs(reset.dynamics_states[0].longitudinal_acceleration) < 1e-6f,
            "reset did not restore actuator state");
    const int world_one_ego = source.capacities.max_agents;
    require(
        std::fabs(reset.states[static_cast<std::size_t>(world_one_ego)].x -
                  stepped.states[static_cast<std::size_t>(world_one_ego)].x) < 1e-6f,
        "selective reset modified another world");

    std::vector<std::uint8_t> mask(
        static_cast<std::size_t>(2 * source.capacities.max_agents), 0);
    std::vector<AgentAction> actions(mask.size(), AgentAction{0.0f, 0.0f});
    mask[0] = 1;
    actions[0] = AgentAction{-6.0f, 0.2f};
    simulator.set_external_control_mask(mask);
    simulator.set_actions(actions);
    simulator.step();
    const SimSnapshot external = simulator.copy_snapshot();
    require(external.external_control[0] == 1, "external control mask was not applied");
    require(std::fabs(external.applied_actions[0].acceleration + 6.0f) < 1e-6f,
            "external acceleration was not applied");
    require(std::fabs(external.applied_actions[0].steering - 0.2f) < 1e-6f,
            "external steering was not applied");

    require(
        simulator.copy_self_observations().size() ==
            static_cast<std::size_t>(2 * source.capacities.max_agents),
        "self observation shape is wrong");
    require(
        simulator.copy_partner_observations().size() ==
            static_cast<std::size_t>(2 * source.capacities.max_agents * 16),
        "partner observation shape is wrong");
    require(
        simulator.copy_map_observations().size() ==
            static_cast<std::size_t>(2 * source.capacities.max_agents * 64),
        "map observation shape is wrong");
}

void test_dynamic_bicycle(const RuntimeScene &source)
{
    RuntimeScene passenger_left = isolated_agent_scene(source, 1, 4.8f, 1.9f, 8.0f);
    RuntimeScene passenger_right = passenger_left;
    RuntimeScene large_left = isolated_agent_scene(source, 1, 10.0f, 2.8f, 8.0f);
    RuntimeScene pedestrian_left = isolated_agent_scene(source, 2, 0.8f, 0.6f, 8.0f);
    const int agents = source.capacities.max_agents;

    SimConfig config;
    DriveSim simulator(
        config,
        {passenger_left, passenger_right, large_left, pedestrian_left});
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(4 * agents), 0);
    std::vector<AgentAction> actions(
        mask.size(), AgentAction{0.0f, 0.0f});
    for (int world = 0; world < 4; ++world) {
        mask[static_cast<std::size_t>(world * agents)] = 1;
    }
    actions[0] = AgentAction{4.0f, 0.4f};
    actions[static_cast<std::size_t>(agents)] = AgentAction{4.0f, -0.4f};
    actions[static_cast<std::size_t>(2 * agents)] = AgentAction{4.0f, 0.4f};
    actions[static_cast<std::size_t>(3 * agents)] = AgentAction{4.0f, 0.4f};
    simulator.set_external_control_mask(mask);
    simulator.set_actions(actions);

    simulator.step();
    SimSnapshot snapshot = simulator.copy_snapshot();
    const VehicleDynamicsState &first = snapshot.dynamics_states[0];
    require(first.longitudinal_acceleration <= config.max_jerk * passenger_left.dt + 1e-4f,
            "acceleration jerk limit was not respected");
    require(first.steering_angle <= config.max_steering_rate * passenger_left.dt + 1e-4f,
            "steering rate limit was not respected");

    for (int step = 1; step < 20; ++step) {
        simulator.step();
    }
    snapshot = simulator.copy_snapshot();
    const int passenger_right_index = agents;
    const int large_index = 2 * agents;
    const int pedestrian_index = 3 * agents;
    require(snapshot.states[0].yaw > 0.01f, "positive steering did not turn left");
    require(snapshot.states[static_cast<std::size_t>(passenger_right_index)].yaw < -0.01f,
            "negative steering did not turn right");
    require(std::fabs(snapshot.states[0].yaw -
                      snapshot.states[static_cast<std::size_t>(large_index)].yaw) > 1e-3f,
            "passenger and large vehicle presets produced identical yaw");
    require(std::fabs(snapshot.states[0].yaw -
                      snapshot.states[static_cast<std::size_t>(pedestrian_index)].yaw) > 1e-3f,
            "non-vehicle agent did not use the kinematic fallback");
    for (int world = 0; world < 4; ++world) {
        const std::size_t index = static_cast<std::size_t>(world * agents);
        require(finite_state(snapshot.states[index]), "dynamic rollout produced a non-finite state");
        require(finite_dynamics(snapshot.dynamics_states[index]),
                "dynamic rollout produced non-finite internal state");
        require(std::fabs(snapshot.dynamics_states[index].steering_angle) <=
                    config.max_abs_steering + 1e-5f,
                "actual steering exceeded its limit");
    }

    RuntimeScene braking_scene = isolated_agent_scene(source, 1, 4.8f, 1.9f, 2.0f);
    DriveSim braking_simulator(config, {braking_scene});
    std::vector<std::uint8_t> braking_mask(static_cast<std::size_t>(agents), 0);
    std::vector<AgentAction> braking_actions(
        static_cast<std::size_t>(agents), AgentAction{0.0f, 0.0f});
    braking_mask[0] = 1;
    braking_actions[0] = AgentAction{-6.0f, 0.0f};
    braking_simulator.set_external_control_mask(braking_mask);
    braking_simulator.set_actions(braking_actions);
    for (int step = 0; step < 30; ++step) {
        braking_simulator.step();
    }
    const SimSnapshot braking = braking_simulator.copy_snapshot();
    require(braking.dynamics_states[0].longitudinal_velocity >= -1e-6f,
            "braking caused reverse motion");
    require(std::hypot(braking.states[0].vx, braking.states[0].vy) < 0.05f,
            "braking did not stop the vehicle");
}

void test_events(RuntimeScene scene)
{
    const int agents = scene.capacities.max_agents;
    require(agents >= 3, "event test requires at least three agent slots");
    scene.counts.agents = 3;
    for (int agent = 0; agent < agents; ++agent) {
        scene.agent_initial_valid[static_cast<std::size_t>(agent)] = agent < 3 ? 1 : 0;
        scene.agent_controllable[static_cast<std::size_t>(agent)] = agent < 3 ? 1 : 0;
    }
    set_state(scene, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    set_state(scene, 1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    set_state(scene, 2, 20.0f, 20.0f, 0.0f, 0.0f, 0.0f);
    std::fill(scene.reference_future_valid.begin(), scene.reference_future_valid.end(), 0);
    scene.agent_goal_valid[0] = 1;
    scene.agent_goal[0] = 0.0f;
    scene.agent_goal[1] = 0.0f;

    std::fill(scene.map_feature_valid.begin(), scene.map_feature_valid.end(), 0);
    std::fill(scene.map_point_valid.begin(), scene.map_point_valid.end(), 0);
    scene.counts.map_features = 2;
    scene.counts.map_points = 6;
    scene.map_feature_valid[0] = 1;
    scene.map_feature_type[0] = 8;
    scene.map_geometry_type[0] = 3;
    scene.map_feature_point_start[0] = 0;
    scene.map_feature_point_count[0] = 4;
    set_map_point(scene, 0, -5.0f, -5.0f);
    set_map_point(scene, 1, 5.0f, -5.0f);
    set_map_point(scene, 2, 5.0f, 5.0f);
    set_map_point(scene, 3, -5.0f, 5.0f);
    scene.map_feature_valid[1] = 1;
    scene.map_feature_type[1] = 4;
    scene.map_geometry_type[1] = 2;
    scene.map_feature_point_start[1] = 4;
    scene.map_feature_point_count[1] = 2;
    set_map_point(scene, 4, -3.0f, 0.0f);
    set_map_point(scene, 5, 3.0f, 0.0f);

    DriveSim simulator(SimConfig{}, {scene});
    std::vector<std::uint8_t> mask(static_cast<std::size_t>(agents), 0);
    std::vector<AgentAction> actions(static_cast<std::size_t>(agents), AgentAction{0.0f, 0.0f});
    mask[0] = 1;
    mask[1] = 1;
    mask[2] = 1;
    simulator.set_external_control_mask(mask);
    simulator.set_actions(actions);
    simulator.step();
    const SimSnapshot snapshot = simulator.copy_snapshot();
    require(snapshot.events[0].collided_vehicle == 1, "vehicle collision was not detected");
    require(snapshot.events[1].collided_vehicle == 1, "paired vehicle collision was not detected");
    require(snapshot.events[0].collided_road == 1, "road edge collision was not detected");
    require(snapshot.events[2].offroad == 1, "offroad event was not detected");
    require(snapshot.events[0].reached_goal == 1, "goal event was not detected");
    require(snapshot.world_done[0] == 0, "collision unexpectedly terminated world");
}

void test_timeout(const RuntimeScene &source)
{
    RuntimeScene short_scene = source;
    short_scene.episode_steps = 2;
    DriveSim simulator(SimConfig{}, {short_scene});
    simulator.step();
    require(simulator.copy_snapshot().world_done[0] == 0, "world ended one step too early");
    simulator.step();
    require(simulator.copy_snapshot().world_done[0] == 1, "world timeout was not raised");
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Usage: simulator_integration_test RUNTIME_SCENE\n";
        return 2;
    }
    try {
        const auto directories = discover_runtime_scenes(std::filesystem::path(argv[1]));
        const RuntimeScene scene = load_runtime_scene(directories.front());
        test_rollout_and_reset(scene);
        test_dynamic_bicycle(scene);
        test_events(scene);
        test_timeout(scene);
        std::cout << "CUDA simulator integration ok\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "CUDA simulator integration failed: " << error.what() << '\n';
        return 1;
    }
}
