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

void test_rollout_and_reset(const RuntimeScene &source)
{
    DriveSim simulator(SimConfig{}, {source, source});
    const SimSnapshot initial = simulator.copy_snapshot();
    require(initial.world_steps[0] == 0 && initial.world_steps[1] == 0, "reset step is not zero");
    require(initial.valid[0] == 1, "ego is invalid after reset");

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
        test_events(scene);
        test_timeout(scene);
        std::cout << "CUDA simulator integration ok\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "CUDA simulator integration failed: " << error.what() << '\n';
        return 1;
    }
}
