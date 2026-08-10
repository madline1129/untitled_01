#include "drive_sim.cuh"
#include "runtime_scene.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
    std::filesystem::path runtime;
    std::filesystem::path output;
    int worlds = 1;
    int steps = 0;
    std::vector<int> external_agents;
    float acceleration = 0.0f;
    float steering = 0.0f;
};

void print_usage(const char *program)
{
    std::cerr
        << "Usage: " << program << " --runtime PATH --output DIR [options]\n"
        << "Options:\n"
        << "  --worlds N                 Number of parallel worlds (default: 1)\n"
        << "  --steps N                  Rollout steps, 0 uses longest episode\n"
        << "  --external-agent SLOT      Externally control a slot; may be repeated\n"
        << "  --acceleration VALUE       Constant external acceleration in m/s^2\n"
        << "  --steering VALUE           Constant external steering in radians\n";
}

Arguments parse_arguments(int argc, char **argv)
{
    Arguments arguments;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        auto require_value = [&](const std::string &name) -> std::string {
            if (index + 1 >= argc) {
                throw std::runtime_error(name + " requires a value");
            }
            return argv[++index];
        };
        if (option == "--runtime") {
            arguments.runtime = require_value(option);
        } else if (option == "--output") {
            arguments.output = require_value(option);
        } else if (option == "--worlds") {
            arguments.worlds = std::stoi(require_value(option));
        } else if (option == "--steps") {
            arguments.steps = std::stoi(require_value(option));
        } else if (option == "--external-agent") {
            arguments.external_agents.push_back(std::stoi(require_value(option)));
        } else if (option == "--acceleration") {
            arguments.acceleration = std::stof(require_value(option));
        } else if (option == "--steering") {
            arguments.steering = std::stof(require_value(option));
        } else if (option == "--help" || option == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + option);
        }
    }
    if (arguments.runtime.empty() || arguments.output.empty()) {
        throw std::runtime_error("--runtime and --output are required");
    }
    if (arguments.worlds <= 0 || arguments.steps < 0) {
        throw std::runtime_error("--worlds must be positive and --steps must be non-negative");
    }
    std::sort(arguments.external_agents.begin(), arguments.external_agents.end());
    arguments.external_agents.erase(
        std::unique(arguments.external_agents.begin(), arguments.external_agents.end()),
        arguments.external_agents.end());
    return arguments;
}

std::string csv_string(const std::string &value)
{
    if (value.find_first_of(",\"\n") == std::string::npos) {
        return value;
    }
    std::string escaped = "\"";
    for (char character : value) {
        if (character == '\"') {
            escaped += '\"';
        }
        escaped += character;
    }
    escaped += '\"';
    return escaped;
}

void write_csv_header(std::ofstream &stream)
{
    stream
        << "world,step,agent_slot,agent_id,valid,control_mode,"
        << "x,y,yaw,vx,vy,acceleration,steering,"
        << "collided_vehicle,collided_road,offroad,reached_goal,world_done\n";
}

void write_snapshot(
    std::ofstream &stream,
    const SimSnapshot &snapshot,
    const std::vector<RuntimeScene> &scenes,
    int max_agents)
{
    stream << std::setprecision(9);
    for (int world = 0; world < static_cast<int>(scenes.size()); ++world) {
        for (int agent = 0; agent < max_agents; ++agent) {
            const int index = world * max_agents + agent;
            const AgentState &state = snapshot.states[static_cast<std::size_t>(index)];
            const AgentAction &action = snapshot.applied_actions[static_cast<std::size_t>(index)];
            const AgentEvent &event = snapshot.events[static_cast<std::size_t>(index)];
            stream
                << world << ','
                << snapshot.world_steps[static_cast<std::size_t>(world)] << ','
                << agent << ','
                << csv_string(scenes[static_cast<std::size_t>(world)].agent_ids[static_cast<std::size_t>(agent)]) << ','
                << static_cast<int>(snapshot.valid[static_cast<std::size_t>(index)]) << ','
                << (snapshot.external_control[static_cast<std::size_t>(index)] != 0 ? "external" : "auto") << ','
                << state.x << ',' << state.y << ',' << state.yaw << ',' << state.vx << ',' << state.vy << ','
                << action.acceleration << ',' << action.steering << ','
                << event.collided_vehicle << ',' << event.collided_road << ','
                << event.offroad << ',' << event.reached_goal << ','
                << snapshot.world_done[static_cast<std::size_t>(world)] << '\n';
        }
    }
}

bool all_worlds_done(const SimSnapshot &snapshot)
{
    return std::all_of(
        snapshot.world_done.begin(), snapshot.world_done.end(), [](int done) { return done != 0; });
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        std::vector<RuntimeScene> scenes = load_runtime_batch(arguments.runtime, arguments.worlds);
        const int max_agents = scenes.front().capacities.max_agents;
        for (int slot : arguments.external_agents) {
            if (slot < 0 || slot >= max_agents) {
                throw std::runtime_error("external agent slot is outside runtime capacity");
            }
        }

        int rollout_steps = arguments.steps;
        if (rollout_steps == 0) {
            for (const RuntimeScene &scene : scenes) {
                rollout_steps = std::max(rollout_steps, scene.episode_steps);
            }
        }
        std::filesystem::create_directories(arguments.output);
        const std::filesystem::path trace_path = arguments.output / "trace.csv";
        const std::filesystem::path summary_path = arguments.output / "summary.json";
        std::ofstream trace(trace_path);
        if (!trace) {
            throw std::runtime_error("cannot create trace: " + trace_path.string());
        }
        write_csv_header(trace);

        DriveSim simulator(SimConfig{}, scenes);
        std::vector<std::uint8_t> control_mask(
            static_cast<std::size_t>(arguments.worlds * max_agents), 0);
        std::vector<AgentAction> actions(
            static_cast<std::size_t>(arguments.worlds * max_agents), AgentAction{0.0f, 0.0f});
        for (int world = 0; world < arguments.worlds; ++world) {
            for (int slot : arguments.external_agents) {
                const int index = world * max_agents + slot;
                control_mask[static_cast<std::size_t>(index)] = 1;
                actions[static_cast<std::size_t>(index)] = AgentAction{
                    arguments.acceleration,
                    arguments.steering,
                };
            }
        }
        simulator.set_external_control_mask(control_mask);
        simulator.set_actions(actions);

        SimSnapshot snapshot = simulator.copy_snapshot();
        write_snapshot(trace, snapshot, scenes, max_agents);
        int executed_steps = 0;
        int vehicle_collision_rows = 0;
        int road_collision_rows = 0;
        int offroad_rows = 0;
        for (int step = 0; step < rollout_steps && !all_worlds_done(snapshot); ++step) {
            simulator.step();
            snapshot = simulator.copy_snapshot();
            write_snapshot(trace, snapshot, scenes, max_agents);
            executed_steps += 1;
            for (const AgentEvent &event : snapshot.events) {
                vehicle_collision_rows += event.collided_vehicle;
                road_collision_rows += event.collided_road;
                offroad_rows += event.offroad;
            }
        }
        trace.close();

        nlohmann::json summary;
        summary["runtime_input"] = std::filesystem::absolute(arguments.runtime).string();
        summary["worlds"] = arguments.worlds;
        summary["max_agents"] = max_agents;
        summary["executed_steps"] = executed_steps;
        summary["trace"] = std::filesystem::absolute(trace_path).string();
        summary["vehicle_collision_agent_steps"] = vehicle_collision_rows;
        summary["road_collision_agent_steps"] = road_collision_rows;
        summary["offroad_agent_steps"] = offroad_rows;
        summary["external_agent_slots"] = arguments.external_agents;
        summary["scenes"] = nlohmann::json::array();
        for (int world = 0; world < arguments.worlds; ++world) {
            const RuntimeScene &scene = scenes[static_cast<std::size_t>(world)];
            summary["scenes"].push_back({
                {"world", world},
                {"scenario_id", scene.source_scenario_id},
                {"directory", scene.directory.string()},
                {"agents", scene.counts.agents},
                {"map_features", scene.counts.map_features},
                {"episode_steps", scene.episode_steps},
                {"dt", scene.dt},
            });
        }
        std::ofstream summary_stream(summary_path);
        if (!summary_stream) {
            throw std::runtime_error("cannot create summary: " + summary_path.string());
        }
        summary_stream << summary.dump(2) << '\n';

        std::cout << "simulation complete\n"
                  << "trace: " << trace_path << '\n'
                  << "summary: " << summary_path << '\n'
                  << "worlds: " << arguments.worlds << '\n'
                  << "steps: " << executed_steps << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "[error] " << error.what() << std::endl;
        print_usage(argv[0]);
        return 1;
    }
}
