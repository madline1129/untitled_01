#include "runtime_scene.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>

namespace {

void expect_failure(const std::function<void()> &operation, const char *name)
{
    try {
        operation();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error(std::string("expected failure was not detected: ") + name);
}

void rewrite_manifest(
    const std::filesystem::path &directory,
    const std::function<void(nlohmann::json &)> &change)
{
    const std::filesystem::path path = directory / "manifest.json";
    std::ifstream input(path);
    nlohmann::json manifest;
    input >> manifest;
    change(manifest);
    std::ofstream output(path);
    output << manifest.dump(2) << '\n';
}

}  // namespace

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Usage: runtime_loader_test RUNTIME_SCENE_OR_PARENT\n";
        return 2;
    }
    try {
        const std::filesystem::path input = argv[1];
        const auto discovered = discover_runtime_scenes(input);
        if (discovered.empty()) {
            throw std::runtime_error("discovery returned no scenes");
        }
        const RuntimeScene scene = load_runtime_scene(discovered.front());
        if (scene.capacities.max_agents <= 0 || scene.counts.agents <= 0) {
            throw std::runtime_error("invalid agent capacities or counts");
        }
        if (scene.agent_initial_state.size() !=
            static_cast<std::size_t>(scene.capacities.max_agents * 5)) {
            throw std::runtime_error("initial state tensor was not loaded");
        }
        if (scene.agent_initial_valid[0] != 1 || scene.agent_is_ego[0] != 1) {
            throw std::runtime_error("slot 0 is not a valid ego");
        }
        if (std::fabs(scene.agent_initial_state[0]) > 1e-4f ||
            std::fabs(scene.agent_initial_state[1]) > 1e-4f ||
            std::fabs(scene.agent_initial_state[2]) > 1e-4f) {
            throw std::runtime_error("ego reset pose is not local origin");
        }

        const auto batch = load_runtime_batch(input, 3);
        if (batch.size() != 3 || !(batch[0].capacities == batch[2].capacities)) {
            throw std::runtime_error("round-robin batch loading failed");
        }

        const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        const std::filesystem::path temporary =
            std::filesystem::temp_directory_path() / ("gpudrive_runtime_test_" + std::to_string(nonce));
        std::filesystem::create_directories(temporary);
        try {
            const std::filesystem::path missing = temporary / "missing";
            std::filesystem::copy(
                discovered.front(), missing, std::filesystem::copy_options::recursive);
            std::filesystem::remove(missing / "agent_initial_state.bin");
            expect_failure([&]() { load_runtime_scene(missing); }, "missing tensor");

            const std::filesystem::path wrong_schema = temporary / "wrong_schema";
            std::filesystem::copy(
                discovered.front(), wrong_schema, std::filesystem::copy_options::recursive);
            rewrite_manifest(wrong_schema, [](nlohmann::json &manifest) {
                manifest["schema_version"] = "rl-runtime-999";
            });
            expect_failure([&]() { load_runtime_scene(wrong_schema); }, "unsupported schema");

            const std::filesystem::path wrong_size = temporary / "wrong_size";
            std::filesystem::copy(
                discovered.front(), wrong_size, std::filesystem::copy_options::recursive);
            std::ofstream(wrong_size / "agent_initial_valid.bin", std::ios::binary | std::ios::app).put('\0');
            expect_failure([&]() { load_runtime_scene(wrong_size); }, "tensor byte size");

            const std::filesystem::path wrong_dtype = temporary / "wrong_dtype";
            std::filesystem::copy(
                discovered.front(), wrong_dtype, std::filesystem::copy_options::recursive);
            rewrite_manifest(wrong_dtype, [](nlohmann::json &manifest) {
                manifest["tensors"]["agent_type"]["dtype"] = "float32";
            });
            expect_failure([&]() { load_runtime_scene(wrong_dtype); }, "tensor dtype");

            const std::filesystem::path wrong_shape = temporary / "wrong_shape";
            std::filesystem::copy(
                discovered.front(), wrong_shape, std::filesystem::copy_options::recursive);
            rewrite_manifest(wrong_shape, [&](nlohmann::json &manifest) {
                manifest["tensors"]["agent_initial_state"]["shape"] = {
                    scene.capacities.max_agents * 5, 1};
            });
            expect_failure([&]() { load_runtime_scene(wrong_shape); }, "tensor shape");

            const std::filesystem::path mixed = temporary / "mixed";
            const std::filesystem::path mixed_a = mixed / "a";
            const std::filesystem::path mixed_b = mixed / "b";
            std::filesystem::create_directories(mixed);
            std::filesystem::copy(
                discovered.front(), mixed_a, std::filesystem::copy_options::recursive);
            std::filesystem::copy(
                discovered.front(), mixed_b, std::filesystem::copy_options::recursive);
            rewrite_manifest(mixed_b, [](nlohmann::json &manifest) {
                manifest["capacities"]["history_steps"] =
                    manifest["capacities"]["history_steps"].get<int>() + 1;
            });
            expect_failure([&]() { load_runtime_batch(mixed, 2); }, "mixed capacities");
        } catch (...) {
            std::filesystem::remove_all(temporary);
            throw;
        }
        std::filesystem::remove_all(temporary);

        std::cout << "runtime loader ok: " << scene.source_scenario_id
                  << " agents=" << scene.counts.agents
                  << " map_features=" << scene.counts.map_features << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "runtime loader test failed: " << error.what() << '\n';
        return 1;
    }
}
