#include "runtime_scene.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

using json = nlohmann::json;

bool host_is_little_endian()
{
    const std::uint16_t value = 1;
    return *reinterpret_cast<const std::uint8_t *>(&value) == 1;
}

std::size_t checked_product(const std::vector<int> &shape, const std::string &name)
{
    std::size_t result = 1;
    for (int dimension : shape) {
        if (dimension <= 0) {
            throw std::runtime_error("tensor " + name + " has a non-positive dimension");
        }
        const auto dim = static_cast<std::size_t>(dimension);
        if (result > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::runtime_error("tensor " + name + " shape overflows size_t");
        }
        result *= dim;
    }
    return result;
}

std::size_t dtype_size(const std::string &dtype)
{
    if (dtype == "float32" || dtype == "int32") {
        return 4;
    }
    if (dtype == "uint8") {
        return 1;
    }
    throw std::runtime_error("unsupported runtime dtype: " + dtype);
}

std::vector<int> read_shape(const json &spec, const std::string &name)
{
    if (!spec.contains("shape") || !spec.at("shape").is_array()) {
        throw std::runtime_error("tensor " + name + " is missing shape");
    }
    return spec.at("shape").get<std::vector<int>>();
}

std::filesystem::path tensor_path(
    const std::filesystem::path &directory,
    const json &spec,
    const std::string &name)
{
    if (!spec.contains("file") || !spec.at("file").is_string()) {
        throw std::runtime_error("tensor " + name + " is missing file");
    }
    const std::filesystem::path filename = spec.at("file").get<std::string>();
    if (filename.is_absolute() || filename.has_parent_path()) {
        throw std::runtime_error("tensor " + name + " file must be a local filename");
    }
    return directory / filename;
}

void validate_all_tensor_files(const std::filesystem::path &directory, const json &manifest)
{
    if (!manifest.contains("tensors") || !manifest.at("tensors").is_object()) {
        throw std::runtime_error("runtime manifest is missing tensors");
    }
    for (const auto &[name, spec] : manifest.at("tensors").items()) {
        if (!spec.contains("dtype") || !spec.at("dtype").is_string()) {
            throw std::runtime_error("tensor " + name + " is missing dtype");
        }
        const std::string dtype = spec.at("dtype").get<std::string>();
        const std::vector<int> shape = read_shape(spec, name);
        const std::filesystem::path path = tensor_path(directory, spec, name);
        const std::size_t expected = checked_product(shape, name) * dtype_size(dtype);
        std::error_code error;
        const std::uintmax_t actual = std::filesystem::file_size(path, error);
        if (error || actual != expected) {
            std::ostringstream message;
            message << "tensor " << name << " byte size mismatch: expected "
                    << expected << ", got " << (error ? 0 : actual);
            throw std::runtime_error(message.str());
        }
    }
}

template <typename T>
std::vector<T> load_tensor(
    const std::filesystem::path &directory,
    const json &manifest,
    const std::string &name,
    const std::string &expected_dtype,
    const std::vector<int> &expected_shape)
{
    const auto &tensors = manifest.at("tensors");
    if (!tensors.contains(name)) {
        throw std::runtime_error("runtime manifest is missing required tensor " + name);
    }
    const json &spec = tensors.at(name);
    const std::string dtype = spec.at("dtype").get<std::string>();
    const std::vector<int> shape = read_shape(spec, name);
    if (dtype != expected_dtype) {
        throw std::runtime_error(
            "tensor " + name + " dtype mismatch: expected " + expected_dtype + ", got " + dtype);
    }
    if (shape != expected_shape) {
        throw std::runtime_error("tensor " + name + " shape does not match runtime capacities");
    }
    if (sizeof(T) != dtype_size(dtype)) {
        throw std::runtime_error("host type size does not match tensor " + name);
    }

    const std::size_t count = checked_product(shape, name);
    std::vector<T> values(count);
    std::ifstream stream(tensor_path(directory, spec, name), std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open tensor " + name);
    }
    stream.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!stream || stream.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("cannot read tensor " + name + " exactly");
    }
    return values;
}

int require_nonnegative_count(const json &counts, const char *name, int capacity)
{
    const int value = counts.at(name).get<int>();
    if (value < 0 || value > capacity) {
        throw std::runtime_error(std::string("runtime count ") + name + " exceeds capacity");
    }
    return value;
}

}  // namespace

bool RuntimeCapacities::operator==(const RuntimeCapacities &other) const
{
    return max_agents == other.max_agents &&
           history_steps == other.history_steps &&
           max_future_steps == other.max_future_steps &&
           max_map_features == other.max_map_features &&
           max_map_points == other.max_map_points &&
           max_map_edges == other.max_map_edges &&
           max_traffic_lights == other.max_traffic_lights &&
           max_route_features == other.max_route_features;
}

RuntimeScene load_runtime_scene(const std::filesystem::path &directory)
{
    if (!host_is_little_endian()) {
        throw std::runtime_error("RuntimeScenario currently requires a little-endian host");
    }
    const std::filesystem::path manifest_path = directory / "manifest.json";
    std::ifstream stream(manifest_path);
    if (!stream) {
        throw std::runtime_error("runtime manifest not found: " + manifest_path.string());
    }

    json manifest;
    try {
        stream >> manifest;
    } catch (const json::exception &error) {
        throw std::runtime_error("invalid runtime manifest: " + std::string(error.what()));
    }
    if (manifest.value("schema_version", "") != "rl-runtime-1.0") {
        throw std::runtime_error("unsupported runtime schema version");
    }
    if (manifest.value("tensor_layout", "") != "C-contiguous little-endian") {
        throw std::runtime_error("unsupported runtime tensor layout");
    }

    RuntimeScene scene;
    scene.directory = std::filesystem::absolute(directory);
    scene.source_scenario_id = manifest.at("source").value("source_scenario_id", directory.filename().string());
    scene.dt = manifest.at("dt").get<float>();
    scene.episode_steps = manifest.at("episode_steps").get<int>();
    if (!(scene.dt > 0.0f) || scene.episode_steps <= 0) {
        throw std::runtime_error("runtime dt and episode_steps must be positive");
    }

    const json &capacities = manifest.at("capacities");
    scene.capacities = RuntimeCapacities{
        capacities.at("max_agents").get<int>(),
        capacities.at("history_steps").get<int>(),
        capacities.at("max_future_steps").get<int>(),
        capacities.at("max_map_features").get<int>(),
        capacities.at("max_map_points").get<int>(),
        capacities.at("max_map_edges").get<int>(),
        capacities.at("max_traffic_lights").get<int>(),
        capacities.at("max_route_features").get<int>(),
    };
    if (scene.capacities.max_agents <= 0 || scene.capacities.max_future_steps <= 0 ||
        scene.capacities.max_map_features <= 0 || scene.capacities.max_map_points <= 0 ||
        scene.capacities.max_traffic_lights <= 0) {
        throw std::runtime_error("runtime capacities must be positive");
    }

    const json &counts = manifest.at("counts");
    scene.counts.agents = require_nonnegative_count(counts, "agents", scene.capacities.max_agents);
    scene.counts.map_features = require_nonnegative_count(
        counts, "map_features", scene.capacities.max_map_features);
    scene.counts.map_points = require_nonnegative_count(
        counts, "map_points", scene.capacities.max_map_points);
    scene.counts.map_edges = require_nonnegative_count(
        counts, "map_edges", scene.capacities.max_map_edges);
    scene.counts.traffic_lights = require_nonnegative_count(
        counts, "traffic_lights", scene.capacities.max_traffic_lights);
    scene.counts.route_features = require_nonnegative_count(
        counts, "route_features", scene.capacities.max_route_features);

    validate_all_tensor_files(directory, manifest);

    const int agents = scene.capacities.max_agents;
    const int future = scene.capacities.max_future_steps;
    const int features = scene.capacities.max_map_features;
    const int points = scene.capacities.max_map_points;
    const int edges = scene.capacities.max_map_edges;
    const int lights = scene.capacities.max_traffic_lights;
    const int route_features = scene.capacities.max_route_features;

    scene.agent_initial_state = load_tensor<float>(
        directory, manifest, "agent_initial_state", "float32", {agents, 5});
    scene.agent_initial_valid = load_tensor<std::uint8_t>(
        directory, manifest, "agent_initial_valid", "uint8", {agents});
    scene.agent_type = load_tensor<std::int32_t>(
        directory, manifest, "agent_type", "int32", {agents});
    scene.agent_is_ego = load_tensor<std::uint8_t>(
        directory, manifest, "agent_is_ego", "uint8", {agents});
    scene.agent_controllable = load_tensor<std::uint8_t>(
        directory, manifest, "agent_controllable", "uint8", {agents});
    scene.agent_dimensions = load_tensor<float>(
        directory, manifest, "agent_dimensions", "float32", {agents, 3});
    scene.agent_goal = load_tensor<float>(
        directory, manifest, "agent_goal", "float32", {agents, 2});
    scene.agent_goal_valid = load_tensor<std::uint8_t>(
        directory, manifest, "agent_goal_valid", "uint8", {agents});
    scene.reference_future = load_tensor<float>(
        directory, manifest, "reference_future", "float32", {future, agents, 5});
    scene.reference_future_valid = load_tensor<std::uint8_t>(
        directory, manifest, "reference_future_valid", "uint8", {future, agents});

    scene.map_points = load_tensor<float>(
        directory, manifest, "map_points", "float32", {points, 3});
    scene.map_point_valid = load_tensor<std::uint8_t>(
        directory, manifest, "map_point_valid", "uint8", {points});
    scene.map_feature_type = load_tensor<std::int32_t>(
        directory, manifest, "map_feature_type", "int32", {features});
    scene.map_geometry_type = load_tensor<std::int32_t>(
        directory, manifest, "map_geometry_type", "int32", {features});
    scene.map_feature_point_start = load_tensor<std::int32_t>(
        directory, manifest, "map_feature_point_start", "int32", {features});
    scene.map_feature_point_count = load_tensor<std::int32_t>(
        directory, manifest, "map_feature_point_count", "int32", {features});
    scene.map_feature_valid = load_tensor<std::uint8_t>(
        directory, manifest, "map_feature_valid", "uint8", {features});
    scene.map_speed_limit = load_tensor<float>(
        directory, manifest, "map_speed_limit", "float32", {features});
    scene.map_speed_limit_valid = load_tensor<std::uint8_t>(
        directory, manifest, "map_speed_limit_valid", "uint8", {features});
    scene.map_edges = load_tensor<std::int32_t>(
        directory, manifest, "map_edges", "int32", {edges, 3});
    scene.map_edge_valid = load_tensor<std::uint8_t>(
        directory, manifest, "map_edge_valid", "uint8", {edges});

    scene.traffic_light_feature_index = load_tensor<std::int32_t>(
        directory, manifest, "traffic_light_feature_index", "int32", {lights});
    scene.traffic_light_state = load_tensor<std::uint8_t>(
        directory, manifest, "traffic_light_state", "uint8", {future + 1, lights});
    scene.traffic_light_valid = load_tensor<std::uint8_t>(
        directory, manifest, "traffic_light_valid", "uint8", {future + 1, lights});
    scene.route_feature_index = load_tensor<std::int32_t>(
        directory, manifest, "route_feature_index", "int32", {route_features});
    scene.route_feature_valid = load_tensor<std::uint8_t>(
        directory, manifest, "route_feature_valid", "uint8", {route_features});
    scene.route_goal = load_tensor<float>(
        directory, manifest, "route_goal", "float32", {2});
    scene.route_goal_valid = load_tensor<std::uint8_t>(
        directory, manifest, "route_goal_valid", "uint8", {1});

    if (scene.agent_goal_valid[0] == 0 && scene.route_goal_valid[0] != 0) {
        scene.agent_goal[0] = scene.route_goal[0];
        scene.agent_goal[1] = scene.route_goal[1];
        scene.agent_goal_valid[0] = 1;
    }

    scene.agent_ids = manifest.at("agent_ids").get<std::vector<std::string>>();
    if (static_cast<int>(scene.agent_ids.size()) != scene.counts.agents) {
        throw std::runtime_error("agent_ids length does not match agent count");
    }
    scene.agent_ids.resize(static_cast<std::size_t>(agents));
    for (int index = scene.counts.agents; index < agents; ++index) {
        scene.agent_ids[static_cast<std::size_t>(index)] = "padding_" + std::to_string(index);
    }
    if (scene.agent_initial_valid.empty() || scene.agent_initial_valid[0] == 0 ||
        scene.agent_is_ego.empty() || scene.agent_is_ego[0] == 0) {
        throw std::runtime_error("runtime slot 0 must contain a valid ego agent");
    }
    return scene;
}

std::vector<std::filesystem::path> discover_runtime_scenes(
    const std::filesystem::path &input)
{
    if (std::filesystem::is_regular_file(input / "manifest.json")) {
        return {input};
    }
    if (!std::filesystem::is_directory(input)) {
        throw std::runtime_error("runtime input does not exist: " + input.string());
    }

    std::vector<std::filesystem::path> directories;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(input)) {
        if (entry.is_regular_file() && entry.path().filename() == "manifest.json") {
            directories.push_back(entry.path().parent_path());
        }
    }
    std::sort(directories.begin(), directories.end());
    directories.erase(std::unique(directories.begin(), directories.end()), directories.end());
    if (directories.empty()) {
        throw std::runtime_error("no runtime manifests found under: " + input.string());
    }
    return directories;
}

std::vector<RuntimeScene> load_runtime_batch(
    const std::filesystem::path &input,
    int num_worlds)
{
    if (num_worlds <= 0) {
        throw std::runtime_error("num_worlds must be positive");
    }
    const std::vector<std::filesystem::path> directories = discover_runtime_scenes(input);
    std::vector<RuntimeScene> unique_scenes;
    unique_scenes.reserve(directories.size());
    for (const auto &directory : directories) {
        unique_scenes.push_back(load_runtime_scene(directory));
        if (unique_scenes.size() > 1 &&
            !(unique_scenes.front().capacities == unique_scenes.back().capacities)) {
            throw std::runtime_error("all runtime scenes in a batch must use identical capacities");
        }
    }

    std::vector<RuntimeScene> worlds;
    worlds.reserve(static_cast<std::size_t>(num_worlds));
    for (int world = 0; world < num_worlds; ++world) {
        worlds.push_back(unique_scenes[static_cast<std::size_t>(world) % unique_scenes.size()]);
    }
    return worlds;
}
