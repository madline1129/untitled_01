#include "drive_sim.cuh"
#include "runtime_scene.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int kThreads = 256;

std::vector<RuntimeScene> load_scenes(
    const std::vector<std::string> &runtime_paths,
    int num_worlds)
{
    if (runtime_paths.empty()) {
        throw std::runtime_error("runtime_paths must contain at least one path");
    }
    if (num_worlds <= 0) {
        throw std::runtime_error("num_worlds must be positive");
    }

    std::vector<std::filesystem::path> directories;
    for (const std::string &value : runtime_paths) {
        const std::vector<std::filesystem::path> discovered =
            discover_runtime_scenes(std::filesystem::path(value));
        directories.insert(directories.end(), discovered.begin(), discovered.end());
    }
    std::sort(directories.begin(), directories.end());
    directories.erase(std::unique(directories.begin(), directories.end()), directories.end());

    std::vector<RuntimeScene> unique_scenes;
    unique_scenes.reserve(directories.size());
    for (const std::filesystem::path &directory : directories) {
        RuntimeScene scene = load_runtime_scene(directory);
        if (!unique_scenes.empty() && !(unique_scenes.front().capacities == scene.capacities)) {
            throw std::runtime_error("all runtime scenes must use identical capacities");
        }
        unique_scenes.push_back(std::move(scene));
    }

    std::vector<RuntimeScene> worlds;
    worlds.reserve(static_cast<std::size_t>(num_worlds));
    for (int world = 0; world < num_worlds; ++world) {
        worlds.push_back(unique_scenes[static_cast<std::size_t>(world) % unique_scenes.size()]);
    }
    return worlds;
}

__global__ void pack_self_kernel(
    const SelfObservation *source,
    float *values,
    std::uint8_t *valid,
    int count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const SelfObservation item = source[index];
    float *output = values + index * 11;
    output[0] = item.x;
    output[1] = item.y;
    output[2] = item.yaw;
    output[3] = item.vx;
    output[4] = item.vy;
    output[5] = item.speed;
    output[6] = item.length;
    output[7] = item.width;
    output[8] = item.goal_dx;
    output[9] = item.goal_dy;
    output[10] = item.steps_remaining;
    valid[index] = static_cast<std::uint8_t>(item.valid != 0);
}

__global__ void pack_partner_kernel(
    const PartnerObservation *source,
    float *values,
    std::uint8_t *valid,
    int count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const PartnerObservation item = source[index];
    float *output = values + index * 9;
    output[0] = item.rel_x;
    output[1] = item.rel_y;
    output[2] = item.rel_vx;
    output[3] = item.rel_vy;
    output[4] = item.rel_yaw;
    output[5] = item.length;
    output[6] = item.width;
    output[7] = item.type;
    output[8] = item.distance;
    valid[index] = static_cast<std::uint8_t>(item.valid != 0);
}

__global__ void pack_map_kernel(
    const MapObservation *source,
    float *values,
    std::uint8_t *valid,
    int count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const MapObservation item = source[index];
    float *output = values + index * 7;
    output[0] = item.start_x;
    output[1] = item.start_y;
    output[2] = item.end_x;
    output[3] = item.end_y;
    output[4] = item.type;
    output[5] = item.speed_limit;
    output[6] = item.distance;
    valid[index] = static_cast<std::uint8_t>(item.valid != 0);
}

void check_cuda_tensor(
    const torch::Tensor &tensor,
    torch::ScalarType dtype,
    const std::vector<std::int64_t> &shape,
    int device_id,
    const char *name)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.get_device() == device_id, name, " is on the wrong CUDA device");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.sizes().vec() == shape, name, " has the wrong shape");
}

template <typename T>
torch::Tensor tensor_view(
    T *pointer,
    const std::vector<std::int64_t> &shape,
    torch::ScalarType dtype,
    int device_id,
    std::shared_ptr<DriveSim> owner)
{
    const torch::TensorOptions options = torch::TensorOptions()
        .dtype(dtype)
        .device(torch::Device(torch::kCUDA, device_id));
    return torch::from_blob(
        pointer,
        shape,
        [owner = std::move(owner)](void *) mutable { owner.reset(); },
        options);
}

class TorchDriveSim {
public:
    TorchDriveSim(
        const std::vector<std::string> &runtime_paths,
        int num_worlds,
        int device_id)
        : device_id_(device_id)
    {
        c10::cuda::CUDAGuard guard(device_id_);
        std::vector<RuntimeScene> scenes = load_scenes(runtime_paths, num_worlds);
        simulator_ = std::shared_ptr<DriveSim>(
            new DriveSim(SimConfig{}, std::move(scenes)),
            [device_id = device_id_](DriveSim *simulator) {
                c10::cuda::CUDAGuard delete_guard(device_id);
                delete simulator;
            });
        worlds_ = simulator_->num_worlds();
        agents_ = simulator_->max_agents();
        partners_ = simulator_->partner_observation_count();
        map_segments_ = simulator_->map_observation_count();
        allocate_tensors();
    }

    py::dict reset()
    {
        c10::cuda::CUDAGuard guard(device_id_);
        torch::Tensor mask = torch::ones(
            {worlds_},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_id_));
        return reset_worlds(mask);
    }

    py::dict reset_worlds(const torch::Tensor &reset_mask)
    {
        c10::cuda::CUDAGuard guard(device_id_);
        check_cuda_tensor(reset_mask, torch::kUInt8, {worlds_}, device_id_, "reset_mask");
        const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id_).stream();
        simulator_->reset_worlds_device(reset_mask.data_ptr<std::uint8_t>(), stream);
        return frame();
    }

    void set_control_modes(const torch::Tensor &control_modes)
    {
        c10::cuda::CUDAGuard guard(device_id_);
        check_cuda_tensor(
            control_modes,
            torch::kUInt8,
            {worlds_, agents_},
            device_id_,
            "control_modes");
        const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id_).stream();
        simulator_->set_control_modes_device(control_modes.data_ptr<std::uint8_t>(), stream);
    }

    py::dict step(const torch::Tensor &actions)
    {
        c10::cuda::CUDAGuard guard(device_id_);
        check_cuda_tensor(
            actions,
            torch::kFloat32,
            {worlds_, agents_, 2},
            device_id_,
            "actions");
        const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id_).stream();
        simulator_->step_device(
            reinterpret_cast<const AgentAction *>(actions.data_ptr<float>()),
            stream);
        return frame();
    }

    py::dict frame()
    {
        c10::cuda::CUDAGuard guard(device_id_);
        const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_id_).stream();
        const SimDeviceView view = simulator_->device_view();
        const int total_agents = worlds_ * agents_;
        const int total_partners = total_agents * partners_;
        const int total_map = total_agents * map_segments_;

        pack_self_kernel<<<(total_agents + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            view.self_observations,
            self_.data_ptr<float>(),
            self_valid_.data_ptr<std::uint8_t>(),
            total_agents);
        pack_partner_kernel<<<(total_partners + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            view.partner_observations,
            partner_.data_ptr<float>(),
            partner_valid_.data_ptr<std::uint8_t>(),
            total_partners);
        pack_map_kernel<<<(total_map + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            view.map_observations,
            map_.data_ptr<float>(),
            map_valid_.data_ptr<std::uint8_t>(),
            total_map);
        TORCH_CHECK(cudaGetLastError() == cudaSuccess, "failed to launch observation packing kernels");

        py::dict output;
        output["states"] = states_;
        output["dynamics"] = dynamics_;
        output["applied_actions"] = applied_actions_;
        output["events"] = events_;
        output["self"] = self_;
        output["self_valid"] = self_valid_;
        output["partners"] = partner_;
        output["partner_valid"] = partner_valid_;
        output["map"] = map_;
        output["map_valid"] = map_valid_;
        output["valid"] = valid_;
        output["agent_type"] = agent_type_;
        output["agent_is_ego"] = agent_is_ego_;
        output["agent_controllable"] = agent_controllable_;
        output["control_modes"] = control_modes_;
        output["world_step"] = world_steps_;
        output["world_done"] = world_done_;
        return output;
    }

    int num_worlds() const { return worlds_; }
    int max_agents() const { return agents_; }

private:
    void allocate_tensors()
    {
        const SimDeviceView view = simulator_->device_view();
        auto float_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCUDA, device_id_);
        auto byte_options = torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(torch::kCUDA, device_id_);

        states_ = tensor_view(
            const_cast<AgentState *>(view.states),
            {worlds_, agents_, 5},
            torch::kFloat32,
            device_id_,
            simulator_);
        dynamics_ = tensor_view(
            const_cast<VehicleDynamicsState *>(view.dynamics_states),
            {worlds_, agents_, 5},
            torch::kFloat32,
            device_id_,
            simulator_);
        applied_actions_ = tensor_view(
            const_cast<AgentAction *>(view.applied_actions),
            {worlds_, agents_, 2},
            torch::kFloat32,
            device_id_,
            simulator_);
        events_ = tensor_view(
            const_cast<AgentEvent *>(view.events),
            {worlds_, agents_, 5},
            torch::kInt32,
            device_id_,
            simulator_);
        valid_ = tensor_view(
            const_cast<std::uint8_t *>(view.valid),
            {worlds_, agents_},
            torch::kUInt8,
            device_id_,
            simulator_);
        agent_type_ = tensor_view(
            const_cast<std::int32_t *>(view.agent_type),
            {worlds_, agents_},
            torch::kInt32,
            device_id_,
            simulator_);
        agent_is_ego_ = tensor_view(
            const_cast<std::uint8_t *>(view.agent_is_ego),
            {worlds_, agents_},
            torch::kUInt8,
            device_id_,
            simulator_);
        agent_controllable_ = tensor_view(
            const_cast<std::uint8_t *>(view.agent_controllable),
            {worlds_, agents_},
            torch::kUInt8,
            device_id_,
            simulator_);
        control_modes_ = tensor_view(
            const_cast<std::uint8_t *>(view.control_modes),
            {worlds_, agents_},
            torch::kUInt8,
            device_id_,
            simulator_);
        world_steps_ = tensor_view(
            const_cast<int *>(view.world_steps),
            {worlds_},
            torch::kInt32,
            device_id_,
            simulator_);
        world_done_ = tensor_view(
            const_cast<int *>(view.world_done),
            {worlds_},
            torch::kInt32,
            device_id_,
            simulator_);

        self_ = torch::empty({worlds_, agents_, 11}, float_options);
        self_valid_ = torch::empty({worlds_, agents_}, byte_options);
        partner_ = torch::empty({worlds_, agents_, partners_, 9}, float_options);
        partner_valid_ = torch::empty({worlds_, agents_, partners_}, byte_options);
        map_ = torch::empty({worlds_, agents_, map_segments_, 7}, float_options);
        map_valid_ = torch::empty({worlds_, agents_, map_segments_}, byte_options);
    }

    int device_id_ = 0;
    int worlds_ = 0;
    int agents_ = 0;
    int partners_ = 0;
    int map_segments_ = 0;
    std::shared_ptr<DriveSim> simulator_;

    torch::Tensor states_;
    torch::Tensor dynamics_;
    torch::Tensor applied_actions_;
    torch::Tensor events_;
    torch::Tensor valid_;
    torch::Tensor agent_type_;
    torch::Tensor agent_is_ego_;
    torch::Tensor agent_controllable_;
    torch::Tensor control_modes_;
    torch::Tensor world_steps_;
    torch::Tensor world_done_;
    torch::Tensor self_;
    torch::Tensor self_valid_;
    torch::Tensor partner_;
    torch::Tensor partner_valid_;
    torch::Tensor map_;
    torch::Tensor map_valid_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    py::class_<TorchDriveSim>(module, "TorchDriveSim")
        .def(
            py::init<const std::vector<std::string> &, int, int>(),
            py::arg("runtime_paths"),
            py::arg("num_worlds"),
            py::arg("device_id") = 0)
        .def("reset", py::overload_cast<>(&TorchDriveSim::reset))
        .def("reset_worlds", &TorchDriveSim::reset_worlds, py::arg("reset_mask"))
        .def("set_control_modes", &TorchDriveSim::set_control_modes, py::arg("control_modes"))
        .def("step", &TorchDriveSim::step, py::arg("actions"))
        .def("frame", &TorchDriveSim::frame)
        .def_property_readonly("num_worlds", &TorchDriveSim::num_worlds)
        .def_property_readonly("max_agents", &TorchDriveSim::max_agents);
}
