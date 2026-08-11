#include "drive_sim.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void cuda_check(cudaError_t result, const char *expression, const char *file, int line)
{
    if (result == cudaSuccess) {
        return;
    }
    std::cerr << "CUDA error for " << expression << ": "
              << cudaGetErrorString(result) << " at " << file << ':' << line << '\n';
    std::exit(1);
}

#define CUDA_CHECK(expression) cuda_check((expression), #expression, __FILE__, __LINE__)

__global__ void reset_kernel(
    AgentState *states,
    AgentAction *actions,
    float *rewards,
    int *dones,
    const int *world_reset_mask,
    int reset_all,
    int num_worlds,
    int num_agents)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_agents = num_worlds * num_agents;
    if (index >= total_agents) {
        return;
    }

    const int world = index / num_agents;
    const int agent = index % num_agents;
    if (reset_all == 0 && world_reset_mask[world] == 0) {
        return;
    }

    states[index] = AgentState{
        static_cast<float>(agent),
        static_cast<float>(world),
        0.0f,
        0.0f,
    };
    actions[index] = AgentAction{0.0f, 0.0f};
    rewards[index] = 0.0f;
    dones[index] = 0;
}

__global__ void step_kernel(
    AgentState *states,
    const AgentAction *actions,
    float *rewards,
    int *dones,
    int total_agents,
    float goal_x,
    float goal_y,
    float dt,
    int next_step,
    int episode_length)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_agents) {
        return;
    }
    if (dones[index] != 0) {
        rewards[index] = 0.0f;
        return;
    }

    AgentState state = states[index];
    const AgentAction action = actions[index];

    state.vx += action.ax * dt;
    state.vy += action.ay * dt;
    state.x += state.vx * dt;
    state.y += state.vy * dt;

    const float dx = state.x - goal_x;
    const float dy = state.y - goal_y;
    const float distance = sqrtf(dx * dx + dy * dy);
    const bool reached_goal = distance < 0.5f;
    const bool timed_out = next_step >= episode_length;

    states[index] = state;
    rewards[index] = -distance;
    dones[index] = reached_goal || timed_out ? 1 : 0;
}

__global__ void self_observation_kernel(
    const AgentState *states,
    SelfObservation *observations,
    int total_agents,
    float goal_x,
    float goal_y)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_agents) {
        return;
    }

    const AgentState state = states[index];
    observations[index] = SelfObservation{
        state.x,
        state.y,
        state.vx,
        state.vy,
        goal_x - state.x,
        goal_y - state.y,
    };
}

__global__ void partner_observation_kernel(
    const AgentState *states,
    PartnerObservation *observations,
    int num_worlds,
    int num_agents)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_agents = num_worlds * num_agents;
    if (index >= total_agents) {
        return;
    }

    const int world = index / num_agents;
    const int self_agent = index % num_agents;
    const int partners_per_agent = num_agents - 1;
    const AgentState self = states[index];
    int partner_slot = 0;

    for (int other_agent = 0; other_agent < num_agents; ++other_agent) {
        if (other_agent == self_agent) {
            continue;
        }

        const int other_index = world * num_agents + other_agent;
        const AgentState other = states[other_index];
        const float rel_x = other.x - self.x;
        const float rel_y = other.y - self.y;
        const int output_index =
            (world * num_agents + self_agent) * partners_per_agent + partner_slot;

        observations[output_index] = PartnerObservation{
            rel_x,
            rel_y,
            other.vx - self.vx,
            other.vy - self.vy,
            sqrtf(rel_x * rel_x + rel_y * rel_y),
        };
        ++partner_slot;
    }
}

}  // namespace

DriveSim::DriveSim(int num_worlds, int num_agents)
    : num_worlds_(num_worlds),
      num_agents_(num_agents),
      total_agents_(num_worlds * num_agents),
      step_count_(0),
      episode_length_(20),
      dt_(0.1f),
      goal_x_(10.0f),
      goal_y_(0.0f)
{
    if (num_worlds_ <= 0 || num_agents_ <= 1) {
        throw std::invalid_argument("DriveSim requires positive worlds and at least two agents");
    }

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_states_), total_agents_ * sizeof(AgentState)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_actions_), total_agents_ * sizeof(AgentAction)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_rewards_), total_agents_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_dones_), total_agents_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_self_observations_),
        total_agents_ * sizeof(SelfObservation)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_partner_observations_),
        total_agents_ * (num_agents_ - 1) * sizeof(PartnerObservation)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_world_reset_mask_), num_worlds_ * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_world_reset_mask_, 0, num_worlds_ * sizeof(int)));
}

DriveSim::~DriveSim()
{
    release();
}

void DriveSim::release() noexcept
{
    cudaFree(d_states_);
    cudaFree(d_actions_);
    cudaFree(d_self_observations_);
    cudaFree(d_partner_observations_);
    cudaFree(d_rewards_);
    cudaFree(d_dones_);
    cudaFree(d_world_reset_mask_);
}

void DriveSim::set_actions(const std::vector<AgentAction> &actions)
{
    if (actions.size() != static_cast<std::size_t>(total_agents_)) {
        throw std::invalid_argument("set_actions size must equal worlds * agents");
    }
    CUDA_CHECK(cudaMemcpy(
        d_actions_,
        actions.data(),
        actions.size() * sizeof(AgentAction),
        cudaMemcpyHostToDevice));
}

void DriveSim::reset()
{
    step_count_ = 0;
    constexpr int threads = 128;
    const int blocks = (total_agents_ + threads - 1) / threads;
    reset_kernel<<<blocks, threads>>>(
        d_states_,
        d_actions_,
        d_rewards_,
        d_dones_,
        d_world_reset_mask_,
        1,
        num_worlds_,
        num_agents_);
    CUDA_CHECK(cudaGetLastError());
    build_observations();
}

void DriveSim::reset_worlds(const std::vector<int> &world_ids)
{
    std::vector<int> reset_mask(static_cast<std::size_t>(num_worlds_), 0);
    for (int world : world_ids) {
        if (world < 0 || world >= num_worlds_) {
            throw std::invalid_argument("reset_worlds received an invalid world index");
        }
        reset_mask[static_cast<std::size_t>(world)] = 1;
    }

    CUDA_CHECK(cudaMemcpy(
        d_world_reset_mask_,
        reset_mask.data(),
        reset_mask.size() * sizeof(int),
        cudaMemcpyHostToDevice));

    constexpr int threads = 128;
    const int blocks = (total_agents_ + threads - 1) / threads;
    reset_kernel<<<blocks, threads>>>(
        d_states_,
        d_actions_,
        d_rewards_,
        d_dones_,
        d_world_reset_mask_,
        0,
        num_worlds_,
        num_agents_);
    CUDA_CHECK(cudaGetLastError());
    build_observations();
    CUDA_CHECK(cudaMemset(d_world_reset_mask_, 0, num_worlds_ * sizeof(int)));
}

void DriveSim::step()
{
    constexpr int threads = 128;
    const int blocks = (total_agents_ + threads - 1) / threads;
    step_kernel<<<blocks, threads>>>(
        d_states_,
        d_actions_,
        d_rewards_,
        d_dones_,
        total_agents_,
        goal_x_,
        goal_y_,
        dt_,
        step_count_ + 1,
        episode_length_);
    CUDA_CHECK(cudaGetLastError());
    ++step_count_;
    build_observations();
}

void DriveSim::build_observations()
{
    constexpr int threads = 128;
    const int blocks = (total_agents_ + threads - 1) / threads;
    self_observation_kernel<<<blocks, threads>>>(
        d_states_, d_self_observations_, total_agents_, goal_x_, goal_y_);
    CUDA_CHECK(cudaGetLastError());
    partner_observation_kernel<<<blocks, threads>>>(
        d_states_, d_partner_observations_, num_worlds_, num_agents_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void DriveSim::print_states() const
{
    std::vector<AgentState> states(static_cast<std::size_t>(total_agents_));
    std::vector<float> rewards(static_cast<std::size_t>(total_agents_));
    std::vector<int> dones(static_cast<std::size_t>(total_agents_));
    CUDA_CHECK(cudaMemcpy(
        states.data(), d_states_, states.size() * sizeof(AgentState), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        rewards.data(), d_rewards_, rewards.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        dones.data(), d_dones_, dones.size() * sizeof(int), cudaMemcpyDeviceToHost));

    for (int world = 0; world < num_worlds_; ++world) {
        std::cout << "world " << world << '\n';
        for (int agent = 0; agent < num_agents_; ++agent) {
            const int index = world * num_agents_ + agent;
            const AgentState &state = states[static_cast<std::size_t>(index)];
            std::cout << "  agent " << agent
                      << " x=" << state.x
                      << " y=" << state.y
                      << " vx=" << state.vx
                      << " vy=" << state.vy
                      << " reward=" << rewards[static_cast<std::size_t>(index)]
                      << " done=" << dones[static_cast<std::size_t>(index)] << '\n';
        }
    }
}

void DriveSim::print_self_observations() const
{
    std::vector<SelfObservation> observations(static_cast<std::size_t>(total_agents_));
    CUDA_CHECK(cudaMemcpy(
        observations.data(),
        d_self_observations_,
        observations.size() * sizeof(SelfObservation),
        cudaMemcpyDeviceToHost));

    for (int world = 0; world < num_worlds_; ++world) {
        std::cout << "self observations world " << world << '\n';
        for (int agent = 0; agent < num_agents_; ++agent) {
            const int index = world * num_agents_ + agent;
            const SelfObservation &observation = observations[static_cast<std::size_t>(index)];
            std::cout << "  agent " << agent
                      << " state=[" << observation.x << ", " << observation.y
                      << ", " << observation.vx << ", " << observation.vy << ']'
                      << " goal_delta=[" << observation.goal_dx
                      << ", " << observation.goal_dy << "]\n";
        }
    }
}

void DriveSim::print_partner_observations() const
{
    const int partners_per_agent = num_agents_ - 1;
    std::vector<PartnerObservation> observations(
        static_cast<std::size_t>(total_agents_ * partners_per_agent));
    CUDA_CHECK(cudaMemcpy(
        observations.data(),
        d_partner_observations_,
        observations.size() * sizeof(PartnerObservation),
        cudaMemcpyDeviceToHost));

    for (int world = 0; world < num_worlds_; ++world) {
        std::cout << "partner observations world " << world << '\n';
        for (int agent = 0; agent < num_agents_; ++agent) {
            std::cout << "  agent " << agent << '\n';
            for (int slot = 0; slot < partners_per_agent; ++slot) {
                const int index =
                    (world * num_agents_ + agent) * partners_per_agent + slot;
                const PartnerObservation &observation =
                    observations[static_cast<std::size_t>(index)];
                std::cout << "    slot " << slot
                          << " rel=[" << observation.rel_x << ", " << observation.rel_y << ']'
                          << " rel_v=[" << observation.rel_vx << ", " << observation.rel_vy << ']'
                          << " distance=" << observation.distance << '\n';
            }
        }
    }
}
