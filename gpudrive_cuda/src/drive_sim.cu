#include "drive_sim.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

/*
1 CPU: Data Preparation
2 CPU: Call GPU malloc
3 CPU -> GPU: memcpy
4 GPU: Muti-kernal Parallel Computation
5 GPU -> CPU: memcpy
6 CPU: Result Demonstration
*/

// Check CUDA runtime calls immediately so failures point to the real source.
#define CUDA_CHECK(expr)                                      \
    do {                                                      \
        cudaError_t err = (expr);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error: "                       \
                      << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":"            \
                      << __LINE__ << std::endl;               \
            std::exit(1);                                     \
        }                                                     \
    } while (0)

// One thread initializes one agent. This is the reset equivalent in our simulator.
__global__ void reset_kernel(
    AgentState *states,
    AgentAction *actions,
    float *rewards,
    int *dones,
    const int *world_resets,
    int reset_all,
    int num_worlds,
    int num_agents
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_agents = num_worlds * num_agents;

    if (idx >= total_agents) {
        return;
    }

    int world_id = idx / num_agents;
    int agent_id = idx % num_agents;

    if (!reset_all && world_resets[world_id] == 0) {
        return;
    }

    states[idx] = AgentState{
        .x = static_cast<float>(agent_id),
        .y = static_cast<float>(world_id),
        .vx = 0.0f,
        .vy = 0.0f,
    };

    actions[idx] = AgentAction{
        .ax = 1.0f,
        .ay = 0.0f,
    };

    rewards[idx] = 0.0f;
    dones[idx] = 0;
}

// One thread updates one agent. This mirrors the core parallelism in GPUDrive.
__global__ void step_kernel(
    AgentState *states,
    const AgentAction *actions,
    float *rewards,
    int *dones,
    int num_worlds,
    int num_agents,
    float goal_x,
    float goal_y,
    float dt,
    int step_count,
    int episode_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_agents = num_worlds * num_agents;

    if (idx >= total_agents) {
        return;
    }

    int world_id = idx / num_agents;
    int agent_id = idx % num_agents;
    (void)world_id;
    (void)agent_id;

    if (dones[idx]) {
        rewards[idx] = 0.0f;
        return;
    }

    AgentState s = states[idx];
    AgentAction a = actions[idx];

    s.vx += a.ax * dt;
    s.vy += a.ay * dt;
    s.x += s.vx * dt;
    s.y += s.vy * dt;

    float dx = s.x - goal_x;
    float dy = s.y - goal_y;
    float dist = sqrtf(dx * dx + dy * dy);

    rewards[idx] = -dist;
    bool reached_goal = dist < 0.5f;
    bool timeout = step_count >= episode_len;
    dones[idx] = (reached_goal || timeout) ? 1 : 0;
    states[idx] = s;
}

__global__ void observation_kernel(
    const AgentState * states,
    SelfObservation * obs,
    int num_worlds,
    int num_agents,
    float goal_x ,
    float goal_y
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_agents = num_worlds * num_agents;

    if (idx >= total_agents) {
        return;
    }

    AgentState s = states[idx];

    obs[idx] = SelfObservation{
        .x = s.x,
        .y = s.y,
        .vx = s.vx,
        .vy = s.vy,
        .goal_dx = goal_x - s.x,
        .goal_dy = goal_y - s.y,
    };
}

__global__ void partner_observation_kernel(
    const AgentState *states,
    PartnerObservation *partner_obs,
    int num_worlds,
    int num_agents
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_agents = num_worlds * num_agents;

    if (idx >= total_agents) {
        return ;
    }
    
    int world_id = idx /num_agents;
    int agent_id = idx % num_agents;

    AgentState self = states[idx];

    int partner_slot = 0;

    for (int other = 0; other < num_agents; other++) {
        if (other == agent_id) {
            continue;
        }

        int other_idx = world_id * num_agents + other;
        AgentState p = states[other_idx];

        float rel_x = p.x - self.x;
        float rel_y = p.y - self.y;
        float rel_vx = p.vx - self.vx;
        float rel_vy = p.vy - self.vy;
        float distance = sqrtf(rel_x * rel_x + rel_y * rel_y);

        int out_idx = world_id * num_agents * (num_agents - 1) + agent_id * (num_agents - 1) + partner_slot;

        partner_obs[out_idx] = PartnerObservation{
            .rel_x = rel_x,
            .rel_y = rel_y,
            .rel_vx = rel_vx,
            .rel_vy = rel_vy,
            .distance = distance,
        };

        partner_slot += 1;
    }

}

DriveSim::DriveSim(int num_worlds, int num_agents)
    : num_worlds_(num_worlds),
      num_agents_(num_agents),
      total_agents_(num_worlds * num_agents),
      step_count_(0),
      episode_len_(20),
      dt_(0.1f),
      goal_x_(10.0f),
      goal_y_(0.0f),
      d_states_(nullptr),
      d_actions_(nullptr),
      d_obs_(nullptr),
      d_partner_obs_(nullptr),
      d_rewards_(nullptr),
      d_dones_(nullptr),
      d_world_resets_(nullptr)
{
    CUDA_CHECK(cudaMalloc(&d_states_, total_agents_ * sizeof(AgentState)));
    CUDA_CHECK(cudaMalloc(&d_actions_, total_agents_ * sizeof(AgentAction)));
    CUDA_CHECK(cudaMalloc(&d_rewards_, total_agents_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dones_, total_agents_ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_obs_, total_agents_ * sizeof(SelfObservation)));
    CUDA_CHECK(cudaMalloc(
        &d_partner_obs_,
        total_agents_ * (num_agents_ - 1) * sizeof(PartnerObservation)
    ));
    CUDA_CHECK(cudaMalloc(&d_world_resets_, num_worlds_ * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_world_resets_, 0, num_worlds_ * sizeof(int)));
}

DriveSim::~DriveSim()
{
    CUDA_CHECK(cudaFree(d_states_));
    CUDA_CHECK(cudaFree(d_actions_));
    CUDA_CHECK(cudaFree(d_rewards_));
    CUDA_CHECK(cudaFree(d_dones_));
    CUDA_CHECK(cudaFree(d_obs_));
    CUDA_CHECK(cudaFree(d_partner_obs_));
    CUDA_CHECK(cudaFree(d_world_resets_));
}

void DriveSim::set_actions(const std::vector<AgentAction> &actions)
{
    if (static_cast<int>(actions.size()) != total_agents_) {
        std::cerr
            <<"set_actions expected " << total_agents_
            << " actions, got " << actions.size()
            << std::endl;
        std::exit(1);
    }

    CUDA_CHECK(cudaMemcpy(
        d_actions_,
        actions.data(),
        total_agents_ * sizeof(AgentAction),
        cudaMemcpyHostToDevice
    ));
}

void DriveSim::reset()
{
    step_count_ = 0;
    const int threads_per_block = 128;
    const int blocks = (total_agents_ + threads_per_block - 1) / threads_per_block;

    reset_kernel<<<blocks, threads_per_block>>>(
        d_states_,
        d_actions_,
        d_rewards_,
        d_dones_,
        d_world_resets_,
        1,
        num_worlds_,
        num_agents_
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    observation_kernel<<<blocks, threads_per_block>>>(
        d_states_,
        d_obs_,
        num_worlds_,
        num_agents_,
        goal_x_,
        goal_y_
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    partner_observation_kernel<<<blocks, threads_per_block>>>(
        d_states_,
        d_partner_obs_,
        num_worlds_,
        num_agents_
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void DriveSim::reset_worlds(const std::vector<int> &worlds)
{
    std::vector<int> h_world_resets(num_worlds_, 0);

    for (int world : worlds) {
        if (world < 0 || world >= num_worlds_) {
            std::cerr << "Invalid world index: " << world << std::endl;
            std::exit(1);
        }

        h_world_resets[world] = 1;
    }

    CUDA_CHECK(cudaMemcpy(
        d_world_resets_,
        h_world_resets.data(),
        num_worlds_ * sizeof(int),
        cudaMemcpyHostToDevice
    ));

    const int threads_per_block = 128;
    const int blocks = (total_agents_ + threads_per_block - 1) / threads_per_block;

    reset_kernel<<<blocks, threads_per_block>>>(
        d_states_,
        d_actions_,
        d_rewards_,
        d_dones_,
        d_world_resets_,
        0,
        num_worlds_,
        num_agents_
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    observation_kernel<<<blocks, threads_per_block>>>(
        d_states_,
        d_obs_,
        num_worlds_,
        num_agents_,
        goal_x_,
        goal_y_
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    partner_observation_kernel<<<blocks, threads_per_block>>>(
        d_states_,
        d_partner_obs_,
        num_worlds_,
        num_agents_
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(d_world_resets_, 0, num_worlds_ * sizeof(int)));
}

void DriveSim::step()
{
    const int threads_per_block = 128;
    const int blocks = (total_agents_ + threads_per_block - 1) / threads_per_block;

    step_kernel<<<blocks, threads_per_block>>>(
        d_states_,
        d_actions_,
        d_rewards_,
        d_dones_,
        num_worlds_,
        num_agents_,
        goal_x_,
        goal_y_,
        dt_,
        step_count_,
        episode_len_
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    observation_kernel<<<blocks, threads_per_block>>>(
        d_states_,
        d_obs_,
        num_worlds_,
        num_agents_,
        goal_x_,
        goal_y_
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    partner_observation_kernel<<<blocks, threads_per_block>>>(
        d_states_,
        d_partner_obs_,
        num_worlds_,
        num_agents_
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    step_count_ += 1;
}

void DriveSim::print() const
{
    std::vector<AgentState> h_states(total_agents_);
    std::vector<float> h_rewards(total_agents_);
    std::vector<int> h_dones(total_agents_);

    CUDA_CHECK(cudaMemcpy(
        h_states.data(),
        d_states_,
        total_agents_ * sizeof(AgentState),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        h_rewards.data(),
        d_rewards_,
        total_agents_ * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        h_dones.data(),
        d_dones_,
        total_agents_ * sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    for (int world = 0; world < num_worlds_; world++) {
        std::cout << "world " << world << std::endl;

        for (int agent = 0; agent < num_agents_; agent++) {
            int idx = world * num_agents_ + agent;

            std::cout
                << "  agent " << agent
                << " x=" << h_states[idx].x
                << " y=" << h_states[idx].y
                << " vx=" << h_states[idx].vx
                << " reward=" << h_rewards[idx]
                << " done=" << h_dones[idx]
                << std::endl;
        }
    }
}

void DriveSim::print_observations() const
{
    std::vector<SelfObservation> h_obs(total_agents_);

    CUDA_CHECK(cudaMemcpy(
        h_obs.data(),
        d_obs_,
        total_agents_ * sizeof(SelfObservation),
        cudaMemcpyDeviceToHost
    ));

    for (int world = 0; world < num_worlds_; world++) {
        std::cout << "observations world " << world << std::endl;

        for (int agent = 0; agent < num_agents_; agent++) {
            int idx = world * num_agents_ + agent;
            const SelfObservation &o = h_obs[idx];

            std::cout
                << "  agent " << agent
                << " obs=["
                << o.x << ", "
                << o.y << ", "
                << o.vx << ", "
                << o.vy << ", "
                << o.goal_dx << ", "
                << o.goal_dy << "]"
                << std::endl;
        }
    }
}

void DriveSim::print_partner_observations() const
{
    const int partners_per_agent = num_agents_ - 1;
    std::vector<PartnerObservation> h_partner_obs(
        total_agents_ * partners_per_agent
    );

    CUDA_CHECK(cudaMemcpy(
        h_partner_obs.data(),
        d_partner_obs_,
        h_partner_obs.size() * sizeof(PartnerObservation),
        cudaMemcpyDeviceToHost
    ));

    for (int world = 0; world < num_worlds_; world++) {
        std::cout << "partner observations world " << world << std::endl;

        for (int agent = 0; agent < num_agents_; agent++) {
            std::cout << "  agent " << agent << std::endl;

            for (int slot = 0; slot < partners_per_agent; slot++) {
                int idx =
                    world * num_agents_ * partners_per_agent
                    + agent * partners_per_agent
                    + slot;

                const PartnerObservation &o = h_partner_obs[idx];

                std::cout
                    << "    partner_slot " << slot
                    << " rel=["
                    << o.rel_x << ", "
                    << o.rel_y << "]"
                    << " rel_v=["
                    << o.rel_vx << ", "
                    << o.rel_vy << "]"
                    << " distance=" << o.distance
                    << std::endl;
            }
        }
    }
}

int DriveSim::step_count() const
{
    return step_count_;
}
