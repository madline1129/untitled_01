#pragma once

#include <vector>
struct AgentState {
    float x;
    float y;
    float vx;
    float vy;
};

struct AgentAction {
    float ax;
    float ay;
};

struct SelfObservation {
    float x;
    float y;
    float vx;
    float vy;
    float goal_dx;
    float goal_dy;
};

struct PartnerObservation {
    float rel_x;
    float rel_y;
    float rel_vx;
    float rel_vy;
    float distance;
};

class DriveSim {
public:
    DriveSim(int num_worlds, int num_agents);
    ~DriveSim();  //Destructor, release memory

    // DriveSim manages GPU pointers, so it cannot be copied freely.
    // This prevents two objects from cudaFree-ing the same memory.
    DriveSim(const DriveSim &) = delete;
    DriveSim &operator=(const DriveSim &) = delete;

    void reset();
    void reset_worlds(const std::vector<int> &worlds);
    void set_actions(const std::vector<AgentAction> &actions);
    void step();
    void print() const;
    void print_observations() const;
    void print_partner_observations() const;

    int step_count() const;


private:
    int num_worlds_;
    int num_agents_;
    int total_agents_;

    float dt_;
    float goal_x_;
    float goal_y_;

    AgentState *d_states_;
    AgentAction *d_actions_;
    SelfObservation *d_obs_;
    PartnerObservation *d_partner_obs_;
    float *d_rewards_;
    int *d_dones_;
    int *d_world_resets_;

    int step_count_;
    int episode_len_;
};
