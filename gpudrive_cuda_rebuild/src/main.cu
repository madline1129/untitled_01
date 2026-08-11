#include "drive_sim.cuh"

#include <iostream>
#include <vector>

int main()
{
    constexpr int num_worlds = 2;
    constexpr int num_agents = 4;
    constexpr int total_agents = num_worlds * num_agents;

    DriveSim simulator(num_worlds, num_agents);
    simulator.reset();

    std::vector<AgentAction> actions(static_cast<std::size_t>(total_agents));
    for (int step = 0; step < 10; ++step) {
        for (int world = 0; world < num_worlds; ++world) {
            for (int agent = 0; agent < num_agents; ++agent) {
                const int index = world * num_agents + agent;
                actions[static_cast<std::size_t>(index)] = AgentAction{
                    1.0f + 0.1f * static_cast<float>(agent),
                    0.05f * static_cast<float>(world),
                };
            }
        }

        simulator.set_actions(actions);
        simulator.step();
        std::cout << "finished step " << simulator.step_count() << '\n';

        if (step == 5) {
            simulator.reset_worlds({1});
            std::cout << "reset world 1\n";
        }
    }

    simulator.print_states();
    simulator.print_self_observations();
    simulator.print_partner_observations();
    return 0;
}
