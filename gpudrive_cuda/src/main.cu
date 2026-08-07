#include "drive_sim.cuh"

#include <vector>
#include <iostream>

int main()
{
    const int num_worlds = 2;
    const int num_agents = 4;
    const int total_agents = num_worlds * num_agents;

    DriveSim sim(num_worlds, num_agents);
    sim.reset();

    std::vector<AgentAction> actions(total_agents);


    for (int step = 0; step < 10; step++) {
        for (int world = 0; world < num_worlds; world++) {
            for (int agent = 0; agent < num_agents; agent++) {
                int idx = world * num_agents + agent;

                actions[idx] = AgentAction{
                    .ax = 1.0f + 0.1f * static_cast<float>(agent),
                    .ay = 0.05f * static_cast<float>(world),
                };
            }
        }

        sim.set_actions(actions);
        sim.step();
        std::cout << "finished step " << sim.step_count() << std::endl;

        if (step == 5) {
            sim.reset_worlds({1});
            std::cout << "reset world 1" << std::endl;
        }
    }

    sim.print();
    sim.print_observations();
    sim.print_partner_observations();

    return 0;
}
