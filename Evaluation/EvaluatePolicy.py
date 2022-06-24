from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np
import matplotlib.pyplot as plt
import torch
N = 4
sc_map = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
visitable_locations = np.vstack(np.where(sc_map != 0)).T
random_index = np.random.choice(np.arange(0,len(visitable_locations)), N, replace=False)
initial_positions = np.asarray([[24, 21],[28,24],[27,19],[24,24]])

env = MultiAgentPatrolling(scenario_map=sc_map,
                           fleet_initial_positions=initial_positions,
                           distance_budget=200,
                           number_of_vehicles=N,
                           seed=0,
                           detection_length=1,
                           movement_length=1,
                           max_collisions=5,
                           forget_factor=0.5,
                           attrittion=0.1,
                           networked_agents=True,
                           hard_penalization=False,
                           max_connection_distance=7,
                           optimal_connection_distance=3,
                           max_number_of_disconnections=1000,
                           obstacles=False)


multiagent = MultiAgentDuelingDQNAgent(env=env,
                                       memory_size=int(1E4),
                                       batch_size=64,
                                       target_update=1000,
                                       soft_update=False,
                                       tau=0.0001,
                                       epsilon_values=[1.0, 0.05],
                                       epsilon_interval=[0.0, 0.33],
                                       learning_starts=10,
                                       gamma=0.99,
                                       lr=1e-4,
                                       noisy=True,
                                       train_every=20,
                                       save_every=5000)

multiagent.load_model('/home/azken/Samuel/MultiAgentPatrollingProblem/Learning/runs/Jun16_20-11-09_M3009R21854/BestPolicy.pth')

multiagent.epsilon = 0.0
done = False
s = env.reset()



R = []

while not done:

    selected_action = []
    for i in range(env.number_of_agents):
        individual_state = env.individual_agent_observation(state=s, agent_num=i)
        q_values = multiagent.dqn(torch.FloatTensor(individual_state).unsqueeze(0).to(multiagent.device)).detach().cpu().numpy().flatten()
        mask = np.asarray([env.fleet.vehicles[i].check_action(a) for a in range(0,8)])
        q_values[mask] = -np.inf
        selected_action.append(np.argmax(q_values))

    s, r, done, i = env.step(selected_action)
    print(env.fleet.number_of_disconnections)
    env.render()
    R.append(r)


env.render()
plt.show()
plt.close()
print(np.sum(R))
R = np.asarray(R)
plt.plot(np.cumsum(R, axis=0))
plt.show()
