from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np

N = 4

N = 4
sc_map = np.genfromtxt('../Environment/example_map.csv', delimiter=',')
visitable_locations = np.vstack(np.where(sc_map != 0)).T
random_index = np.random.choice(np.arange(0,len(visitable_locations)), N, replace=False)
initial_positions = visitable_locations[random_index]

env = MultiAgentPatrolling(scenario_map=sc_map,
                           fleet_initial_positions=initial_positions,
                           distance_budget=200,
                           number_of_vehicles=N,
                           seed=0,
                           detection_length=2,
                           max_collisions=5,
                           forget_factor=0.5,
                           attrittion=0.1,
                           networked_agents=False,
                           max_connection_distance=20,
                           min_isolation_distance=10,
                           max_number_of_disconnections=50)

multiagent = MultiAgentDuelingDQNAgent(env=env,
                                       memory_size=int(1E6),
                                       batch_size=64,
                                       target_update=1000,
                                       soft_update=False,
                                       tau=0.0001,
                                       epsilon_values=[1.0, 0.05],
                                       epsilon_interval=[0.0, 0.33],
                                       learning_starts=10,
                                       gamma=0.99,
                                       lr=1e-4,
                                       noisy=False,
                                       train_every=10)

multiagent.train(episodes=50000)
