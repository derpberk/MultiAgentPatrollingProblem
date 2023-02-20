import sys
sys.path.append('.')

from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.DuelingDQNAgent import MultiAgentDuelingDQNAgent
import numpy as np


N = 4
reward_weights = (0.1, 1)

sc_map = np.genfromtxt('Environment/Maps/example_map.csv', delimiter=',')
initial_positions = np.asarray([[24, 21],[28,24],[27,19],[24,24]])

env = MultiAgentPatrolling(scenario_map=sc_map,
							fleet_initial_positions=initial_positions,
							distance_budget=150,
							number_of_vehicles=N,
							seed=10,
							miopic=True,
							detection_length=2,
							movement_length=2,
							max_collisions=10,
							forget_factor=0.5,
							attrittion=0.1,
							networked_agents=False,
							reward_type='model_changes',
							ground_truth_type='algae_bloom',
							obstacles=True,
							frame_stacking = 1,
							state_index_stacking = (2,3,4),
							reward_weights=reward_weights)

multiagent = MultiAgentDuelingDQNAgent(env=env,
									memory_size=int(1E5),
									batch_size=128,
									target_update=1000,
									soft_update=True,
									tau=0.001,
									epsilon_values=[1.0, 0.05],
									epsilon_interval=[0.0, 0.33],
									learning_starts=100,
									gamma=0.99,
									lr=1e-4,
									noisy=False,
									train_every=15,
									save_every=5000,
									distributional=False,
									masked_actions=True,
									device='cuda:1',
				logdir=f'runs/Multiagent/Vehicles_{N}/Experimento_' + reward_weights.replace(', ', '_') + '_short_dinamico',
				eval_episodes=10,
				eval_every=1000)

multiagent.train(episodes=10000)
