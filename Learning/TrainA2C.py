import sys
sys.path.append('.')

from Environment.PatrollingEnvironments import MultiAgentPatrolling
from Algorithm.RainbowDQL.Agent.AdvanteActorCritic import A2CAgent
import numpy as np

N = 4

sc_map = np.genfromtxt('./Environment/maps/example_map.csv', delimiter=',')
visitable_locations = np.vstack(np.where(sc_map != 0)).T
random_index = np.random.choice(np.arange(0,len(visitable_locations)), N, replace=False)
initial_positions = np.asarray([[24, 21], [28,24], [27,19], [24,24]])

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
							   ground_truth_type='shekel',
	                           obstacles=True,
                               frame_stacking = 1,
                               state_index_stacking = (2,3,4))

multiagent = A2CAgent(
				env = env,
				n_steps=None,
				learning_rate=0.0001,
				gamma=0.99,
				logdir='runs/a2c_2',
				log_name='A2C_2',
				save_every=1000,
                device='cuda:0',
                )

multiagent.train(episodes=20000)
