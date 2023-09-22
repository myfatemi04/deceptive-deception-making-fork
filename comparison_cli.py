"""
Usage: comparison_cli --in=<in> --out=<out> --horizon=<horizon> --step_cost=<step_cost> --discount=<discount>
"""

import pickle
import docopt

args = docopt.docopt(__doc__)

import numpy as np
from grid_world import grid_world
from MDP_class import MDP
from graph import Graph
from Simulations import simulate_Markovian, simulate_Stationary
from grid_world_michael import load_grid
import csv
import time
import json

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
'''Environment Specifications'''

# gridworld = load_grid("../robotic-deception/Pursuit/gridworlds/square_rect_based/8x8C.tmx")
gridworld = load_grid(args['--in'])

# Constructing input information. Grid size, absorbing states, goals, initial position.
rows, cols = gridworld.tiles.shape
# Define the time_horizon
time_horizon = int(args['--horizon'])

row_and_column_to_flat_index = lambda row_number, column_number: (rows - 1 - row_number) * cols + column_number

goals = [row_and_column_to_flat_index(row, col) for (row, col) in gridworld.goal_positions]
true_goal = goals[0]
# Find obstacle positions
absorb = []
for i in range(rows):
    for j in range(cols):
        if gridworld.obstacle_mask[i, j]:
            absorb.append(row_and_column_to_flat_index(i, j))
absorb.extend(goals)
init = row_and_column_to_flat_index(*gridworld.start_positions[0])

# Defines the motion model
# model: (model_state[key] = [actions], model_state_action[(key,action)] = ([probabilities],[successors]))
model = grid_world(rows, cols, absorb, slip=0)
num_of_states = len(model[0])

# Define the discount factor
discount = float(args['--discount'])

# Define the cost of taking a step in the MDP (used in the shortest path computation)
one_step_cost = int(args['--step_cost']) # default: 30

# Define the prior distribution on the goals
prior_goals = np.ones((len(goals),1)) / len(goals)

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
'''Define the posterior goal probabilities for each state'''

# Perform the computations on the base MDP model
base_MDP = MDP(model)


# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
'''Approximation algorithm 1'''

# This approximation algorithm performs on the base MDP model.
# It defines the costs for each state-action pair as follows:
#   Let c(s) be the posterior goal probability of the state s.
#   Let T(s) be the minimum time steps to reach the state s from the initial state.
#   Let base_MDP_costs(s,a) be the cost for the state-action pair (s,a), which is used in the linear problem
# We have base_MDP_costs(s,a) = c(s) * discount ** T(s)

start_time = time.time()

# Calculation of the posteriors is a necessary component
goal_posteriors = base_MDP.compute_goal_posteriors(prior_goals, init, goals, one_step_cost, discount)

graph = Graph(model)
min_times = graph.Dijkstra(init)

# This dictionary contains the "deceptive" rewards for each state-action pair. We can
# use this to evaluate results from the GNN.
from get_mdp_costs import get_exaggeration_mdp_costs

# Use MDP costs for "exaggeration" deception.
base_MDP_costs = get_exaggeration_mdp_costs(base_MDP, goals, init, discount, prior_goals, absorb)

# Save the MDP costs to a file
with open(args['--out'] + '.mdp_costs', 'wb') as f:
    pickle.dump(base_MDP_costs, f)

[a,policy] = base_MDP.compute_min_cost_subject_to_max_reach(init,[true_goal], absorb, base_MDP_costs)

end_time = time.time()

first_policy_time = end_time - start_time


f = open("grid_world_policy_1.csv", "w")
w_1 = csv.writer(f)
for key, val in policy.items():
    w_1.writerow([key, val])
f.close()
# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
'''Approximation algorithm 2'''

# This approximation algorithm performs on the product MDP model.
# For time_horizon N, the state-space of the product MDP model is S x N.
# The costs for each state-action pair is defined as follows:
#   Let c(s) be the posterior goal probability of the state s.
#   Let t be the current time step.
#   Let product_MDP_costs(s,a) be the cost for the state-action pair (s,a), which is used in the linear problem
# We have product_MDP_costs(s,a) = c(s) * discount ** t

product_model, absorb_product, true_goals_product = base_MDP.product_MDP(time_horizon, true_goal)
Product_MDP = MDP(product_model)

start_time = time.time()

# Product_MDP_costs= {}
# for state in Product_MDP.states():
#     for act in Product_MDP.active_actions()[state]:
#         if state not in absorb_product:
#             Product_MDP_costs[(state,act)] = goal_posteriors[0][(state % num_of_states , act)] * discount ** int(state/num_of_states)
#         else:
#             Product_MDP_costs[(state,act)] = 0

# [a,policy] = Product_MDP.compute_min_cost_subject_to_max_reach(init,true_goals_product, absorb_product, Product_MDP_costs)
# f = open("grid_world_policy_2.csv", "w")
# w_2 = csv.writer(f)
# for key, val in policy.items():
#     w_2.writerow([key, val])
# f.close()

end_time = time.time()
second_policy_time = end_time - start_time

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
'''Simulations'''
config_file = "config_grid_world.txt"
f = open(config_file, 'w')
f.write('HEIGHT: '+str(rows)+'\n')
f.write('WIDTH: '+str(cols)+'\n')
f.write('BLOCK: agent '+str(int(init%cols))+ ' '+str(int(rows-1-np.floor(init/cols)))+'\n')
for k in goals:
    f.write('BLOCK: goal '+str(int(k%cols))+ ' '+str(int(rows-1-np.floor(k/cols)))+'\n')
# add obstacles
for i in range(rows):
    for j in range(cols):
        if gridworld.obstacle_mask[i, j]:
            # col, row
            f.write('BLOCK: obstacle '+str(j)+ ' '+str(rows-1-i)+'\n')
f.close()

output_information = {
    'policies': [
        {'time': first_policy_time, 'states': simulate_Stationary("grid_world_policy_1.csv", config_file, rows, cols)},
        # {'time': second_policy_time, 'states': simulate_Markovian("grid_world_policy_2.csv", config_file, time_horizon, rows, cols)}
    ]
}

with open(args['--out'], 'w') as f:
    json.dump(output_information, f)

print("Time for first policy:", first_policy_time)
print("Path length:", len(output_information['policies'][0]['states']))
# print("The time taken for the second policy is: ", end_time - start_time, " seconds")
