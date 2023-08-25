import numpy as np
from grid_world import grid_world
from random import randint,seed
from MDP_class import MDP
from graph import Graph
from Simulations import simulate_Markovian, simulate_Stationary
import csv
import time

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
'''Environment Specifications'''

# Define the grid world environment
from grid_world_michael import load_grid, OBSTACLE_TILE_TYPE

gridworld = load_grid("../robotic-deception/Pursuit/gridworlds/square_rect_based/8x8C.tmx")
rows, cols = gridworld.tiles.shape

def row_and_column_to_flat_index(cols, row_number, column_number):
    return row_number * cols + column_number

goals = [row_and_column_to_flat_index(cols, row, col) for (row, col) in gridworld.goal_positions]
# Find obstacle positions
absorb = []
for i in range(rows):
    for j in range(cols):
        if gridworld.obstacle_mask[i, j]:
            absorb.append(row_and_column_to_flat_index(cols, i, j))
absorb.extend(goals)
true_goal = goals[0]

# rows, cols = 7,7
# goals = [42,45,48]
# true_goal = goals[0] # NOTE: Always choose the first element of the goals array as the true goal
# absorb = goals # these include obstacles

# row, column = 17,17
# goals = [288,16,6,176]
# absorb = set([5, 5+17, 5+ 3*17, 5+ 4*17, 5 + 5*17, 5 + 6*17, 5 + 7*17])
# absorb.update(set([5+ 9*17, 5+ 10*17, 5 + 11*17, 5 + 12*17, 5 + 13*17, 5 + 15*17, 5 + 16*17]))
# absorb.update( set([11, 11+17, 11+ 3*17, 11+ 4*17, 11 + 5*17, 11 + 6*17, 11 + 7*17]))
# absorb.update(set([11+ 9*17, 11+ 10*17, 11 + 11*17, 11 + 12*17, 11 + 13*17, 11 + 15*17, 11 + 16*17]))
# absorb.update(set([85,86,88,89,90,91,92, 94,95,96,97,98, 100, 101]))
# absorb.update(set([187,188,190,191,192,193,194,196,197,198,199,200,202,203]))
# absorb.update(set(goals))

init = 0
slip = 0
# Defines the motion model
model = grid_world(rows,cols,absorb,slip)
# model = (model_state[key] = [actions], model_state_action[(key,action)] = ([probabilities],[successors]))
num_of_states = len(model[0])

# Define the time_horizon
time_horizon = 180

# Define the discount factor
discount = 0.9

# Define the cost of taking a step in the MDP (used in the shortest path computation)
one_step_cost = 30

# Define the prior distribution on the goals
prior_goals = np.ones((len(goals),1)) / len(goals)

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
'''Define the posterior goal probabilities for each state'''

# Perform the computations on the base MDP model
base_MDP = MDP(model)

goal_posteriors = base_MDP.compute_goal_posteriors(prior_goals, init, goals, one_step_cost, discount)

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

graph = Graph(model)
min_times = graph.Dijkstra(init)
base_MDP_costs= {}
for state in base_MDP.states():
    for act in base_MDP.active_actions()[state]:
        if state not in absorb:
            base_MDP_costs[(state,act)] = goal_posteriors[0][(state,act)] * discount**min_times[state]
        else:
            base_MDP_costs[(state,act)] = 0

[a,policy] = base_MDP.compute_min_cost_subject_to_max_reach(init,[true_goal], absorb, base_MDP_costs)

end_time = time.time()

print("The time taken for the first policy is: ", end_time - start_time, " seconds")

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

Product_MDP_costs= {}
for state in Product_MDP.states():
    for act in Product_MDP.active_actions()[state]:
        if state not in absorb_product:
            Product_MDP_costs[(state,act)] = goal_posteriors[0][(state % num_of_states , act)] * discount ** int(state/num_of_states)
        else:
            Product_MDP_costs[(state,act)] = 0

[a,policy] = Product_MDP.compute_min_cost_subject_to_max_reach(init,true_goals_product, absorb_product, Product_MDP_costs)
f = open("grid_world_policy_2.csv", "w")
w_2 = csv.writer(f)
for key, val in policy.items():
    w_2.writerow([key, val])
f.close()

end_time = time.time()

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
f.close()

# Simulate the first policy
#policy_file = "grid_world_policy_1.csv"
#simulate_Stationary(policy_file, config_file, row, column)

# Simulate the second policy
policy_file = "grid_world_policy_2.csv"
simulate_Markovian(policy_file, config_file, time_horizon, rows, cols)

print("The time taken for the second policy is: ", end_time - start_time, " seconds")
