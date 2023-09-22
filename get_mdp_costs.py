import numpy as np

def get_exaggeration_mdp_costs(MDP, goals, init, discount, prior_goals, absorb):
    rewards = [{} for _ in range(len(goals))]
    for k in range(len(goals)):
        for state in MDP.states():
            for act in MDP.active_actions()[state]:
                if state != goals[k]:
                    rewards[k][(state,act)] = -60
                else:
                    rewards[k][(state,act)] = 0

    goal_values = []
    for k in range(len(goals)):
        goal_values.append(MDP.soft_max_val_iter(rewards[k], [goals[k]], discount))

    goal_posteriors = [{} for _ in range(len(goals))]
    for state in MDP.states():
        for act in MDP.active_actions()[state]:
            for k in range(len(goals)):
                goal_posteriors[k][(state,act)] = np.exp(goal_values[k][state]-goal_values[k][init])*prior_goals[k]
            denom = sum(goal_posteriors[i][(state,act)] for i in range(len(goals)))
            for k in range(len(goals)):
                goal_posteriors[k][(state,act)] = goal_posteriors[k][(state,act)]/ np.sum(denom)
    
    MDP_costs = {}
    # Creates "exaggeration" deception
    for state in MDP.states():
        for act in MDP.active_actions()[state]:
            if state not in absorb:
                most_probable_decoy = np.amax([goal_posteriors[i][(state,act)] for i in range(1,len(goals))])
                deception_index =  most_probable_decoy - goal_posteriors[0][(state,act)]
                #deception_index = most_probable_decoy
                MDP_costs[(state,act)] = (1 - deception_index)
            else:
                MDP_costs[(state,act)] = 0
    
    return MDP_costs
