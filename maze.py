# Import libraries
import numpy as np
import mdp

def create_matrices(maze, reward, penalty_s, penalty_l, prob):
    """
    Create reward and transition matrices for input into the mdp QLearning function.
    """
    
    r, c = np.shape(maze)
    states = r*c
    p = prob
    q = (1 - prob)*0.5
    
    path = maze*penalty_s
    walls = (1 - maze)*penalty_l
    combined = path + walls
    
    combined[-1, -1] = reward
            
    R = np.reshape(combined, states)
    
    T_up = np.zeros((states, states))
    T_left = np.zeros((states, states))
    T_right = np.zeros((states, states))
    T_down = np.zeros((states, states))
    
    wall_ind = np.where(R == penalty_l)[0]

    for i in range(states):
        if (i - c) < 0 or (i - c) in wall_ind :
            T_up[i, i] += p
        else:
            T_up[i, i - c] += p
        
        if i%c == 0 or (i - 1) in wall_ind:
            T_up[i, i] += q
        else:
            T_up[i, i-1] += q
        
        if i%c == (c - 1) or (i + 1) in wall_ind:
            T_up[i, i] += q
        else:
            T_up[i, i+1] += q
            
        if (i + c) > (states - 1) or (i + c) in wall_ind:
            T_down[i, i] += p
        else:
            T_down[i, i + c] += p
        
        if i%c == 0 or (i - 1) in wall_ind:
            T_down[i, i] += q
        else:
            T_down[i, i-1] += q
        
        if i%c == (c - 1) or (i + 1) in wall_ind:
            T_down[i, i] += q
        else:
            T_down[i, i+1] += q
            
        if i%c == 0 or (i - 1) in wall_ind:
            T_left[i, i] += p
        else:
            T_left[i, i-1] += p
            
        if (i - c) < 0 or (i - c) in wall_ind:
            T_left[i, i] += q
        else:
            T_left[i, i - c] += q
        
        if (i + c) > (states - 1) or (i + c) in wall_ind:
            T_left[i, i] += q
        else:
            T_left[i, i + c] += q
        
        if i%c == (c - 1) or (i + 1) in wall_ind:
            T_right[i, i] += p
        else:
            T_right[i, i+1] += p
            
        if (i - c) < 0 or (i - c) in wall_ind:
            T_right[i, i] += q
        else:
            T_right[i, i - c] += q
        
        if (i + c) > (states - 1) or (i + c) in wall_ind:
            T_right[i, i] += q
        else:
            T_right[i, i + c] += q
    
    T = [T_up, T_left, T_right, T_down] 
    
    return T, R

maze =  np.array([
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  0.,  1.,  0.],
    [ 0.,  0.,  0.,  1.,  1.,  1.,  0.],
    [ 1.,  1.,  1.,  1.,  0.,  0.,  1.],
    [ 1.,  0.,  0.,  0.,  1.,  1.,  1.],
    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.]]) 
    
T, R = create_matrices(maze, 1, -0.04, -0.75,  0.8)

gamma = 0.9 
alpha = 0.3 
eps = 0.5
decay = 1.0 
iters = 50000 

np.random.seed(1)
q = mdp.QLearning(T, R, gamma, alpha, eps, decay, iters)
q.run()

pol = np.reshape(np.array(list(q.policy)), np.shape(maze))
print(pol)
