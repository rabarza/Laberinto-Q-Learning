import numpy as np
from Q_maze import Q_maze, plot_steps_per_episode_comp

# Generacion de un ejemplo de matriz de recompensas (ver excel)
rewards = np.ones((9,9))*-100
rewards[1,0:6] = - 1
rewards[2,3] = -1
rewards[3,1:4] = -1
rewards[4,1] = -1
rewards[5,1:8] = -1
rewards[6,1] = -1
rewards[7,0:2] = -1
rewards[7,3:6] = -1
rewards[3:8,5] = -1
rewards[1:8,7] = -1
rewards[2,8] = 500
# print(rewards)

estado_inicio = [1, 0] # Cambiar estos valores si se quiere evaluar otro estado en el que el agente comience su recorrido.

n_iter = 60
alpha = 0.9
gamma = 0.9
epsilon = 0.1

#Laberinto con paredes de fuego.
exp3 = Q_maze(rewards, episodes = n_iter, discount_rate = gamma, alpha = alpha, method = 'exp3', game='fire-walls') 
exp3.train()

exp3_t = Q_maze(rewards, episodes = n_iter, discount_rate = gamma, alpha = alpha, method = 'exp3', expression='t', game='fire-walls') 
exp3_t.train()

exp3_t2_T = Q_maze(rewards, episodes = n_iter, discount_rate = gamma, alpha = alpha, method = 'exp3', expression='t^2/T', game='fire-walls') 
exp3_t2_T.train()

exp3_ln_t = Q_maze(rewards, episodes = n_iter, discount_rate = gamma, alpha = alpha, method = 'exp3', expression='\log t', game='fire-walls') 
exp3_ln_t.train()

exp3_sqrt_t = Q_maze(rewards, episodes = n_iter, discount_rate = gamma, alpha = alpha, method = 'exp3', expression='\sqrt{t}', game='fire-walls') 
exp3_sqrt_t.train()

softmax_exp_1 = Q_maze(rewards, episodes = n_iter, discount_rate = gamma, alpha = alpha, method = 'softmax', temperature = 0.1, game='fire-walls') 
softmax_exp_1.train()

softmax_exp_2 = Q_maze(rewards, episodes = n_iter, discount_rate = gamma, alpha = alpha, method = 'softmax', temperature = 2, game='fire-walls') 
softmax_exp_2.train()

softmax_exp_3 = Q_maze(rewards, episodes = n_iter, discount_rate = gamma, alpha = alpha, method = 'softmax', temperature = 3, game='fire-walls') 
softmax_exp_3.train()

softmax_exp_4 = Q_maze(rewards, episodes = n_iter, discount_rate = gamma, alpha = alpha, method = 'softmax', temperature = 4, game='fire-walls') 
softmax_exp_4.train()

ucb1 = Q_maze(rewards, episodes = n_iter, discount_rate = gamma, alpha = alpha, method = 'UCB1', game='fire-walls') 
ucb1.train()

e_greedy = Q_maze(rewards, episodes = n_iter, discount_rate = gamma, alpha = alpha, method = 'e-greedy', epsilon=epsilon, game='fire-walls') 
e_greedy.train()
# plot_steps_per_episode_comp([exp3, softmax_exp_1, softmax_exp_2,softmax_exp_3, softmax_exp_4, ucb1], dpi=100)
# plot_steps_per_episode_comp([exp3,exp3_t, exp3_t2_T, ucb1], dpi=100)
# plot_steps_per_episode_comp([exp3,exp3_t, exp3_ln_t, exp3_sqrt_t, exp3_t2_T, ucb1], dpi=100)
plot_steps_per_episode_comp([e_greedy, exp3_ln_t, exp3_sqrt_t, exp3_t2_T, ucb1], dpi=100)
