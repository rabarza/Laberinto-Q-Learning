import numpy as np
import pandas as pd
import random
from  Q_maze import Q_maze

def plot_steps_per_episode_comp(lista):
    '''
    Realiza una comparación gráfica de la cantidad de pasos que tardó cada agente en un episodio en llegar a un estado terminal.
    '''
    import matplotlib.pyplot as plt

    plt.figure(dpi=100)

    for model in lista:
        plt.plot(range(model.episodes),model.steps, label=model.game+' | '+model.method)
        plt.legend(loc='upper right')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.grid()
    plt.show()


def create_maze(rows, cols):
    maze = [[-100 for _ in range(cols)] for _ in range(rows)] # Inicializamos el laberinto con todas las celdas como paredes (-100)
    visited = [[False for _ in range(cols)] for _ in range(rows)] # Creamos una matriz de visitados para llevar un seguimiento de los nodos visitados
    start_row, start_col = 0, 0 # La posición inicial del laberinto es (0,0)
    visited[start_row][start_col] = True # Marcamos el nodo inicial como visitado

    while not all(all(visited[row][col] for col in range(cols)) for row in range(rows)): # Mientras no hayamos visitado todos los nodos del laberinto
        neighbors = [] # Creamos una lista de vecinos
        if start_row > 0 and not visited[start_row - 1][start_col]: # Si podemos movernos hacia arriba y el nodo no ha sido visitado
            neighbors.append((start_row - 1, start_col)) # Añadimos el nodo vecino a la lista de vecinos
        if start_row < rows - 1 and not visited[start_row + 1][start_col]: # Si podemos movernos hacia abajo y el nodo no ha sido visitado
            neighbors.append((start_row + 1, start_col)) # Añadimos el nodo vecino a la lista de vecinos
        if start_col > 0 and not visited[start_row][start_col - 1]: # Si podemos movernos hacia la izquierda y el nodo no ha sido visitado
            neighbors.append((start_row, start_col - 1)) # Añadimos el nodo vecino a la lista de vecinos
        if start_col < cols - 1 and not visited[start_row][start_col + 1]: # Si podemos movernos hacia la derecha y el nodo no ha sido visitado
            neighbors.append((start_row, start_col + 1)) # Añadimos el nodo vecino a la lista de vecinos
        
        if neighbors: # Si hay vecinos disponibles
            row, col = random.choice(neighbors) # Elegimos uno al azar
            visited[row][col] = True # Marcamos el nodo como visitado
            maze[row][col] = -1 # Establecemos el valor de la celda a -1 para indicar que es un pasillo
            start_row, start_col = row, col # Movemos nuestra posición al nodo vecino elegido al azar
        else: # Si no hay vecinos disponibles
            # Retrocedemos al último nodo visitado que tenga vecinos disponibles
            for row in range(rows):
                for col in range(cols):
                    if visited[row][col]:
                        if row > 0 and not visited[row - 1][col]:
                            start_row, start_col = row - 1, col
                            break
                        if row < rows - 1 and not visited[row + 1][col]:
                            start_row, start_col = row + 1, col
                            break
                        if col > 0 and not visited[row][col]:
                            start_row, start_col = row, col - 1
                            break
                        if col < cols - 1 and not visited[row][col + 1]:
                            start_row, start_col = row, col + 1
                            break
                        else: # Si no encontramos ningún nodo visitado con vecinos disponibles
                            # Escogemos un nodo aleatorio como punto de partida
                            start_row, start_col = random.randint(0, rows - 1), random.randint(0, cols - 1)
                            visited[start_row][start_col] = True

    # Establecemos la meta del laberinto en una celda aleatoria
    goal_row, goal_col = random.randint(0, rows - 1), random.randint(0, cols - 1)
    maze[goal_row][goal_col] = 500 # Establecemos el valor de la celda a 500 para indicar que es la meta

    return maze

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
print(rewards)


### Probando el modelo...###

# model_pit = Q_maze(rewards, episodes = 1000, discount_rate = 1, alpha = 0.9, method = 'e-greedy', epsilon = 0.1, temperature=1, game='pit-walls')
# model_pit.train()
# print('Modelo Laberinto con fosas Entrenado!')
# print(f'El mejor camino para llegar a la meta comenzando desde {estado_inicio} es {model_pit.mejor_camino(estado_inicio)}\n')
# model_pit.plot_steps_per_episode()

model_fire = Q_maze(rewards, episodes = 1000, discount_rate = 1, alpha = 0.9, method = 'en-greedy', epsilon = 0.1, temperature=1, game='fire-walls', c = 5, d = 0.8) 
model_fire.train()
print('Modelo Laberinto con paredes de fuego Entrenado!')
# print(f'El mejor camino para llegar a la meta comenzando desde {estado_inicio} es {model_fire.mejor_camino(estado_inicio)}\n')
model_fire.plot_steps_per_episode()

# print(min(model_pit.steps))
# print(min(model_fire.steps))


n_matrices = 10
n_episodios = 500
estado_inicio = [0, 0] # Cambiar estos valores si se quiere evaluar otro estado en el que el agente comience su recorrido. [1, 0]
n_cols = 10
n_rows = 10
mazes = [np.array(create_maze(n_cols, n_rows)) for i in range(n_matrices)]

models = []
for maze in mazes:
    model_fire = Q_maze(maze, episodes = n_episodios, discount_rate = 0.9, alpha = 0.9, method = 'softmax', game='fire-walls') 
    model_fire.train()
    models.append(model_fire)
#Laberinto con paredes de fuego. Notar que se esta utilizando el metodo UCB1, y no e-greedy. Cambiar el parametro a method = 'e-greedy' si lo desea.

# Graficos de convergencia

plot_steps_per_episode_comp(models)
