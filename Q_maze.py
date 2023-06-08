import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import numpy as np

class Q_maze():
    '''
    Resuelve laberintos empleando el algoritmo Q-Learning. Se consideran dos casos. El primer caso considera un laberinto compuestos por fosas, es decir, cuando el agente cae en una fosa se muere y finaliza el episodio, notar que tambien finaliza cuando llega a la meta. El segundo caso es un laberinto con paredes de fuego, el fuego no mata al agente, pero le hace daño al quemarlo, por lo que recibe un castigo cada vez que toca el fuego, el episodio termina exclusivamente cuando el agente llega a la meta, sin importar cuánto se haya quemado. 


    #### Parámetros:

    `rewards` --  matriz de recompensas que compone la estructura del laberinto.

    `episodes` -- cantidad de iteraciones que se empleará en el algoritmo Q-Learning.

    `discount_rate` -- tasa de descuento.
    
    `method` -- estrategia de exploracion-explotacion, puede ser e-greedy o UCB1.

    `epsilon` -- valor de epsilon que se emplea cuando se utiliza una estrategia e-greedy (por defecto epsilon = 0.1).

    `game` -- tipo de laberinto compuesto por fosas, o por paredes de fuego (pit-walls o fire-walls, respectivamente).

    '''
    def __init__(self, rewards, episodes=1000, discount_rate=0.9, alpha=0.2, method = 'e-greedy', epsilon = 0.1, temperature = 1, game = 'pit-walls', c = 2, d = 0.8, expression='t^2'):

        self.episodes = episodes
        self.alpha = alpha
        self.discount_rate = discount_rate
        self.method = method
        self.epsilon = epsilon
        self.tau = temperature
        self.rewards = rewards
        self.game = game

        self.actions = np.array([0, 1, 2, 3]) #{0:'izquierda', 1:'arriba', 2:'derecha', 3:'abajo'}
        
        #Se cuenta la cantidad de veces que se tomo una accion en cada estado
        self.times_actions = np.zeros((rewards.shape[0], rewards.shape[1], 4))

        #Se cuenta la cantidad de veces que se visita un estado
        self.times_states = np.zeros((rewards.shape[0], rewards.shape[1]))

        self.action_weights = np.ones(len(self.actions))#L U R D Inicializo pesos de cada accion (Exp3)
        self.action_prob = np.zeros(len(self.actions))

        # en-greedy
        self.c = c
        self.d = d

        # exp3
        self.expression = expression

        self.steps = np.zeros(episodes)

    def estado_terminal(self, state):
        '''Devuelve True o False si un estado es terminal o no, respectivamente
        
        #### Parametros:
        `state` -- estado a evaluar si es absorbente o no.      
        
        '''
        #No se puede traspasar las paredes (aquellas ubicaciones que tienen un casigo de -100), ya que representan un estado terminal (absorbente). Ej: Chocar con una pared te mata.
        if self.game == 'pit-walls':
            if (self.rewards[state[0],state[1]] != -1):
                return True
            else:
                return False
            
        #Las ubicaciones que tienen un castigo de -100 no representan un estado terminal (absorbente). Ej: Chocar con una pared te quema.
        elif self.game == 'fire-walls':
            if (self.rewards[state[0],state[1]] == 500):
                return True
            else:
                return False
        else:
            raise ValueError('El tipo de juego debe ser valido (pit-walls o fire-walls).')
            
    def estado_inicial(self, fixed = True, estado = [1,0]):
        ''' Devuelve el estado inicial en el cual el agente comienza a recorrer el laberinto
        
        
        #### Parámetros:
        
        `fixed` -- True o False, indica si se parte de un estado fijo o aleatorio, respectivamente.
        Por defecto se comienza a partir de la celda [1, 0]

        '''

        if not fixed:
            # Generar una posicion aleatoria dentro de la matriz de recompensas
            dims = self.rewards.shape
            estado = np.random.randint(dims[0]), np.random.randint(dims[1]) 
            
            # Generar una nueva ubicación mientras la ubicacion sea un estado terminal.
            while self.estado_terminal(estado):
                estado = np.random.randint(dims[0]), np.random.randint(dims[1])
            return estado
        else:
            return estado

    def next_state(self, state, action):
        '''
        Devuelve en una lista los valores del estado siguiente luego de realizar una acción a partir de un estado previo.


        #### Parámetros:

        `state` -- estado previo a realizar la transicion.

        `action` -- accion que se toma para realizar la transicion, puede ser arriba (1), abajo (3), izquierda (0), o derecha (4).
        
        '''
        dims = self.rewards.shape
        s_row, s_column = state
        if action == 0 and s_column > 0: #Se mueve hacia la izquierda
            s_column -= 1

        elif action == 1 and s_row > 0: #Se mueve hacia arriba
            s_row -= 1

        elif action == 2 and s_column + 1 < dims[1]: #Se mueve hacia la derecha
            s_column += 1

        elif action == 3 and s_row + 1 < dims[0]: #Se mueve hacia abajo
            s_row += 1

        self.times_states[s_row,s_column] +=1
        return s_row, s_column

    def select_action(self, state, method, epsilon = float(0), epoch = 1):
        '''Devuelve una acción de acuerdo a la estrategia de selección de acción.
        
        #### Parámetros:

        `state` -- estado desde el cual se realiza la accion.

        `method` --  método que se usará para explorar-explotar.

        `epsilon` -- solo se utiliza en el método epsilon-greedy (por defecto epsilon = 0).

        '''
        def epsilon_greedy(self, state, epsilon):
            greedy_action = np.argmax(self.q_table[state[0], state[1]])
            explore_action = np.random.choice(self.actions)

            num = np.random.random()
            if num <= epsilon:
                # Explorar
                return explore_action
            else:
                # Explotar
                return greedy_action
                
        def decreasing_epsilon_greedy(self, state):
            K = len(self.actions)
            c = self.c
            d = self.d

            # Incrementar cantidad de visitas al estado
            self.times_states[state[0], state[1]]+=1
            t = self.times_states[state[0], state[1]]

            greedy_action = np.argmax(self.q_table[state[0],state[1]])
            explore_action = np.random.choice(self.actions)
            
            # Actualizar valor de la tasa epsilon_n decreciente
            epsilon_n = min(1, (c * K) / ((d ** 2) * t))

            # Seleccionar acción
            num = np.random.random()
            if num <= epsilon_n:
                # Explorar
                return explore_action
            else:
                # Explotar
                return greedy_action
            
        def softmax_exploration(self, state, normalize=True):            
            # Incrementar cantidad de visitas al estado
            self.times_states[state[0],state[1]]+=1

            # Actualizar el valor de temperatura
            tau = self.tau

            q_state = self.q_table[state[0], state[1], :]

            if normalize:
                exp_values = np.exp((q_state - np.max(q_state)) * tau)
            else:
                exp_values = np.exp(q_state * tau)

            # Generar distribucion de probabilidad exponencial 
            probabilities = exp_values /  np.sum(exp_values) 
            
            # Muestrear acción de acuerdo a la distribución generada
            action = random.choices(list(self.actions), weights=list(probabilities))[0]
            return action
            
        def upper_confidence_bound(self, state, c = 2):
            # Contador de visitas para cada accion en el estado actual
            times_actions = self.times_actions[state[0], state[1]]
            
            # Contar la cantidad de acciones que no han sido escogidas en el estado actual
            number_not_chosen_actions = np.count_nonzero(times_actions == 0) 

            if (number_not_chosen_actions > 0):
                # Escoger una acción no visitada anteriormente
                action = np.argwhere( times_actions == 0 ).flatten()[0]
                
                # Incrementar el contador de visitas para la acción en el estado actual
                self.times_actions[state[0], state[1], action] += 1

                # Devolver acción
                return action

            else:
                # Obtener los valores de q para cada acción a partir del estado s
                q_state = self.q_table[state[0], state[1], :]

                # Calcular el valor de los estimadores de q utilizando la estrategia UCB
                ucb =  q_state + np.sqrt( c * np.log(self.episodes) / times_actions[0] )

                # Seleccionar acción
                action = np.argmax(ucb)

                # Incrementar el contador de visitas para la acción en el estado actual
                self.times_actions[state[0], state[1], action] += 1

                # Devolver acción
                return action

        def exp3_action_selection(self, state, normalize=True):

            # Incrementar cantidad de visitas al estado
            self.times_states[state[0], state[1]] += 1
            t = self.times_states[state[0], state[1]]

            # Actualizar el valor de temperatura decreciente
            match self.expression:
                case 't':
                    eta =  t
                case 't/T':
                    eta =  t / self.episodes
                case 't^2':
                    eta =  t**2
                case 't^2/T':
                    eta =  t**2 / self.episodes
                case 't^3':
                    eta = t**3
                case '\log t':
                    eta = np.log(t)
                case '\sqrt{t}':
                    eta = np.sqrt(t)
                case _:
                    raise ValueError("Expresión inválida para eta.")
            
            # Obtener los valores de q para cada acción a partir del estado s
            q_state = self.q_table[state[0], state[1], :]

            # Valores exponenciales de Q(s,a)
            if normalize:
                exp_values = np.exp((q_state - np.max(q_state)) * eta)
            else:
                exp_values = np.exp(q_state * eta)

            # Generar distribucion de probabilidad exponencial 
            probabilities = exp_values /  np.sum(exp_values) 
            # print(probabilities)

            # Muestrear acción de acuerdo a la distribución generada
            action = random.choices(list(self.actions), weights=list(probabilities))[0]
            return action

        match method:
            case 'e-greedy':
                return epsilon_greedy(self, state, epsilon)

            case 'en-greedy':
                return decreasing_epsilon_greedy(self, state)
                
            case 'UCB1':
                return upper_confidence_bound(self, state)

            case 'softmax':
                return softmax_exploration(self, state)
            
            case 'exp3':
                return exp3_action_selection(self, state)
            
            case _:
                raise ValueError('El método seleccionado debe ser valido (como e-greedy, en-greedy, UCB1, softmax)')
            
    def train(self):
        '''
        Resuelve el problema del laberinto usando el algoritmo Q-Learning
        '''

        episodes = self.episodes
        epsilon = self.epsilon
        alpha = self.alpha
        gamma = self.discount_rate
        rewards = self.rewards

        # Inicializar de valores Q: Q_table (all values to zero)
        self.q_table = np.zeros((rewards.shape[0], rewards.shape[1], 4))

        for episode in range(episodes):
            
            # seleccionar un estado inicial (fixed = False, selecciona un estado aleatorio)
            state = self.estado_inicial(fixed = True)
            
            while not self.estado_terminal(state):

                # Se realiza la transición (action, next_state, reward)
                action = self.select_action(state, self.method, epsilon, episode)
                next_state = self.next_state(state, action)
                reward = rewards[next_state[0], next_state[1]]
                
                # Actualizar valores Q_table
                Q_old = self.q_table[state[0], state[1], action]
                Q_new = Q_old * (1-alpha) + alpha * (reward + gamma * np.max(self.q_table[next_state[0], next_state[1]]))
                self.q_table[state[0], state[1], action] = Q_new

                # Ir al estado siguiente
                state = next_state

                # Aumentar la cantidad de pasos del episodio
                self.steps[episode] += 1

        # return self.q_table

    def best_path(self, state):
        '''Devuelve una lista que contiene todas las ubicaciones que se deben recorrer para llegar desde un estado a la solución del laberinto realizando la menor cantidad de pasos.
        
        Parámetros:

        `state` -- estado (debe ser una lista que contenga las cordenadas x, y. ej: [x, y])
        '''
        camino = []
        if self.estado_terminal(state):
            return camino

        else:
            camino.append(state)

        while not self.estado_terminal(state):
            action = self.select_action(state,'e-greedy') #Escojo un epsilon = 0 para que siempre escoja la mejor accion, por defecto es 0, por lo tanto no es necesario indicarlo
            state = self.next_state(state, action)
            camino.append(state)
        return camino    

    def plot_steps_per_episode(self):
        '''
        Grafica la cantidad de pasos que tardó cada episodio en llegar a un estado terminal.
        '''
        import matplotlib.pyplot as plt

        plt.figure(dpi=100)
        plt.plot(range(self.episodes),self.steps)
        plt.title(self.game+'-'+self.method)
        plt.xlabel('Episodes')
        plt.ylabel('Steps')

        # plt.yticks(range(0,int(np.max(self.steps)),10))
        plt.grid()
        plt.show()


#=========================================================== Comparación de modelos ===========================================================
def plot_steps_per_episode_comp(lista, dpi = 50):
    '''
    Realiza una comparación gráfica de la cantidad de pasos que tardó cada agente en un episodio en llegar a un estado terminal.
    '''

    plt.figure(dpi=dpi)

    for model in lista:
        if model.method in ['softmax']:
            plt.plot(range(model.episodes),model.steps, label=model.game+ ' | ' + model.method + ' |  $\eta$ = '+ str(model.tau))
        
        elif model.method in ['exp3']:
            plt.plot(range(model.episodes),model.steps, label=model.game+ ' | ' + model.method + ' |  $\eta = ' + str(model.expression) + '$')

        elif model.method in ['e-greedy', 'en-greedy']:
            plt.plot(range(model.episodes),model.steps, label=model.game+ ' | ' + model.method + ' | $\epsilon = $' + str(model.epsilon))
        
        elif model.method in ['UCB1']:
            plt.plot(range(model.episodes),model.steps, label=model.game+' | ' + model.method +' | $c = $' + str(model.c))
            
        plt.legend(loc='upper right')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.grid()
    plt.show()

def generate_maze(rows, cols, start, end):
    # Inicializar todas las celdas como paredes (-100)
    maze = [[-100] * cols for _ in range(rows)]  

    # Definir punto de inicio y meta
    maze[start[0]][start[1]] = -1
    # maze[end[0]][end[1]] = 500 

    # Utilizaremos un stack para rastrear las celdas visitadas
    stack = [start]  
    # Conjunto para almacenar las celdas visitadas
    visited = set([start])  
    
    # Mientras el stack no esté vacío
    while stack:
        current_row, current_col = stack[-1]

        # Obtener celdas vecinas no visitadas
        unvisited_neighbours = []
        if current_row > 1 and (current_row - 2, current_col) not in visited:
            unvisited_neighbours.append((current_row - 2, current_col))
        if current_row < rows - 2 and (current_row + 2, current_col) not in visited:
            unvisited_neighbours.append((current_row + 2, current_col))
        if current_col > 1 and (current_row, current_col - 2) not in visited:
            unvisited_neighbours.append((current_row, current_col - 2))
        if current_col < cols - 2 and (current_row, current_col + 2) not in visited:
            unvisited_neighbours.append((current_row, current_col + 2))

        if unvisited_neighbours:
            # Elegir una celda vecina aleatoriamente
            next_row, next_col = random.choice(unvisited_neighbours)

            # Eliminar la pared entre la celda actual y la vecina
            maze[next_row][next_col] = -1
            maze[(current_row + next_row) // 2][(current_col + next_col) // 2] = -1

            visited.add((next_row, next_col))
            stack.append((next_row, next_col))

            if (next_row == end[0] and abs(next_col - end[1]) == 1) or (next_col == end[1] and abs(next_row - end[0]) == 1):
                # Si la celda vecina es adyacente a la meta, terminar el algoritmo
                break

        else:
            # No hay celdas vecinas no visitadas, retroceder
            stack.pop()
    
    # Definir meta
    maze[end[0]][end[1]] = 500 

    
    return np.matrix(maze)


def plot_maze(maze, start_point, end_point, path = []):

    rows = maze.shape[0]
    cols = maze.shape[1]

    # Crear una figura y un eje
    fig, ax = plt.subplots()

    # Configurar el tamaño de la figura en función del tamaño del laberinto
    fig.set_size_inches(cols/2, rows/2)

    # Configurar límites del eje
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)

    # Ocultar ejes
    ax.set_axis_off()

    # Dibujar las paredes
    for row in range(rows):
        for col in range(cols):
            if maze[row,col] == -100:
                rect = Rectangle((col, rows - row - 1), 1, 1, facecolor="black")
                ax.add_patch(rect)

    # Dibujar el camino
    if (path):
        for cell in path:
            path_rect = Rectangle((cell[1], rows - cell[0] - 1), 1, 1, facecolor="palegreen")
            ax.add_patch(path_rect)
            
    # Dibujar el punto de inicio
    start_row, start_col = start_point
    start_rect = Rectangle((start_col, rows - start_row - 1), 1, 1, facecolor="red")
    ax.add_patch(start_rect)

    # Dibujar la meta
    end_row, end_col = end_point
    end_rect = Rectangle((end_col, rows - end_row - 1), 1, 1, facecolor="lime")
    ax.add_patch(end_rect)

    # Mostrar la figura
    plt.show()