import numpy as np

class Q_maze():
    def __init__(self, rewards, episodes=1000, discount_rate=0.9, alpha=0.2, method = 'e-greedy', epsilon = 0.1, game = 'fosa'):

        self.episodes = episodes
        self.alpha = alpha
        self.discount_rate = discount_rate
        self.method = method
        self.epsilon = epsilon
        self.rewards = rewards
        self.game = game
        self.actions = np.array([0, 1, 2, 3]) #{0:'izquierda', 1:'arriba', 2:'derecha', 3:'abajo'}
        self.times_actions = np.zeros((rewards.shape[0], rewards.shape[1], 4))#Se cuenta la cantidad de veces que se tomo una accion en cada estado
        self.q_table = None

    def chequear_estado_terminal(self, state):

        if (self.rewards[state[0],state[1]] != -1):
            return True
        else:
            return False
            
    def estado_inicial(self): #Funcion que retorna el estado inicial del objeto.
        dims = self.rewards.shape
        estado = np.random.randint(dims[0]), np.random.randint(dims[1]) #Genera una posicion aleatoria dentro de la matriz de recompensas

        while self.chequear_estado_terminal(estado):#Mientras la ubicacion sea un estado terminal, genero una nueva.
            estado = np.random.randint(dims[0]), np.random.randint(dims[1])
        return estado



    def select_action(self, estado, epsilon, method):
        if method == 'e-greedy':
            greedy_action = np.argmax(self.q_table[estado[0],estado[1]])
            explore_action = np.random.choice(self.actions)
            num = np.random.random()

            if num < epsilon:
                while explore_action == greedy_action:
                    explore_action = np.random.choice(self.actions)    
                return explore_action #No la mejor accion
            else:
                return greedy_action #Mejor accion

        elif method == 'UCB1':

            times_actions = self.times_actions[estado[0],estado[1]]#Contiene la cantidad de veces que se ha tomado cada accion en el estado indicado. (es un arreglo)

            number_not_choosen_actions = np.count_nonzero(times_actions==0) #Cuenta la cantidad de acciones que nunca se han escogido
            
            if (number_not_choosen_actions > 0):
                action = np.argwhere(times_actions==0).flatten()[0] #Devuelve la primera accion que nunca se ha usado, podria tambien escogerse aleatoriamente. Hablarlo con el profe.
                self.times_actions[estado[0],estado[1],action] +=1
                return action

            else:

                qa = self.q_table[estado[0],estado[1],0]
                qb = self.q_table[estado[0],estado[1],1]
                qc = self.q_table[estado[0],estado[1],2]
                qd = self.q_table[estado[0],estado[1],3]

                ucba =  qa+ np.sqrt(2*np.log(self.episodes)/(times_actions[0]))
                ucbb =  qb+ np.sqrt(2*np.log(self.episodes)/(times_actions[1]))
                ucbc =  qc+ np.sqrt(2*np.log(self.episodes)/(times_actions[2]))
                ucbd =  qd+ np.sqrt(2*np.log(self.episodes)/(times_actions[3]))

                ucb = np.array([ucba, ucbb, ucbc, ucbd])

                action = np.argmax(ucb)

                self.times_actions[estado[0],estado[1],action] +=1 #Se suma una vez mas el numero de veces que la accion asignada 

                return action
        else:
            raise ValueError('El método seleccionado debe ser valido (como e-greedy o UCB-1)')


    def next_state(self, state, action):
        '''Devuelve en una lista los valores del estado siguiente luego de realizar una acción a partir de un estado previo.


        Parámetros:

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

        return s_row, s_column

    def train(self):
        '''Resuelve el problema del laberinto usando el algoritmo Q-Learning'''
        episodes = self.episodes
        epsilon = self.epsilon
        alpha = self.alpha
        gamma = self.discount_rate
        rewards = self.rewards

        self.q_table = np.zeros((rewards.shape[0], rewards.shape[1], 4))# Inicializacion de la tabla de valores Q para cada par estado-accion (comienzan todos con cero)

        for episode in range(episodes):

            state = self.estado_inicial()

            while not self.chequear_estado_terminal(state):

                action = self.select_action(state, epsilon, self.method)

                next_state = self.next_state(state, action)
                reward = rewards[next_state[0], next_state[1]]# La recompensa depende del par estado/accion (transicion).
                
                Q_old = self.q_table[state[0], state[1], action]

                Q_new = Q_old * (1-alpha) + alpha * (reward + gamma * np.max(self.q_table[next_state[0], next_state[1]]))

                self.q_table[state[0], state[1], action] = Q_new

                state = next_state
        
        return self.q_table

    def mejor_camino(self, state):
        '''Devuelve una lista que contiene todas las ubicaciones que se deben recorrer para llegar desde un estado a la solución del laberinto realizando la menor cantidad de pasos.
        
        Parámetros:

        `state` -- estado (debe ser una lista que contenga las cordenadas x, y. ej: [x, y])
        '''
        camino = []
        if self.chequear_estado_terminal(state):
            return camino

        else:
            
            camino.append(state)

        while not self.chequear_estado_terminal(state):
            action = self.select_action(state, 0, 'e-greedy') #Escojo un epsilon = 0 para que siempre escoja la mejor accion
            state = self.next_state(state, action)
            camino.append(state)
        return camino    

# Generacion de un ejemplo (ver excel)
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



model = Q_maze(rewards, episodes = 10000, discount_rate = 0.9, alpha = 0.9, method = 'UCB1', epsilon = 0.1)
model.train()

print('Modelo Entrenado')

print(model.mejor_camino([1, 5]))

