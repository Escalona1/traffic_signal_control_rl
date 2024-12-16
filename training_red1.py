import torch
import torch.nn as nn
import torch.optim as optim
import random
from red1 import DQNnetwork
from collections import deque # este es el buffer replay

class DQN:
    '''Parametros red neuronal'''
    def __init__(self):
        self.dim_input = 18 
        self.dim_output = 9
        self.lr = 0.00005
        self.gamma = 0
        self.epsilon = 0.1
        self.epsilon_decay = 1
        self.batch_size = 1048
        self.replay_buffer_size = 100000

        '''Instancias de las redes neuronales'''
        self.policy_network = DQNnetwork(self.dim_input,self.dim_output) # la que se va a modificar mas seguido
        self.target_network = DQNnetwork(self.dim_input,self.dim_output) # la que se modificara mas lento
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        ''' optimizador, perdida y buffer'''
        self.optimizador = optim.Adam(self.policy_network.parameters(), self.lr)
        self.func_perdida = nn.MSELoss()

        self.replay_buffer = deque(maxlen=self.replay_buffer_size)

    def select_action(self,estado):
        state = torch.tensor(estado,dtype=torch.float32)
        q_values = self.policy_network.forward(state)
        rand_num = random.random()
        if rand_num > self.epsilon:
            accion1 = q_values[0:3].argmax().item()
            accion2 = q_values[3:6].argmax().item()
            accion3 = q_values[6:9].argmax().item()
            print('acciones: ', accion1,',',accion2,',',accion3)
        else: 
            accion1 = random.randint(0,2)
            accion2 = random.randint(0,2)
            accion3 = random.randint(0,2)
        
        return [accion1,accion2,accion3]
    
    def train(self, iterator, estado_anterior, estado_actual, r, accion1, accion2, accion3):
        reward = r
        state = torch.tensor(estado_anterior, dtype=torch.float32).view(1, -1)  # Reshape to (1, dim_input)
        next_state = torch.tensor(estado_actual, dtype=torch.float32).view(1, -1)  # Reshape to (1, dim_input)
        print('reward: ', reward)
        print('epsilon: ', self.epsilon)
        # Add to replay buffer
        self.replay_buffer.append((state, accion1, accion2, accion3, reward, next_state))

        if len(self.replay_buffer) >= self.batch_size:
            rand = random.randint(0, len(self.replay_buffer) - self.batch_size)
            batch = list(self.replay_buffer)[rand:rand + self.batch_size]
            self.train_batch(batch)
        
        if iterator % 100 == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.epsilon *= self.epsilon_decay

    def train_batch(self, batch):
        states, actions1, actions2, actions3, rewards, next_states = zip(*batch)

        # Concatenate and reshape states
        states = torch.cat(states).view(-1, self.dim_input)
        next_states = torch.cat(next_states).view(-1, self.dim_input)

        actions1 = torch.tensor(actions1).unsqueeze(1)
        actions2 = torch.tensor(actions2).unsqueeze(1)
        actions3 = torch.tensor(actions3).unsqueeze(1)
        
        rewards = torch.tensor(rewards).unsqueeze(1)

        # Predicted Q-values
        q_values1 = self.policy_network(states).gather(1, actions1)
        q_values2 = self.policy_network(states).gather(1, actions2)
        q_values3 = self.policy_network(states).gather(1, actions3)

        # Target Q-values
        with torch.no_grad():
            next_q_values1 = self.target_network(next_states)[:, 0:3].max(1, keepdim=True)[0]
            next_q_values2 = self.target_network(next_states)[:, 3:6].max(1, keepdim=True)[0]
            next_q_values3 = self.target_network(next_states)[:, 6:9].max(1, keepdim=True)[0]

            target_q1 = rewards + self.gamma * next_q_values1
            target_q2 = rewards + self.gamma * next_q_values2
            target_q3 = rewards + self.gamma * next_q_values3

        loss1 = self.func_perdida(q_values1, target_q1)
        loss2 = self.func_perdida(q_values2, target_q2)
        loss3 = self.func_perdida(q_values3, target_q3)

        total_loss = loss1 + loss2 + loss3

        self.optimizador.zero_grad()
        total_loss.backward()
        self.optimizador.step()
