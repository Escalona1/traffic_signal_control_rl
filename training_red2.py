import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque  # Replay buffer
from red2 import SharedSubDQNN  # Import your shared sub-network DQN

class DQNsharedbasedNetwork:
    def __init__(self):
        self.dim_input = 36
        self.dim_output = 9
        self.num_lanes = 18
        self.input_per_lane = 2
        self.lr = 0.00005
        self.gamma = 0.2
        self.epsilon = 0.1
        self.epsilon_decay = 1
        self.batch_size = 1048
        self.replay_buffer_size = 100000

        # Initialize networks
        self.policy_network = SharedSubDQNN(self.input_per_lane, self.num_lanes, self.dim_output)
        self.target_network = SharedSubDQNN(self.input_per_lane, self.num_lanes, self.dim_output)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        # Optimizer and loss function
        self.optimizador = optim.Adam(self.policy_network.parameters(), self.lr)
        self.func_perdida = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)

    def select_action(self, estado):
        state = torch.tensor(estado, dtype=torch.float32).view(1, self.num_lanes, self.input_per_lane)
        q_values = self.policy_network(state).view(-1)  # Flatten the output
        if random.random() > self.epsilon:
            accion1 = q_values[0:3].argmax().item()
            accion2 = q_values[3:6].argmax().item()
            accion3 = q_values[6:9].argmax().item()
        else:
            accion1 = random.randint(0, 2)
            accion2 = random.randint(0, 2)
            accion3 = random.randint(0, 2)
        return [accion1, accion2, accion3]

    def train(self, iterator, estado_anterior, estado_actual, r, accion1, accion2, accion3):
        reward = r
        state = torch.tensor(estado_anterior, dtype=torch.float32).view(1, self.num_lanes, self.input_per_lane)
        next_state = torch.tensor(estado_actual, dtype=torch.float32).view(1, self.num_lanes, self.input_per_lane)

        # Add transition to replay buffer
        self.replay_buffer.append((state, accion1, accion2, accion3, reward, next_state))

        # Train if replay buffer has enough samples
        if len(self.replay_buffer) >= self.batch_size:
            batch = random.sample(self.replay_buffer, self.batch_size)
            self.train_batch(batch)

        if iterator % 100 == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.epsilon *= self.epsilon_decay
            
    def train_batch(self, batch):
        # Unpack batch
        states, actions1, actions2, actions3, rewards, next_states = zip(*batch)
        states = torch.cat(states)
        next_states = torch.cat(next_states)
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

        # Compute and backpropagate loss
        loss1 = self.func_perdida(q_values1, target_q1)
        loss2 = self.func_perdida(q_values2, target_q2)
        loss3 = self.func_perdida(q_values3, target_q3)
        total_loss = loss1 + loss2 + loss3
        
        self.optimizador.zero_grad()
        total_loss.backward()
        self.optimizador.step()
