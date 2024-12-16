import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class SharedPolicyValueNetwork(nn.Module):
    def __init__(self, input_dim, num_junctions, actions_per_junction):
        super(SharedPolicyValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.policy_head = nn.Linear(64, num_junctions * actions_per_junction)
        self.value_head = nn.Linear(64, 1)
        self.num_junctions = num_junctions
        self.actions_per_junction = actions_per_junction

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy_logits = self.policy_head(x).view(-1, self.num_junctions, self.actions_per_junction)
        policy_logits = policy_logits - policy_logits.max(dim=-1, keepdim=True)[0]
        policy_probs = nn.Softmax(dim=-1)(policy_logits)
        value = self.value_head(x)

        return policy_probs, value

class PPOAgent:
    def __init__(self, input_dim, num_junctions, actions_per_junction, lr=3e-4, gamma=0.2, clip_epsilon=0.2):
        self.policy_network = SharedPolicyValueNetwork(input_dim, num_junctions, actions_per_junction)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_loss_fn = nn.MSELoss()
        self.entropy_coef = 0.1 

    def policy(self, states):
        
        policy_probs, values = self.policy_network(states)
        return policy_probs, values

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        returns = []
        adv = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + (self.gamma * next_values[i] * (1 - dones[i])) - values[i]
            adv = delta + (self.gamma * adv)
            advantages.insert(0, adv)
            returns.insert(0, adv + values[i])
        return torch.tensor(advantages), torch.tensor(returns)

    def update(self, states, actions, old_log_probs, advantages, returns):
    
        policy_probs, values = self.policy_network(states)
        dist = Categorical(probs=policy_probs)
        new_log_probs = dist.log_prob(actions)
        ratios = torch.exp(new_log_probs - old_log_probs)

        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(-1)  

        clipped_ratios = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()


        value_loss = self.value_loss_fn(values.squeeze(-1), returns)

        entropy = dist.entropy().mean()


        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
        if self.entropy_coef > 0.01:
            self.entropy_coef -= 0.000001 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()
