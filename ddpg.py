import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, dim_states, dim_actions, action_high, hidden_dim = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(dim_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dim_actions)
        
        self.action_high = action_high
        
    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        
        return x * self.action_high
    
class QNetwork(nn.Module):
    def __init__(self, dim_states, dim_actions, hidden_dim = 256):
        super().__init__()
        
        self.fc1 = nn.Linear(dim_states + dim_actions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        
        x = torch.cat([state, action], dim = 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return self.fc3(x) 
    
    
class DDPGAgent():
    def __init__(self, dim_states, dim_actions, action_high, action_low, 
                 actor_lr = 1e-3, Q_lr = 1e-2, gamma = 0.99, tau = 0.9, Q_updates = 1):
        
        # actor and critic
        self.actor = Actor(dim_states, dim_actions, action_high)
        self.Q = QNetwork(dim_states, dim_actions)
        
        # actor and critic targets
        self.actor_target = Actor(dim_states, dim_actions, action_high)
        self.Q_target = QNetwork(dim_states, dim_actions)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.Q_target.load_state_dict(self.Q.state_dict())
        
        # optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr = actor_lr)
        self.Q_optimizer = Adam(self.Q.parameters(), lr = Q_lr)
        
        self.action_high = action_high
        self.action_low = action_low
        self.gamma = gamma
        self.tau = tau
        self.Q_updates = Q_updates
        
        self.actor_loss = []
        self.Q_loss = []
        
    def select_action(self, state, action_noise = 0.2):
        state = torch.tensor(state).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state).squeeze(0)
            action += Normal(0, action_noise * self.action_high).sample() if action_noise > 0 else torch.zeros(1)
            action = torch.clamp(action, self.action_low, self.action_high)

        return action.numpy()

    
    def update(self, state, action, reward, state_t1, done):
        
        state = torch.tensor(state).float()
        action = torch.tensor(action).float()
        reward = torch.tensor(reward).unsqueeze(dim = 1)
        state_t1 = torch.tensor(state_t1).float()
        done = torch.tensor(done).unsqueeze(dim = 1)
        
        for _ in range(self.Q_updates):
            self.update_Q(state, action, reward, state_t1, done)
        
        self.update_actor(state)
        
        # update the target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def update_actor(self, state):
        
        action = self.actor(state) # add clamp?
        actor_loss = -self.Q(state, action).mean() 
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.actor_loss.append(actor_loss.item())
    
    def update_Q(self, state, action, reward, state_t1, done):
        
        next_Q = reward + self.Q(state, action) * (1 - done) * self.gamma
        
        with torch.no_grad():
            action_t1 = self.actor_target(state_t1) # add clamp?
            #action_t1 = torch.clamp(action_t1, self.action_low, self.action_high)
            Q_t1 = self.Q_target(state_t1, action_t1)
        
        next_Q = next_Q.float()
        Q_loss = F.mse_loss(next_Q, Q_t1)
        
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()
        
        self.Q_loss.append(Q_loss.item())