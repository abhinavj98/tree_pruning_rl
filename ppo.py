# E. Culurciello
# February 2021

# PyBullet UR-5 from https://github.com/josepdaniel/UR5Bullet
# PPO from: https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
import torchvision.models as models    
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding='same'),  # b, 16, 224, 224
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride = 2),  # b, 32, 112, 112
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding='same'),  #  b, 64, 112, 112
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride = 2),  #  b, 64, 56, 56
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding='same'),  #  b, 128, 56, 56
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1, stride = 2),  #  b, 256, 28, 28
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # b, 256,14,14
            nn.ReLU(),
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),  # b, 256,7,7
            nn.ReLU()
        )
      
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, stride=2, output_padding=1, padding = 1), # 256. 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, stride=2, output_padding=1, padding=1),  # b, 256, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, output_padding=1, padding=1),  # b, 128, 56, 56
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1, padding=1),  # b, 64, 112, 112
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1, padding=1),  # b, 32, 224, 224
            nn.ReLU(),
            nn.Conv2d(32, 4, 3, padding = 'same'),  # b, 4, 224, 224
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, device, state_dim, emb_size, action_dim, action_std):
        self.device = device
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.vgg = models.vgg16(pretrained=True).to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        #actor
        emb_ds = int(emb_size/4)
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_ds),
                nn.ReLU(),
                nn.Linear(emb_ds, action_dim),
                nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_ds),
                nn.ReLU(),
                nn.Linear(emb_ds, 1)
                )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(self.device)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, rgb,  state, memory):
        self.image_features = self.vgg.features(rgb.unsqueeze(0)).detach()
        self.image_features = torch.nn.functional.avg_pool2d(self.image_features,7)
        #print(self.image_features.shape)
        state = torch.cat((self.image_features.view(-1).unsqueeze(0), state, state, state),1)
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(self.device)
        
        distribution = MultivariateNormal(action_mean, cov_mat)
        action = distribution.sample()
        action_logprob = distribution.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):
        #state = torch.cat((self.image_features.view(-1).unsqueeze(0), state),1)   Check if we need rthis
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        
        distribution = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = distribution.log_prob(action)
        distribution_entropy = distribution.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), distribution_entropy



class PPO:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.device = self.args.device

        self.state_dim = self.env.observation_space.shape[0]*3 + 512  #!!!Get this right
        self.action_dim = self.env.action_space.shape[0]
        
        self.policy = ActorCritic(self.device ,self.state_dim, self.args.emb_size, self.action_dim, self.args.action_std).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.args.lr, betas=self.args.betas)
        
        self.policy_old = ActorCritic(self.device, self.state_dim, self.args.emb_size, self.action_dim, self.args.action_std).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, rgb,  state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        rgb = torch.FloatTensor(rgb).to(self.device)
        return self.policy_old.act(rgb, state, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.args.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(np.array(rewards)).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = rewards.float().squeeze()
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(self.device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.args.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, distribution_entropy = self.policy.evaluate(old_states, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + \
                    + self.args.loss_value_c*self.MseLoss(state_values, rewards) + \
                    - self.args.loss_entropy_c*distribution_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

