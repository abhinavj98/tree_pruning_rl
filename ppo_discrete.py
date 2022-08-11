# E. Culurciello
# February 2021

# PyBullet UR-5 from https://github.com/josepdaniel/UR5Bullet
# PPO from: https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py

from enum import auto
from os import stat
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import gym
import numpy as np
import torchvision.models as model

from torch.utils.data import TensorDataset, DataLoader    
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.depth = []
        self.depth_features = []
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.depth_features[:]
        if len(self.depth)>1000:
            del self.depth[:-1000]

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding='same'),  # b, 16, 224, 224
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
            nn.Conv2d(256, 256, 3, stride=2, padding=1),  # b, 256,14,14
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),  # b, 256,7,7
            nn.ReLU()
        )
      
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, stride=2, output_padding=1, padding = 1), # 256. 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, stride=2, output_padding=1, padding=1),  # b, 256, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, output_padding=1, padding=1),  # b, 128, 56, 56
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1, padding=1),  # b, 64, 112, 112
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1, padding=1),  # b, 32, 224, 224
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding = 'same'),  # b, 1, 224, 224
            nn.Sigmoid()
        )

    def forward(self, x):
        encoding = self.encoder(x)
        recon = self.decoder(encoding)
        return encoding,recon

class Actor(nn.Module):
    def __init__(self, device, state_dim, emb_size, action_dim, action_std):
        super(Actor, self).__init__()
        emb_ds = int(emb_size/4)
        self.conv = nn.Sequential(
                    nn.Conv2d(256, 128, 1, padding='same'),
                    nn.ReLU(),
                    nn.Conv2d(128, 16, 1, padding='same'),
                    nn.ReLU(),
                    )
        self.dense =  nn.Sequential(
                    nn.Linear(state_dim, emb_size),
                    nn.ReLU(),
                    nn.Linear(emb_size, emb_size),
                    nn.ReLU(),
                    nn.Linear(emb_size, emb_ds),
                    nn.ReLU(),
                    nn.Linear(emb_ds, action_dim),
                    nn.Softmax(dim=-1) #discrete action
                    )
    def forward(self, image_features, state):
        state = torch.cat((state, state, state),1)
        conv_head = self.conv(image_features)
        if len(image_features.shape) == 4:
            conv_head = conv_head.view(conv_head.shape[0], -1)
        else:
            conv_head = conv_head.view(1, -1)
        dense_input = torch.cat((conv_head, state),1) 
        action = self.dense(dense_input)
        return action

class Critic(nn.Module):
    def __init__(self, device, state_dim, emb_size, action_dim, action_std):
        super(Critic, self).__init__()
        emb_ds = int(emb_size/4)
        self.conv = nn.Sequential(
                    nn.Conv2d(256, 128, 1, padding='same'),
                    nn.ReLU(),
                    nn.Conv2d(128, 16, 1, padding='same'),
                    nn.ReLU(),
                    )
        self.dense = nn.Sequential(
                nn.Linear(state_dim, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_ds),
                nn.ReLU(),
                nn.Linear(emb_ds, 1)
                )
    def forward(self, image_features, state):
        state = torch.cat((state, state, state),1)
        conv_head = self.conv(image_features)
        if len(image_features.shape) == 4:
            conv_head = conv_head.view(conv_head.shape[0], -1)
        else:
            conv_head = conv_head.view(1, -1)

        dense_input = torch.cat((conv_head, state),1) 
        value = self.dense(dense_input)
        return value


class ActorCritic(nn.Module):
    def __init__(self, device, state_dim, emb_size, action_dim, action_std):
        self.device = device
        super(ActorCritic, self).__init__()
          # autoencoder
        self.depth_autoencoder = AutoEncoder().to(self.device)
        # actor
        self.actor = Actor(device, state_dim, emb_size, action_dim, action_std).to(self.device)
        # critic
        self.critic = Critic(device, state_dim, emb_size, action_dim, action_std).to(self.device)
      
        
        # discrete action
    def forward(self):
        raise NotImplementedError
    
    def act(self, depth_features, state):
        action_probs = self.actor(depth_features, state)
        distribution = Categorical(action_probs)

        action = distribution.sample()
        action_logprob = distribution.log_prob(action)
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, depth, action):
        action_probs = self.actor(depth, state)

        distribution = Categorical(action_probs)
        
        action_logprobs = distribution.log_prob(action)
        distribution_entropy = distribution.entropy()
        state_value = self.critic(depth, state)
        
        return action_logprobs, torch.squeeze(state_value), distribution_entropy



class PPO:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.device = self.args.device

        self.state_dim = self.env.observation_space.shape[0]*3 + 7*7*16  #!!!Get this right
        #print('--------------------------------')
        #print(self.env.action_space.shape)
        #self.action_dim = self.env.action_space.shape[0]

        #self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_dim
        
        self.policy = ActorCritic(self.device ,self.state_dim, self.args.emb_size, self.action_dim, self.args.action_std).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.args.lr, betas=self.args.betas)
        
        self.policy_old = ActorCritic(self.device, self.state_dim, self.args.emb_size, self.action_dim, self.args.action_std).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        #self.autoencoder_optimizer = torch.optim.Adam(self.policy.depth_autoencoder.parameters(), lr=self.args.lr, betas=self.args.betas)
        self.MseLoss = nn.MSELoss()
        self.train_ae = True
    
    def select_action(self, depth_features, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        #image_features_avg_pooled = torch.nn.functional.avg_pool2d(depth,7)
        action = self.policy_old.act(depth_features, state)
        return action[0].cpu().data.numpy().flatten(), action[1]

    def get_depth_features(self, img):
        return self.policy.depth_autoencoder(img)

    
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
        old_depth = torch.squeeze(torch.stack(memory.depth_features), 0).to(self.device).detach()
        
         #Plotting
        plot_dict = {}
        plot_dict['surr2'] =  0
        plot_dict['surr1'] =  0
        plot_dict['critic_loss'] =  0
        plot_dict['actor_loss'] =  0
        plot_dict['total_loss'] =  0
        plot_dict['ae_loss'] =  0
        # Optimize policy for K epochs:
        for _ in range(self.args.K_epochs):
           
            # Evaluating old actions and values :
            logprobs, state_values, distribution_entropy = self.policy.evaluate(old_states, old_depth, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + \
                    + self.args.loss_value_c*self.MseLoss(state_values, rewards) + \
                    - self.args.loss_entropy_c*distribution_entropy
            #Make plotting
            plot_dict['surr1']-=surr1.mean()/self.args.K_epochs
            plot_dict['surr2']-=surr2.mean()/self.args.K_epochs
            plot_dict['critic_loss']+=self.args.loss_value_c*self.MseLoss(state_values, rewards).mean()/self.args.K_epochs
            plot_dict['actor_loss']+=(-torch.min(surr1, surr2) - self.args.loss_entropy_c*distribution_entropy).mean()/self.args.K_epochs
            plot_dict['total_loss']+=loss.mean()/self.args.K_epochs
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        #print(memory.depth)
        if self.train_ae:
            depth_ds = torch.stack(memory.depth, 0).to(self.device).detach()
            #recon_out = torch.squeeze(torch.stack(memory.rgbd_recon).to(self.device), 1)

            ae_dataset = TensorDataset(depth_ds, depth_ds) # create your datset
            ae_dataloader = DataLoader(ae_dataset, batch_size=32, shuffle=True) # create your dataloader
            total_loss = 0
            ae_loss = 0
            
            for depth_data in ae_dataloader:
                _, recon = self.policy.depth_autoencoder(depth_data[0])
                ae_loss = self.MseLoss(recon, depth_data[1])
                total_loss += ae_loss.data
                self.optimizer.zero_grad()
                ae_loss.backward()
                self.optimizer.step()
            plot_dict['ae_loss']=total_loss
        return plot_dict
