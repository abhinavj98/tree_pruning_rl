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
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader    
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
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
        del self.depth[:]

class AutoEncoderGAP(nn.Module):
    def __init__(self):
        super(AutoEncoderGAP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding='same'),  # b, 16, 224, 224
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),  #  b, 64, 112, 112
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride = 2),  #  b, 64, 56, 56
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride = 2),  #  b, 128, 28, 28
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, stride = 2),  #  b, 128, 14, 14
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, stride = 2),  #  b, 128, 7, 7
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding = 1), 
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding = 1), 
            nn.MaxPool2d(7),
            Reshape(-1, 128)
        )
        output_conv = nn.Conv2d(3, 1, 3, padding = 1)
        # output_conv.bias.data.fill_(0.3)
        self.decoder = nn.Sequential(
            nn.Linear(128, 7*7*32),
            Reshape(-1, 32, 7, 7),
            nn.ConvTranspose2d(32, 32, 3, padding = 1, stride=1), # 32. 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 2, stride=2), # 32. 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, padding = 1, stride=1),  # b, 16, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 2, stride=2),  # b, 16, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),  # b, 8, 56, 56
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 2, stride=2),  # b, 16, 112, 112
            # nn.ReLU(),
            # nn.ConvTranspose2d(16, 8, 2, stride=2), # b, 8, 224, 224
            # nn.ReLU(),
            nn.Conv2d(8, 3, 3, padding = 1), # b, 3, 224, 224
            nn.ReLU(),
            output_conv, # b, 1, 224, 224
            #nn.ReLU()
        )

    def forward(self, observation):
        # print(observation)
        encoding = self.encoder(observation)
        recon = self.decoder(encoding)
        return encoding,recon

class SpatialAutoEncoder(nn.Module):
    def __init__(self):
        super(SpatialAutoEncoder, self).__init__()
        self.latent_space = 32
        self.output_size = 112
        self.input_size = 224
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding='same'),  # b, 16, 224, 224
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),  #  b, 64, 112, 112
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride = 2),  #  b, 64, 56, 56
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, padding='same')
            # nn.Conv2d(64, 128, 3, padding=1, stride = 2),  #  b, 128, 28, 28
            # nn.ReLU(),
            # nn.Conv2d(128, 128, 3, padding=1, stride = 2),  #  b, 128, 14, 14
            # nn.ReLU(),
            # nn.Conv2d(128, 128, 3, padding=1, stride = 2),  #  b, 128, 7, 7
            # nn.ReLU(),
            # nn.Conv2d(128, 128, 3, padding = 1), 
            # nn.ReLU(),
            # nn.Conv2d(128, 128, 3, padding = 1), 
            # nn.ReLU()
        )
        self.spatial_softmax = SpatialSoftmax(56, 56, 32)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        # self.encoding = PositionalEncoding1D(64)
        #64*2
        output_conv = nn.Conv2d(3, 1, 3, padding = 1)
        output_linear = nn.Linear(32*2 + 32, 7*7*16)
       # output_linear.bias.data.fill_(0.5)
        self.decoder = nn.Sequential(
            output_linear,
            nn.ReLU(),
            Reshape(-1, 16, 7, 7),
            nn.ConvTranspose2d(16, 32, 3, padding = 1, stride=1), # 32. 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 2, stride=2), # 32. 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, padding = 1, stride=1),  # b, 16, 28, 28
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 2, stride=2),  # b, 16, 28,  28
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),  # b, 8, 56, 56
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 2, stride=2),  # b, 16, 112, 112
            # nn.ReLU(),
            # nn.ConvTranspose2d(16, 8, 2, stride=2), # b, 8, 224, 224
            # nn.ReLU(),
            nn.Conv2d(8, 3, 3, padding = 1), # b, 3, 224, 224
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding = 1), # b, 3, 224, 224
            nn.ReLU(),
            output_conv, # b, 1, 224, 224
            #nn.ReLU()
        )
        # output_conv = nn.Conv2d(3, 1, 3, padding = 1)
        # output_conv.bias.data.fill_(0.3)
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(128, 128, 3, padding = 1, stride=1), # 128. 14, 14
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 128, 2, stride=2), # 128. 14, 14
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(128, 64, 3, padding = 1, stride=1),  # b, 64, 28, 28
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 64, 2, stride=2),  # b, 64, 28, 28
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64, 32, 2, stride=2),  # b, 32, 56, 56
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(32, 16, 2, stride=2),  # b, 16, 112, 112
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(16, 8, 2, stride=2), # b, 8, 224, 224
        #     nn.ReLU(),
        #     nn.Conv2d(8, 3, 3, padding = 1), # b, 3, 224, 224
        #     nn.ReLU(),
        #     output_conv  # b, 1, 224, 224
        #     #nn.ReLU()
        # )

    def forward(self, x):
        encoding = self.encoder(x)
        argmax = self.spatial_softmax(encoding)
        maxval = self.maxpool(encoding).squeeze(-1).squeeze(-1)
        #print(features.shape)
        features = state = torch.cat((argmax, maxval),-1)
        # print((features))
        # features = self.encoding(features)
      
        recon = self.decoder(features).reshape(-1,1,self.output_size,self.output_size)
        return features,recon

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:  
            self.temperature = torch.ones(1)*temperature   
        else:   
            self.temperature = Parameter(torch.ones(1))   

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1,  self.channel*2)

        return feature_keypoints
    
class Actor(nn.Module):
    def __init__(self, device, state_dim, emb_size, action_dim, action_std):
        super(Actor, self).__init__()
        emb_ds = int(emb_size/4)
        self.conv = nn.Sequential(
                    nn.Conv2d(128, 16, 1, padding='same'),
                    nn.ReLU(),
                    )
        self.dense =  nn.Sequential(
                    nn.Linear(state_dim, emb_size),
                    nn.ReLU(),
                    nn.Linear(emb_size, action_dim),
                    nn.Softmax(dim=-1) #discrete action
                    )
    def forward(self, image_features, state):
        state = torch.cat((image_features, state, state, state),-1)
      #  print(state.shape)
        # conv_head = self.conv(image_features)
        # if len(image_features.shape) == 4:
        #     conv_head = conv_head.view(conv_head.shape[0], -1)
        # else:
        #     conv_head = conv_head.view(1, -1)
        # dense_input = torch.cat((conv_head, state),-1) 
        action = self.dense(state)
        return action

class Critic(nn.Module):
    def __init__(self, device, state_dim, emb_size, action_dim, action_std):
        super(Critic, self).__init__()
        emb_ds = int(emb_size/4)
        self.conv = nn.Sequential(
                    nn.Conv2d(128, 16, 1, padding='same'),
                    nn.ReLU()
                    )
        self.dense = nn.Sequential(
                nn.Linear(state_dim, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, 1)
                )
    def forward(self, image_features, state):
        state = torch.cat((image_features, state, state, state),-1)
        # conv_head = self.conv(image_features)
        # if len(image_features.shape) == 4:
        #     conv_head = conv_head.view(conv_head.shape[0], -1)
        # else:
        #     conv_head = conv_head.view(1, -1)

        # dense_input = torch.cat((conv_head, state),1) 
        value = self.dense(state)
        return value


class ActorCritic(nn.Module):
    def __init__(self, device, state_dim, emb_size, action_dim, action_std, writer = None):
        self.device = device
        super(ActorCritic, self).__init__()
          # autoencoder
        self.depth_autoencoder = SpatialAutoEncoder().to(self.device)
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
    
    def evaluate(self, state, depth, action, num_episode):
        if torch.isnan(depth).any():
            print("Depth Nan in eval")
        depth_features = self.depth_autoencoder(depth)
        if torch.isnan(depth_features[0]).any():
            print("Depth_features  in eval is Nan!!!!!!!!!!!!!!!!!")
            if self.writer:
                self.writer.add_image("train/random", depth+0.5, 0)
        
        if num_episode > 2000:
            action_probs = self.actor(depth_features[0].detach(), state)

            distribution = Categorical(action_probs)
            
            action_logprobs = distribution.log_prob(action)
            distribution_entropy = distribution.entropy()
            state_value = self.critic(depth_features[0].detach(), state)
        else:
            action_probs = self.actor(depth_features[0].detach()*0, state)

            distribution = Categorical(action_probs)
            
            action_logprobs = distribution.log_prob(action)
            distribution_entropy = distribution.entropy()
            state_value = self.critic(depth_features[0].detach()*0, state)
        
        return action_logprobs, torch.squeeze(state_value), distribution_entropy, depth_features



class PPO:
    def __init__(self, args, env, writer):
        self.args = args
        self.env = env
        self.writer = writer
        self.device = self.args.device
        self.action_dim = self.env.action_dim
        self.state_dim = self.env.observation_space.shape[0]*3 + 32*3 #!!!Get this right
        self.policy = ActorCritic(self.device ,self.state_dim, self.args.emb_size, self.action_dim, self.args.action_std, writer = self.writer).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.args.lr, betas=self.args.betas)
        
        self.policy_old = ActorCritic(self.device, self.state_dim, self.args.emb_size, self.action_dim, self.args.action_std, writer = self.writer).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.aeMseLoss = nn.MSELoss(reduction='none')
        
        
    def select_action(self, depth_features, state):
        #state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        #image_features_avg_pooled = torch.nn.functional.avg_pool2d(depth,7)
        action = self.policy_old.act(depth_features, state)
        return action[0].cpu().data.numpy().flatten(), action[1]

    def get_depth_features(self, img):
        return self.policy_old.depth_autoencoder(img)

    
    def update(self, memory, num_episode):
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
        old_depth = torch.squeeze(torch.stack(memory.depth), 0).to(self.device).detach()
        train_ds = TensorDataset(old_states, old_actions, old_logprobs, old_depth, rewards)
        train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)
         #Plotting
        plot_dict = {}
        plot_dict['surr2'] =  0
        plot_dict['surr1'] =  0
        plot_dict['critic_loss'] =  0
        plot_dict['actor_loss'] =  0
        plot_dict['total_loss'] =  0
        plot_dict['ae_loss'] =  0
        plot_dict['entropy_loss'] = 0
        plot_dict['random'] = []
        # Optimize policy for K epochs:
        for epoch in range(self.args.K_epochs):
            for old_states_batch, old_actions_batch, old_logprobs_batch, old_depth_batch, old_rewards in train_dataloader:

                # Evaluating old actions and values :
                logprobs, state_values, distribution_entropy, autoencoder_io = self.policy.evaluate(old_states_batch, old_depth_batch, old_actions_batch, num_episode)
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs_batch.detach())
                # Finding Surrogate Loss:
                advantages = old_rewards - state_values.detach()
                n,c, h, w = (old_depth_batch.shape)
                #print(F.interpolate(old_depth_batch, size = (56,56)).shape, autoencoder_io[1].shape)
                ae_loss = self.aeMseLoss(F.interpolate(old_depth_batch, size = (112,112)), autoencoder_io[1]) 
                critic_loss = self.args.loss_value_c*self.MseLoss(state_values, old_rewards)
                entropy_loss = self.args.loss_entropy_c*distribution_entropy  
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + \
                        + critic_loss + \
                        - entropy_loss +\
                        +  ae_loss.mean()*self.args.loss_ae_c
                #Make plotting
                #print((old_depth_batch[(ae_loss>0.1)]).shape)
               # print(ae_loss)
                # ae_loss = self.aeMseLoss(F.interpolate(old_depth_batch, size = (56,56)), autoencoder_io[1]) 
                # loss = ae_loss.mean()
                
                elem_aeloss = ae_loss.reshape(-1,1,112,112).mean(dim = [2,3], keepdim = True).squeeze().squeeze().squeeze()
                plot_dict['random'].extend(old_depth_batch[torch.where(elem_aeloss>10)])
                plot_dict['surr1']-=surr1.mean()/self.args.K_epochs
                plot_dict['surr2']-=surr2.mean()/self.args.K_epochs
                plot_dict['critic_loss']+=critic_loss.mean()/self.args.K_epochs
                plot_dict['actor_loss']+=-torch.min(surr1, surr2).mean()/self.args.K_epochs
                plot_dict['entropy_loss']+=-entropy_loss.mean()/self.args.K_epochs
                plot_dict['total_loss']+=loss.mean()/self.args.K_epochs
                plot_dict['ae_loss']+=ae_loss.mean()/self.args.K_epochs
                
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
            
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        return plot_dict

# class Variance(Stat):
#     '''
#     Running computation of mean and variance. Use this when you just need
#     basic stats without covariance.
#     '''

#     def __init__(self, state=None):
#         if state is not None:
#             return super().__init__(state)
#         self.count = 0
#         self.batchcount = 0
#         self._mean = None
#         self.v_cmom2 = None
#         self.data_shape = None

#     def add(self, a):
#         a = self._normalize_add_shape(a)
#         if len(a) == 0:
#             return
#         batch_count = a.shape[0]
#         batch_mean = a.sum(0) / batch_count
#         centered = a - batch_mean
#         self.batchcount += 1
#         # Initial batch.
#         if self._mean is None:
#             self.count = batch_count
#             self._mean = batch_mean
#             self.v_cmom2 = centered.pow(2).sum(0)
#             return
#         # Update a batch using Chan-style update for numerical stability.
#         oldcount = self.count
#         self.count += batch_count
#         new_frac = float(batch_count) / self.count
#         # Update the mean according to the batch deviation from the old mean.
#         delta = batch_mean.sub_(self._mean).mul_(new_frac)
#         self._mean.add_(delta)
#         # Update the variance using the batch deviation
#         self.v_cmom2.add_(centered.pow(2).sum(0))
#         self.v_cmom2.add_(delta.pow_(2).mul_(new_frac * oldcount))

#     def size(self):
#         return self.count

#     def mean(self):
#         return self._restore_result_shape(self._mean)

#     def variance(self, unbiased=True):
#         return self._restore_result_shape(self.v_cmom2
#                 / (self.count - (1 if unbiased else 0)))

#     def stdev(self, unbiased=True):
#         return self.variance(unbiased=unbiased).sqrt()

#     def to_(self, device):
#         if self._mean is not None:
#             self._mean = self._mean.to(device)
#         if self.v_cmom2 is not None:
#             self.v_cmom2 = self.v_cmom2.to(device)

#     def load_state_dict(self, state):
#         self.count = state['count']
#         self.batchcount = state['batchcount']
#         self._mean = torch.from_numpy(state['mean'])
#         self.v_cmom2 = torch.from_numpy(state['cmom2'])
#         self.data_shape = None if state['data_shape'] is None else tuple(state['data_shape'])

#     def state_dict(self):
#         return dict(
#             constructor=self.__module__ + '.' + self.__class__.__name__ + '()',
#             count=self.count,
#             data_shape=self.data_shape and tuple(self.data_shape),
#             batchcount=self.batchcount,
#             mean=self._mean.cpu().numpy(),
#             cmom2=self.v_cmom2.cpu().numpy())
