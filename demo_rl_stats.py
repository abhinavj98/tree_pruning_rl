# E. Culurciello
# February 2021

# PyBullet UR-5 from https://github.com/josepdaniel/UR5Bullet

import numpy as np
import gym
import argparse, random
from ppo_discrete import PPO, Memory
from gym_env_discrete import ur5GymEnv
import torch
import statistics

title = 'PyBullet UR-5'

def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    arg('trained_file', type=str, default='PPO_continuous.pth', help='environment name')
    # env
    # arg('--env_name', type=str, default='ur10GymEnv', help='environment name')
    arg('--render', action='store_true', default=False, help='render the environment')
    arg('--randObjPos', action='store_true', default=True, help='fixed object position to pick up')
    arg('--mel', type=int, default=100, help='max episode length')
    arg('--repeat', type=int, default=1, help='repeat action')
    arg('--simgrip', action='store_true', default=False, help='simulated gripper')
    arg('--task', type=int, default=0, help='task to learn: 0 move, 1 pick-up, 2 drop')
    arg('--lp', type=float, default=0.1, help='learning parameter for task')
    # train:
    arg('--seed', type=int, default=987, help='random seed')
    arg('--emb_size',   type=int, default=512  , help='embedding size')
    arg('--n_episodes', type=int, default=100, help='max training episodes')
    arg('--action_std', type=float, default=0.25, help='constant std for action distribution (Multivariate Normal)')
    arg('--K_epochs', type=int, default=100, help='update policy for K epochs')
    arg('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
    arg('--gamma', type=float, default=0.99, help='discount factor')
    arg('--lr', type=float, default=1e-3, help='parameters for Adam optimizer')
    arg('--betas', type=float, default=(0.9, 0.999), help='')
    arg('--cuda', dest='cuda', action='store_true', default=False, help='Use cuda to train model')
    arg('--device_num', type=str, default=0, help='GPU number to use')
    args = parser.parse_args()
    return args

args = get_args() # Holds all the input arguments
print(args)

args.device = torch.device('cuda:'+str(args.device_num) if args.cuda else 'cpu')
print('Using device:', 'cuda' if args.cuda else 'cpu', ', device number:', args.device_num, ', GPUs in system:', torch.cuda.device_count())

# create the environment
print(title)
args.env_name = title
env = ur5GymEnv(renders=args.render, maxSteps=args.mel, 
        actionRepeat=args.repeat, task=args.task, randObjPos=args.randObjPos,
        simulatedGripper=args.simgrip, learning_param=args.lp)

env.seed(args.seed)
torch.manual_seed(args.seed)

args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# memory = Memory()
ppo = PPO(args, env)

print('Loading model:', args.trained_file)
ppo.policy_old.load_state_dict(torch.load(args.trained_file, map_location = torch.device(args.device) ))
ppo.policy_old.to()
memory = Memory()
avg_reward = []
num_collisions = 0
avg_distance = []
# running test:
for ep in range(1, args.n_episodes+1):
    ep_reward = 0
    state = env.reset()
    for t in range(args.mel):
        state = torch.FloatTensor(state.reshape(1, -1)).to(args.device)
        rgb = torch.FloatTensor(env.rgb).to(args.device)
        #state = torch.FloatTensor(state.reshape(1, -1)).to(args.device)
        action = ppo.policy_old.act(rgb, state, memory)
        action = action.data.cpu().numpy().flatten()
        state, reward, done, _, collision = env.step(action)
        ep_reward += reward
        num_collisions+=collision
        if t == args.mel-1:
            avg_distance.append(env.target_dist)

        # print(t, env.target_dist)
        # input()
 
        if done:
            break
        
    print('Episode: {}\tSteps: {}\tReward: {}'.format(ep, t, int(ep_reward)))
    print(env.target_dist)
    avg_reward.append(ep_reward)
    ep_reward = 0
    env.close()
print("Average reward over {} episodes: {} and  with variance {}".format(args.n_episodes, statistics.mean(avg_reward), statistics.stdev(avg_reward)))
print("Num collisions over {} episodes: {}".format(args.n_episodes, num_collisions))
print("Avg distance over {} episodes: {} and  with variance {}".format(args.n_episodes, statistics.mean(avg_distance), statistics.stdev(avg_distance)))

