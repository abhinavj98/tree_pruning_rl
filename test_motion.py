import os
import argparse
from datetime import datetime
import numpy as np
from itertools import count
from collections import namedtuple, deque
import pickle
import torch
import gym
import random
from ppo_discrete import PPO, Memory, ActorCritic
from gym_env_discrete import ur5GymEnv
import torchvision
import imageio
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import cv2
import pybullet
writer = SummaryWriter()



title = 'PyBullet UR5 robot'

def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    # env
    # arg('--env_name', type=str, default='ur5GymEnv', help='environment name')
    arg('--render', action='store_true', default=False, help='render the environment')
    arg('--randObjPos', action='store_true', default=True, help='fixed object position to pick up')
    arg('--mel', type=int, default=100, help='max episode length')
    arg('--repeat', type=int, default=1, help='repeat action')
    arg('--simgrip', action='store_true', default=False, help='simulated gripper')
    arg('--task', type=int, default=0, help='task to learn: 0 move, 1 pick-up, 2 drop')
    arg('--lp', type=float, default=0.1, help='learning parameter for task')
    # train:
    arg('--seed', type=int, default=123, help='random seed')
    arg('--emb_size',   type=int, default=512, help='embedding size')
    arg('--solved_reward', type=int, default=0, help='stop training if avg_reward > solved_reward')
    arg('--log_interval', type=int, default=10, help='interval for log')
    arg('--save_interval', type=int, default=150, help='interval for saving model')
    arg('--max_episodes', type=int, default=150000, help='max training episodes')
    arg('--update_timestep', type=int, default=1000, help='update policy every n timesteps')
    arg('--action_std', type=float, default=1.0, help='constant std for action distribution (Multivariate Normal)')
    arg('--K_epochs', type=int, default=10, help='update policy for K epochs')
    arg('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
    arg('--gamma', type=float, default=0.99, help='discount factor')
    arg('--lr', type=float, default=1e-3, help='parameters for Adam optimizer')
    arg('--betas', type=float, default=(0.9, 0.999), help='')
    arg('--loss_entropy_c', type=float, default=0.01, help='coefficient for entropy term in loss')
    arg('--loss_value_c', type=float, default=0.5, help='coefficient for value term in loss')
    arg('--save_dir', type=str, default='saved_rl_models/', help='path to save the models')
    arg('--cuda', dest='cuda', action='store_true', default=False, help='Use cuda to train model')
    arg('--mps', dest='mps', action='store_true', default=False, help='Use mps to train model')
    arg('--device_num', type=str, default=0,  help='GPU number to use')
    arg('--complex_tree', type = int, default=0, help='Use complex tree to train model')
    args = parser.parse_args()
    return args

args = get_args() # Holds all the input arguments

np.set_printoptions(precision=2)
torch.set_printoptions(profile="full", precision=2)

# Color Palette
CP_R = '\033[31m'
CP_G = '\033[32m'
CP_B = '\033[34m'
CP_Y = '\033[33m'
CP_C = '\033[0m'

def write_file(filepath, data, mode):
    f = open(filepath, mode)
    f.write(data)
    f.close()

args.filename_tl = 'training_log.txt' # log file

env = ur5GymEnv(renders=args.render, maxSteps=args.mel,
            actionRepeat=args.repeat, task=args.task, randObjPos=args.randObjPos,
            simulatedGripper=args.simgrip, learning_param=args.lp, complex_tree=args.complex_tree)

env.seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.reset()
print(env.actions)
while True:
    keys = pybullet.getKeyboardEvents()
    for k,state in keys.items():
        if ord('-') == k:
            action = 11
        elif ord('=') == k:
            action = 12
        else:
            action = (k - ord('0'))
            if action == 0:
                action = 10
        if action > 12 or action < 0:
            print("Bad action")
            continue
        if state&pybullet.KEY_WAS_TRIGGERED:
            print(env.rev_actions[action])
            r = env.step(action, False)
        
        
"""
print("Initial position: ", env.achieved_goal, pybullet.getEulerFromQuaternion(env.achieved_orient))
try:
    action = int(input('action please'))
except:
    continue
if action == 0:
    quit()
if action > 12:
    print("Wrong action")
    continue
print(env.rev_actions[action])

r = env.step(action, False)
print(r[1][-1])
print("Final position: ", env.achieved_goal, pybullet.getEulerFromQuaternion(env.achieved_orient))
"""