# E. Culurciello
# February 2021

# PyBullet UR-5 from https://github.com/josepdaniel/UR5Bullet
# PPO from: https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py

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
    arg('--loss_value_c', type=float, default=0.12, help='coefficient for value term in loss')
    arg('--loss_ae_c', type=float, default=0.25, help='coefficient for ae loss term in loss')
    arg('--save_dir', type=str, default='saved_rl_models/', help='path to save the models')
    arg('--cuda', dest='cuda', action='store_true', default=False, help='Use cuda to train model')
    arg('--mps', dest='mps', action='store_true', default=False, help='Use mps to train model')
    arg('--device_num', type=str, default=0,  help='GPU number to use')
    arg('--complex_tree', type = int, default=0, help='Use complex tree to train model')
    arg('--name', type = str, default="", help='Name for tensorboard logging')

    args = parser.parse_args()
    return args

args = get_args() # Holds all the input arguments

writer = SummaryWriter(log_dir = "runs/"+args.name)
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

args.device = torch.device('cuda:'+str(args.device_num) if args.cuda else 'cpu')
if args.mps:
    args.device = torch.device('mps')
print('Using device:', args.device, args.device_num, ', GPUs in system:', torch.cuda.device_count())


def main():
    args.env_name = title
    print(CP_G + 'Environment name:', args.env_name, ''+CP_C)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    env = ur5GymEnv(renders=args.render, maxSteps=args.mel,
            actionRepeat=args.repeat, task=args.task, randObjPos=args.randObjPos,
            simulatedGripper=args.simgrip, learning_param=args.lp, complex_tree = args.complex_tree)

    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    memory = Memory()
    ppo = PPO(args, env, writer)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    
    # training loop:
    print('Starting training with learning_param:', args.lp)

    layout = {
    "RewardCritic": {
        "Critic vs Reward": ["Multiline", ["critic_value/train", "reward/train"]],
    },
    }

    random_count = 0
    writer.add_custom_scalars(layout)
    for i_episode in range(1, args.max_episodes+1):
        state = env.reset()
        ep_collision = 0
        ep_goal_reward = 0
        ep_success = 0
        ep_total = 0
        ep_gif = []
        for t in range(args.mel):
            time_step += 1
            if i_episode%args.log_interval == 0:
                gif = True
            else:
                gif = False
            depth = (torch.tensor(env.depth-0.5).to(args.device).unsqueeze(0))
            print(depth)
            if torch.isnan(depth).any():
                print("Depth is NAN!!!!!!!!!!!!!!!!!")
            if (depth > 0.5).any() or (depth < -0.5).any():
                print("DEPTH out of bounds")
            depth_features = ppo.get_depth_features(depth.unsqueeze(0))[0]
            if torch.isnan(depth_features).any():
                print("Depth features in train_rl is Nan!!!!!!!!!!!!!!!!!")
                writer.add_image("train/random", depth+0.5, random_count)
                random_count+=1
            state = torch.FloatTensor(state.reshape(1, -1)).to(args.device)
            if torch.isnan(state).any():
                print("State  in train_rl is Nan!!!!!!!!!!!!!!!!!")
            memory.depth.append(depth)
            memory.states.append(state)
            memory.depth_features.append(depth_features.squeeze())

            action, action_logprob = ppo.select_action(depth_features, state)

            memory.actions.append(torch.Tensor(action))
            memory.logprobs.append(action_logprob)

            state, reward_tuple, done, debug_img,  _ = env.step(action, gif)
            
            reward = reward_tuple[0]
            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            critic_value = ppo.policy.critic(memory.depth_features[-1].unsqueeze(0), memory.states[-1])
            if gif:
                debug_img = cv2.putText(debug_img, "Critic: "+str(critic_value.item()), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255,0,0), 2, cv2.LINE_AA)
                debug_img = cv2.putText(debug_img, "Reward: "+str(memory.rewards[-1]), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255,0,0), 2, cv2.LINE_AA)
                debug_img = cv2.putText(debug_img, "Action: "+str(env.rev_actions[int(action)+1]), (0,110), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255,0,0), 2, cv2.LINE_AA)
                debug_img = cv2.putText(debug_img, "Current: "+str(env.achieved_goal), (0,140), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255,0,0), 2, cv2.LINE_AA)
                debug_img = cv2.putText(debug_img, "Goal: "+str(env.desired_goal), (0,170), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255,0,0), 2, cv2.LINE_AA)
                ep_gif.append(torch.tensor(debug_img))

            # learning: 
            if i_episode == 1:
                memory.clear_memory()
                continue
            
            if time_step % args.update_timestep == 0:
                loss_dict = ppo.update(memory)
                memory.clear_memory()
                time_step = 0
                for k,v in loss_dict.items():
                    if k == 'random':
                        continue
                    writer.add_scalar("{}/train".format(k), v, i_episode)
                ae_image = torchvision.utils.make_grid([depth+0.5, ppo.policy.depth_autoencoder(depth.unsqueeze(0))[1].squeeze(0)+0.5])
                writer.add_image("train/ae", ae_image, i_episode)
                for i in loss_dict['random']:
                    if i.shape[0]==0:
                        break
                    ae_image = torchvision.utils.make_grid([i+0.5, ppo.policy.depth_autoencoder(i.unsqueeze(0))[1].squeeze(0)+0.5])
                    writer.add_image("train/random", ae_image, random_count)
                    random_count+=1
                del loss_dict['random'][:]
            writer.add_scalar("critic_value/train", critic_value, (i_episode-1)*args.mel+t)
            writer.add_scalar("reward/train", reward, (i_episode-1)*args.mel+t)

            running_reward += reward
            ep_total += reward
            ep_goal_reward += reward_tuple[1]
            ep_success += reward_tuple[2]
            ep_collision += int(reward_tuple[3])

            if done:
                break

        avg_length += t
        if ep_gif:
            ep_gif = torch.stack(ep_gif).unsqueeze(0)
            ep_gif = ep_gif.permute(0,1,4,2,3)
            writer.add_video("episode/train", ep_gif , i_episode, fps = 5)
            #imageio.mimsave('/Users/abhinav/Desktop/gradstuff/coursework/DeepLearning/tree_pruning_rl/animation_2/episode_{}.gif'.format(i_episode), ep_gif)
        # stop training if avg_reward > solved_reward
        #Update tensorboard
        writer.add_scalar("reward_goal/train", ep_goal_reward, i_episode)
        writer.add_scalar("reward_success/train", ep_success, i_episode)
        writer.add_scalar("reward_collision/train", ep_collision, i_episode)
        writer.add_scalar("reward_total/train", ep_total, i_episode)
        
        # if running_reward > (args.log_interval*args.solved_reward):
        #     print("########## Solved! ##########")
        #     torch.save(ppo.policy.state_dict(), args.save_dir+'./model_solved.pth')
        #     break

        # save every few episodes
        if i_episode % args.save_interval == 0:
            torch.save(ppo.policy.state_dict(), args.save_dir+'/model_epoch_'+str(int(i_episode/args.save_interval))+'.pth')

        # logging
        if i_episode % args.log_interval == 0:
            avg_length = int(avg_length/args.log_interval)
            running_reward = ((running_reward/args.log_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()
