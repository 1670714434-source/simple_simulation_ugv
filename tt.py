import torch
import numpy as np

import os
import sys

import datetime
from pathlib import Path

import argparse
import airsim

from alg.hdp import HDP
from env.UGV import Multirotor

curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path

def get_args():
    """
    Hyper parameters
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    curr_time = "20220728-131136"
    parser = argparse.ArgumentParser(description="hyper parameters")
    parser.add_argument('--algo_name', default='DDPG', type=str, help="name of algorithm")
    parser.add_argument('--env_name', default='UE4 and Airsim', type=str, help="name of environment")
    parser.add_argument('--seed', default=1, type=int, help="random seed")
    parser.add_argument('--n_state', default=3 + 1 + 3 + 1 + 13, type=int, help="numbers of state space")
    parser.add_argument('--n_action', default=3, type=int, help="numbers of state action")
    parser.add_argument('--update_times', default=1, type=int, help="update times")
    parser.add_argument('--train_eps', default=1500, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=100, type=int, help="episodes of testing")
    parser.add_argument('--max_step', default=1000, type=int, help="max step for getting target")
    parser.add_argument('--gamma', default=0.98, type=float, help="discounted factor")
    parser.add_argument('--critic_lr', default=1e-3, type=float, help="learning rate of critic")
    parser.add_argument('--actor_lr', default=1e-4, type=float, help="learning rate of actor")
    parser.add_argument('--memory_capacity', default=2**17, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--soft_tau', default=1e-2, type=float)
    parser.add_argument('--result_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                                                 '/' + curr_time + '/results/')
    parser.add_argument('--model_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + \
                                                '/' + curr_time + '/models/')  # path to save models
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # check GPU
    return args

def make_dir(*paths):
    """
    创建文件夹
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
if __name__ == '__main__':
    cfg = get_args()
    make_dir(cfg.result_path, cfg.model_path)
    client = airsim.MultirotorClient()
    agent = HDP(client, cfg.model_path)
    rewards = agent.learning()
