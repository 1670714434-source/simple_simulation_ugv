import torch
import numpy as np

import os
import sys

import datetime
from pathlib import Path

import argparse
import airsim

from env.UGV import Multirotor

from torch.autograd import Variable
import matplotlib.pyplot as plt

state_dim = 2                                                                  # 状态维度
v_dim = 1                                                                      # 价值维度
action_dim = 2                                                                 # 动作维度
learing_rate = 0.001                                                          # 学习率
# learing_num = 2
learing_num = 200                                                              # 学习次数
sim_num = 20                                                                   # 仿真步长
x0 = np.array([0,0])                                                          # 初始状态
epislon = 1.4                                                                  # 阈值

torch.manual_seed(1)

