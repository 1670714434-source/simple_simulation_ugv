import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn as nn
import os

from alg.actor import Actor
from alg.critic import Critic
from env.UGV import UGV
from torch.utils.tensorboard import SummaryWriter


class HDP:
    def __init__(self,path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.critic = Critic(3).to(self.device).to(self.device)
        self.actor = Actor(3, 2).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0005)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0005)
        self.soft_tau = 0.01  # 软更新率
        self.gamma = 0.98  # 折扣率
        self.criterion = torch.nn.MSELoss()
        self.path = path
        self.env = UGV()
    # 变成二维tensor，[1,3]，因为一维的标量不能做tensor的乘法，actor中第一层的weight形状为[3,512](标量也可以做乘法)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        # tensor.detach()与tensor.data的功能相同，但是若因外部修改导致梯度反向传播出错，.detach()会报错，.data不行，且.detach()得到的数据不带梯度
        action = action.detach().cpu().squeeze(0).numpy()
        return action

    def update(self, state, action, next_state):


        return


    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'checkpoint.pt')  # 后缀.pt和.pth没什么区别

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'checkpoint.pt'))

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state = np.array([1,1,1])
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    print(state)