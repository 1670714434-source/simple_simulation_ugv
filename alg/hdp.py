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
    def __init__(self,path, env : UGV):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.critic = Critic(3, 2).to(self.device)
        self.actor = Actor(3, 2).to(self.device)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0005)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)

        self.soft_tau = 0.01  # 软更新率
        self.gamma = 0.98  # 折扣率
        self.criterion = torch.nn.MSELoss()
        self.path = path
        self.env = env
    # 变成二维tensor，[1,3]，因为一维的标量不能做tensor的乘法，actor中第一层的weight形状为[3,512](标量也可以做乘法)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        # tensor.detach()与tensor.data的功能相同，但是若因外部修改导致梯度反向传播出错，.detach()会报错，.data不行，且.detach()得到的数据不带梯度
        action = action.detach().cpu().squeeze(0).numpy()
        return action

    def update(self):
        # 转换为 Tensor 格式
        # 注意: action 传入时是 numpy array，需要转 tensor

        state = self.env.get_state()
        action = self.choose_action(state)
        next_state, reward, done = self.env.step(action)
 
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).to(self.device)  # Actor 输出的 action 作为 Critic 的输入
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        reward = torch.FloatTensor([reward]).unsqueeze(0).to(self.device)
        done = torch.FloatTensor([done]).unsqueeze(0).to(self.device)


        # ----------------------
        # 2. Critic (评价网络) 更新
        # ----------------------

        # 计算目标价值 (Bellman Target): y = r + gamma * V(s_next)
        # 使用 no_grad() 是因为我们不希望梯度通过目标值反向传播给 Critic target (半梯度方法)
        with torch.no_grad():
            next_value = self.critic(next_state, self.actor(next_state).to(self.device))
            target_value = reward + self.gamma * next_value * (1 - done)

        # 计算当前状态的预测价值: V(s)
        current_value = self.critic(state, action)

        # 计算贝尔曼误差 (Loss): MSE(Prediction, Target)
        critic_loss = self.criterion(current_value, target_value)

        # 梯度下降更新 Critic 参数
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------------
        # 3. Actor (执行网络) 更新
        # ----------------------


        # Actor 的目标是最大化未来的价值 V(s_next_pred)
        # 因此 loss = -V(s_next_pred)
        actor_loss = - self.critic(state, self.actor(state).to(self.device))

        # 梯度下降更新 Actor 参数
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()


    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'checkpoint.pt')  # 后缀.pt和.pth没什么区别

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'checkpoint.pt'))

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state = np.array([1,1,1])
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    print(state)