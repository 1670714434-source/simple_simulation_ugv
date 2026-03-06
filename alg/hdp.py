import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn as nn
import airsim
import os

from airsim import MultirotorClient
from alg.actor import Actor
from alg.critic import critic
from env.UGV import Multirotor
from torch.utils.tensorboard import SummaryWriter


def model(current_state, u):
    next_state = np.zeros([1,3])  # 初始化下一个状态
    a = 1e-9
    M = np.dot(np.array([
                                        [math.cos(current_state[2]), -a*math.sin(current_state[2])],
                                        [math.sin(current_state[2]), a*math.cos(current_state[2])],
                                        [1,0]]),u.transpose())
    next_state = current_state + M.transpose()
    return next_state


class HDP:
    def __init__(self,client: MultirotorClient,path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.critic = critic(3).to(self.device)
        self.actor = Actor(3, 2).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0005)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0005)
        self.memory = ReplayBuffer(131072)
        self.batch_size = 256
        self.soft_tau = 0.01  # 软更新率
        self.gamma = 0.98  # 折扣率
        self.client = client
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.path = path
    # 变成二维tensor，[1,3]，因为一维的标量不能做tensor的乘法，actor中第一层的weight形状为[3,512](标量也可以做乘法)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        # tensor.detach()与tensor.data的功能相同，但是若因外部修改导致梯度反向传播出错，.detach()会报错，.data不行，且.detach()得到的数据不带梯度
        action = action.detach().cpu().squeeze(0).numpy()
        return action

    def learning(self):
        curr_path = os.path.dirname(os.path.abspath(__file__))
        self.loss = []
        rewards = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        writer = SummaryWriter('./train_image')
        for train_index in range(1000):
            print('the ', train_index + 1, ' --th  learing start')
            env = Multirotor(self.client)
            x,y,yaw = env.get_state()
            state = np.array([x, y, yaw])
            state = torch.from_numpy(state).float().to(device)
            finish_step = 0
            done = False
            delta_a = 1.0
            delta_c = 1.0
            Q = torch.tensor(np.diag([15, 15, 0.005]), dtype=torch.float32).to(device)
            R = torch.tensor(np.diag([2, 0.5]), dtype=torch.float32).to(device)
            A = torch.tensor(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), dtype=torch.float32).to(device)
            final_distance = (x**2+y**2)**0.5
            ep_reward = 0

            for i_step in range(1000):
                finish_step = finish_step + 1
                #while (delta_a > 1e-4):
                a = 1e-9
                x, y, yaw = env.get_state()
                B = torch.tensor(np.array([
                    [math.cos(yaw), -a * math.sin(yaw)],
                    [math.sin(yaw), a * math.cos(yaw)],
                    [0, 1]]), dtype=torch.float32).to(device)

                J = self.critic(state)
                best_action = - (torch.inverse(R + (B.T * J).mm(B)).mm(B.T) * J).mm(A).mm(state.unsqueeze(0).T).squeeze(1)
                actor_action = self.actor(state)
                next_state, reward, done = env.step(actor_action.detach().cpu().numpy())
                rewards.append(reward)
                next_state = np.array([next_state[0], next_state[1], next_state[2]])
                next_state = torch.from_numpy(next_state).float().to(device)
                V_next, dV_dx_next = self.critic.gradient(next_state)
                #print(0.01*B.T)
                # R_1 = torch.inverse(0.5*R)
                # BT = 0.01 * B.T
                # dV = dV_dx_next.unsqueeze(0).T
                # best_action = torch.matmul(R_1,BT)
                # best_action = torch.matmul(best_action,dV)

                actor_loss = self.criterion(best_action, actor_action)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                delta_a = actor_loss.item()



                #while (delta_c > 1e-4):
                predict_value_k = self.critic(state)

                predict_value_next_k = self.critic(next_state)
                target_value_k = reward + self.gamma * predict_value_next_k
                critic_loss = self.criterion(predict_value_k, target_value_k)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                delta_c = critic_loss.item()

                final_distance = (next_state[0]**2+next_state**2)**0.5
                state = next_state
                print("状态是：", state, i_step,actor_action,best_action,reward)
                if done:
                    break

            writer.add_scalars(main_tag='train',
                               tag_scalar_dict={
                                   'reward': ep_reward,
                               },
                               global_step=train_index)

            if train_index %10 ==0:
                self.save(self.path)

        writer.close()
        return rewards


    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'checkpoint.pt')  # 后缀.pt和.pth没什么区别

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'checkpoint.pt'))

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state = np.array([1,1,1])
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    print(state)