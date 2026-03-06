import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, init_w=3e-3):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(n_states, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, n_actions)

        # tensor.uniform_()函数，从参数的均匀分布中采样进行填充
        nn.init.uniform_(self.linear3.weight.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(self.linear3.bias.detach(), a=-init_w, b=init_w)

        # 另一种写法
        # self.linear5.weight.data.uniform_(-init_w, init_w)
        # self.linear5.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        action = self.linear3(x)  # torch.tanh与F.tanh没有区别
        return action


if __name__ == '__main__':
    actor = Actor(n_states=3, n_actions=2)
    print(sum(p.numel() for p in actor.parameters() if p.requires_grad))