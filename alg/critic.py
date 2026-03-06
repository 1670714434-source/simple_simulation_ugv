import torch
import torch.nn as nn
import torch.nn.functional as F


class critic(nn.Module):
    def __init__(self, n_states, init_w=3e-3):
        super(critic, self).__init__()
        self.linear1 = nn.Linear(n_states, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 1)
        # 随机初始化为较小的值
        nn.init.uniform_(self.linear3.weight.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(self.linear3.bias.detach(), a=-init_w, b=init_w)

        # 另一种写法
        # self.linear5.weight.data.uniform_(-init_w, init_w)
        # self.linear5.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        # 按维数1拼接(按维数1拼接为横着拼，按维数0拼接为竖着拼)
        x = state
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        q = self.linear3(x)

        return q

    def gradient(self, x):
        """计算V(x)对x的梯度 ∂V/∂x"""
        x.requires_grad_(True)  # 确保x需要梯度
        V = self.forward(x)

        # 计算梯度
        grad_V = torch.autograd.grad(
            outputs=V,
            inputs=x,
            grad_outputs=torch.ones_like(V),  # dV/dV = 1
            create_graph=True,  # 允许高阶导数
            retain_graph=True
        )[0]

        return V, grad_V

if __name__ == '__main__':
    critic = critic(n_states=3)
    print(sum(p.numel() for p in critic.parameters() if p.requires_grad))
