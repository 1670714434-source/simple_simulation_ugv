import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, n_states, init_w=3e-3):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(n_states, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, 128)
        self.linear5 = nn.Linear(128, 1)

        self.linear5.weight.data.uniform_(-init_w, init_w)
        self.linear5.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        # 按维数1拼接(按维数1拼接为横着拼，按维数0拼接为竖着拼)
        x = state
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        q = self.linear5(x)
        return q


if __name__ == '__main__':
    import numpy as np
    import sys
    import os

    # 添加项目根目录到环境变量以便导入 env
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env.UGV import UGV

    # 1. 获取环境状态维度
    env = UGV()
    state = env.get_state()
    n_states = len(state)
    print(f"State dimension from UGV env: {n_states}")

    # 2. 初始化 Critic 网络
    critic = Critic(n_states)
    print("\nCritic Network Structure:")
    print(critic)

    # 3. 测试单个状态的前向传播
    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 增加 batch 维度 [1, n_states]
    value = critic(state_tensor)

    print(f"\nTest with initial state:")
    print(f"Input state: {state}")
    print(f"Output value (Q-value/V-value): {value.item()}")

    # 4. 测试 Batch 输入
    batch_size = 4
    dummy_batch = torch.randn(batch_size, n_states)
    batch_output = critic(dummy_batch)

    print(f"\nTest with random batch (batch_size={batch_size}):")
    print(f"Input shape: {dummy_batch.shape}")
    print(f"Output shape: {batch_output.shape}")
    print(f"Outputs: \n{batch_output.detach().numpy()}")
