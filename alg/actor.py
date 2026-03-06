import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, max_v=3.0, max_w=np.pi/2, init_w=3e-3):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(n_states, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, n_actions)

        self.max_v = max_v
        self.max_w = max_w

        # tensor.uniform_()函数，从参数的均匀分布中采样进行填充
        nn.init.uniform_(self.linear3.weight.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(self.linear3.bias.detach(), a=-init_w, b=init_w)

    def forward(self, state):
        x = state
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        action = torch.tanh(self.linear3(x))

        # v: [-1, 1] -> [0, max_v] (只允许前进)
        v = (action[:, 0:1] + 1) / 2 * self.max_v

        # w: [-1, 1] -> [-max_w, max_w] (允许左右转向)
        w = action[:, 1:2] * self.max_w

        return torch.cat([v, w], dim=1)


if __name__ == '__main__':
    import sys
    import os
    from env.UGV import UGV

    # 添加项目根目录到环境变量
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # 1. 从环境获取维度
    env = UGV()
    state = env.get_state()
    n_states = len(state)
    n_actions = 2  # [v, w]
    print(f"State dimension: {n_states}, Action dimension: {n_actions}")

    # 2. 初始化 Actor (设置最大速度限制)
    # 建议值：
    # max_v: 2.0 ~ 5.0 m/s (对于50x50的地图)
    # max_w: 1.0 ~ 2.0 rad/s (约57~115度/秒)
    actor = Actor(n_states=n_states, n_actions=n_actions, max_v=3.0, max_w=1.57) # 1.57 rad/s approx 90 deg/s
    print("\nActor Network Structure:")
    print(actor)

    # 3. 前向传播测试
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    action = actor(state_tensor)

    print(f"\nTest with initial state:")
    print(f"Input state: {state}")
    print(f"Output action (scaled): {action.detach().numpy()}")
    print(f"Max Velocity set to: {actor.max_v} m/s")
    print(f"Max Angular Velocity set to: {actor.max_w} rad/s")
