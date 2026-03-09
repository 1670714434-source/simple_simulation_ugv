import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from alg.actor import Actor
from alg.critic import Critic
from env.UGV import UGV


class HDP:
    def __init__(self, path, env: UGV, gamma=0.98, critic_lr=5e-4, actor_lr=1e-4):
        if env is None:
            raise ValueError("env must not be None")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.path = path
        self.gamma = gamma
        self.criterion = torch.nn.MSELoss()

        state_dim = len(self.env.get_state())
        action_dim = 2

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.actor = Actor(state_dim, action_dim).to(self.device)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

    def choose_action(self, state):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(state_tensor)
        return action.detach().cpu().squeeze(0).numpy()

    def update(self):
        state_np = self.env.get_state()
        env_action = self.choose_action(state_np)
        next_state_np, reward, done = self.env.step(env_action)

        state = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = torch.as_tensor(env_action, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state = torch.as_tensor(next_state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward_tensor = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor([[float(done)]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action = self.actor(next_state)
            next_value = self.critic(next_state, next_action)
            target_value = reward_tensor + self.gamma * next_value * (1.0 - done_tensor)

        current_value = self.critic(state, action)
        critic_loss = self.criterion(current_value, target_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_action = self.actor(state)
        actor_loss = -self.critic(state, actor_action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item(), float(reward), bool(done)

    def save(self, path=None, filename="actor_checkpoint.pt"):
        save_dir = self.path if path is None else path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        torch.save(self.actor.state_dict(), save_path)
        return save_path

    def load(self, path=None, filename="actor_checkpoint.pt"):
        load_dir = self.path if path is None else path
        load_path = os.path.join(load_dir, filename)
        self.actor.load_state_dict(torch.load(load_path, map_location=self.device))


if __name__ == "__main__":
    env = UGV()
    agent = HDP(path="./models", env=env)
    c_loss, a_loss, reward, done = agent.update()
    print(f"critic_loss={c_loss:.4f}, actor_loss={a_loss:.4f}, reward={reward:.4f}, done={done}")
