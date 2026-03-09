import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import torch

from alg.hdp import HDP
from env.UGV import UGV


def parse_args():
    curr_path = os.path.dirname(os.path.abspath(__file__))
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    default_model_dir = os.path.join(curr_path, "outputs", "UGV-HDP", curr_time, "models")

    parser = argparse.ArgumentParser(description="Train HDP actor-critic network")
    parser.add_argument("--episodes", type=int, default=500, help="training episodes")
    parser.add_argument("--max_steps", type=int, default=300, help="max steps per episode")
    parser.add_argument("--gamma", type=float, default=0.98, help="discount factor")
    parser.add_argument("--critic_lr", type=float, default=5e-4, help="critic learning rate")
    parser.add_argument("--actor_lr", type=float, default=1e-4, help="actor learning rate")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--log_interval", type=int, default=10, help="print interval")
    parser.add_argument("--model_dir", type=str, default=default_model_dir, help="directory to save actor model")
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(cfg):
    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)

    env = UGV()
    agent = HDP(path=cfg.model_dir, env=env, gamma=cfg.gamma, critic_lr=cfg.critic_lr, actor_lr=cfg.actor_lr)

    episode_rewards = []
    best_reward = -float("inf")

    for episode in range(1, cfg.episodes + 1):
        env.reset()
        total_reward = 0.0
        critic_losses = []
        actor_losses = []

        for _ in range(cfg.max_steps):
            critic_loss, actor_loss, reward, done = agent.update()
            total_reward += reward
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
            if done:
                break

        episode_rewards.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(filename="actor_best.pt")

        if episode % cfg.log_interval == 0 or episode == 1:
            avg_reward = float(np.mean(episode_rewards[-cfg.log_interval:]))
            avg_critic_loss = float(np.mean(critic_losses))
            avg_actor_loss = float(np.mean(actor_losses))
            print(
                f"Episode {episode:4d} | "
                f"Reward {total_reward:9.3f} | "
                f"AvgReward {avg_reward:9.3f} | "
                f"CriticLoss {avg_critic_loss:8.4f} | "
                f"ActorLoss {avg_actor_loss:8.4f}"
            )

    final_model_path = agent.save(filename="actor_final.pt")
    print(f"Training finished. BestReward={best_reward:.3f}")
    print(f"Actor model saved to: {final_model_path}")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    train(args)
