import numpy as np
import torch
from env import MonocularUAVEnv
from ddpg import DDPGAgent, OUActionNoise
from config import cfg


def main():
    print(f"Initializing Training on {cfg.DEVICE}...")
    env = MonocularUAVEnv()

    # 状态维度4，动作维度2
    agent = DDPGAgent(
        state_dim=4,
        action_dim=2,
        max_action=np.array([cfg.V_MAX, cfg.YAW_RATE_MAX])
    )

    noise = OUActionNoise(mean=np.zeros(2), std_deviation=float(0.2) * np.ones(2))
    rewards_history = []

    for episode in range(cfg.MAX_EPISODES):
        state, _ = env.reset()
        noise.reset()
        episode_reward = 0

        for step in range(cfg.MAX_STEPS):
            if len(agent.memory) < cfg.WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)

            if len(agent.memory) > cfg.WARMUP_STEPS:
                agent.update()

            state = next_state
            episode_reward += reward

            if done:
                break

        rewards_history.append(episode_reward)
        avg_rew = np.mean(rewards_history[-20:]) if rewards_history else 0
        print(f"Ep {episode + 1}/{cfg.MAX_EPISODES} | Reward: {episode_reward:.2f} | Avg(20): {avg_rew:.2f}")

    # 保存模型
    torch.save(agent.actor.state_dict(), cfg.MODEL_ACTOR_PATH)
    torch.save(agent.critic.state_dict(), cfg.MODEL_CRITIC_PATH)
    print(f"Models saved to {cfg.MODEL_ACTOR_PATH} & {cfg.MODEL_CRITIC_PATH}")
    env.close()


if __name__ == "__main__":
    main()