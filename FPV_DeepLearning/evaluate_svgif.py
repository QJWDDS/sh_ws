import torch
import numpy as np
from env import MonocularUAVEnv
from ddpg import DDPGAgent
from config import cfg
import os
import imageio


def main():
    if not os.path.exists(cfg.MODEL_ACTOR_PATH):
        print("Error: Model file not found. Please run train.py first.")
        return

    # [新增] 创建保存 GIF 的文件夹
    gif_dir = "gifs"
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
        print(f"Created directory '{gif_dir}' for saving GIFs.")

    print("Starting Evaluation with Visualization...")
    # 开启 render_mode='human'
    env = MonocularUAVEnv(render_mode='human')

    agent = DDPGAgent(
        state_dim=4,
        action_dim=2,
        max_action=np.array([cfg.V_MAX, cfg.YAW_RATE_MAX])
    )

    # 加载权重
    agent.actor.load_state_dict(torch.load(cfg.MODEL_ACTOR_PATH, map_location=cfg.DEVICE))
    agent.actor.eval()  # 切换到评估模式

    episodes_to_watch = 20  # [修改] 示例设为 5，避免生成过多文件，可根据需要调整
    for i in range(episodes_to_watch):
        state, _ = env.reset()
        print(f"Evaluation Episode {i + 1}/{episodes_to_watch}")

        frames = []  # [新增] 用于存储当前回合的帧
        done = False

        while not done:
            # 评估时不加噪声
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)

            # 渲染画面
            env.render()

            # [新增] 捕获当前帧
            if env.fig:
                try:
                    # 强制刷新画布（确保获取最新图像）
                    env.fig.canvas.draw()

                    # 获取 RGBA 缓冲区 (Matplotlib 3.8+ 推荐方式)
                    # 注意：如果环境运行在无显示器的服务器上，这里可能需要调整 backend
                    image = np.array(env.fig.canvas.buffer_rgba())

                    # 去除 Alpha 通道 (RGBA -> RGB) 并存入列表
                    frames.append(image[:, :, :3])
                except AttributeError:
                    print("Warning: Could not capture frame. Ensure Matplotlib backend supports buffer access.")

            done = terminated or truncated
            if terminated:
                print(" -> Intercepted or Crashed")

        # [新增] 保存当前回合为 GIF
        if len(frames) > 0:
            gif_path = os.path.join(gif_dir, f'eval_episode_{i + 1}.gif')
            # fps=10 表示每秒 10 帧，可根据 dt=0.1 调整为更平滑的效果
            imageio.mimsave(gif_path, frames, fps=15, loop=0)
            print(f" -> Saved GIF to {gif_path}")

    print("Evaluation finished.")
    env.close()


if __name__ == "__main__":
    main()