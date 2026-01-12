import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import matplotlib.image as mpimg
from config import cfg


class MonocularUAVEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None):
        super(MonocularUAVEnv, self).__init__()

        # 动作: [线速度 v, 偏航角速度 omega]
        self.action_space = spaces.Box(
            low=np.array([0.0, -cfg.YAW_RATE_MAX]),
            high=np.array([cfg.V_MAX, cfg.YAW_RATE_MAX]),
            dtype=np.float32
        )

        # 观测: [无人机v, 无人机omega, 视线角偏差, 可见性标志]
        # 核心约束：观测不到距离信息
        self.observation_space = spaces.Box(
            low=np.array([0, -cfg.YAW_RATE_MAX, -np.pi, 0]),
            high=np.array([cfg.V_MAX, cfg.YAW_RATE_MAX, np.pi, 1]),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.fig, self.ax = None, None

        # 加载资源
        try:
            self.uav_img = mpimg.imread('UAV.png')
        except FileNotFoundError:
            print("Warning: UAV.png not found, using shapes instead.")
            self.uav_img = None

    def reset(self, seed=None):
        super().reset(seed=seed)

        # 初始化无人机: [x, y, theta]
        self.uav_state = np.array([0.0, 0.0, np.pi / 2])
        self.uav_vel = 0.0

        # 初始化目标 (确保初始在视野内)
        valid_spawn = False
        while not valid_spawn:
            t_x = np.random.uniform(-cfg.FIELD_SIZE / 2, cfg.FIELD_SIZE / 2)
            t_y = np.random.uniform(-cfg.FIELD_SIZE / 2, cfg.FIELD_SIZE / 2)

            dx = t_x - self.uav_state[0]
            dy = t_y - self.uav_state[1]
            dist = np.hypot(dx, dy)
            angle_to_target = np.arctan2(dy, dx)
            angle_diff = self._wrap_angle(angle_to_target - self.uav_state[2])

            if dist < cfg.MAX_SENSE_DIST and abs(angle_diff) < cfg.FOV / 2 and dist > 2.0:
                valid_spawn = True
                self.target_pos = np.array([t_x, t_y])
                vel_angle = np.random.uniform(-np.pi, np.pi)
                v_mag = np.random.uniform(0.5, cfg.TARGET_V_MAX)
                self.target_vel = np.array([np.cos(vel_angle) * v_mag, np.sin(vel_angle) * v_mag])

        # 轨迹记录
        self.uav_traj_x = [self.uav_state[0]]
        self.uav_traj_y = [self.uav_state[1]]
        self.target_traj_x = [self.target_pos[0]]
        self.target_traj_y = [self.target_pos[1]]

        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        # 1. 动作执行与物理更新
        cmd_v = np.clip(action[0], 0, cfg.V_MAX)
        cmd_omega = np.clip(action[1], -cfg.YAW_RATE_MAX, cfg.YAW_RATE_MAX)

        # 简单的一阶惯性模拟
        self.uav_vel += (cmd_v - self.uav_vel) * 0.2

        # 无人机运动学
        self.uav_state[2] += cmd_omega * cfg.DT
        self.uav_state[2] = self._wrap_angle(self.uav_state[2])
        self.uav_state[0] += self.uav_vel * np.cos(self.uav_state[2]) * cfg.DT
        self.uav_state[1] += self.uav_vel * np.sin(self.uav_state[2]) * cfg.DT

        # 目标运动学 (边界反弹)
        self.target_pos += self.target_vel * cfg.DT
        limit = cfg.FIELD_SIZE / 2
        if abs(self.target_pos[0]) > limit: self.target_vel[0] *= -1
        if abs(self.target_pos[1]) > limit: self.target_vel[1] *= -1

        self.uav_traj_x.append(self.uav_state[0])
        self.uav_traj_y.append(self.uav_state[1])
        self.target_traj_x.append(self.target_pos[0])
        self.target_traj_y.append(self.target_pos[1])

        # 2. 计算相对关系
        dx = self.target_pos[0] - self.uav_state[0]
        dy = self.target_pos[1] - self.uav_state[1]
        dist = np.hypot(dx, dy)
        angle_to_target = np.arctan2(dy, dx)
        angle_error = self._wrap_angle(angle_to_target - self.uav_state[2])
        is_visible = (dist < cfg.MAX_SENSE_DIST) and (abs(angle_error) < cfg.FOV / 2)

        # 3. 奖励计算
        reward = -0.01  # 时间惩罚 0.01
        terminated = False
        truncated = False

        if is_visible:
            # 视线对准奖励
            reward += 0.1 * (1.0 - abs(angle_error) / (cfg.FOV / 2))
            # 距离逼近奖励 (引导Agent学会加速)
            reward += (cfg.MAX_SENSE_DIST - dist) * 0.01   #0.01

            if dist < cfg.CAPTURE_DIST:
                reward += 2000.0  #20
                terminated = True
        else:
            reward -= 0.5  # 丢失目标惩罚

        # 撞墙惩罚
        if abs(self.uav_state[0]) > limit or abs(self.uav_state[1]) > limit:
            reward -= 10
            terminated = True

        self.steps += 1
        if self.steps >= cfg.MAX_STEPS:
            truncated = True

        return self._get_obs(angle_error, is_visible), reward, terminated, truncated, {}

    def _get_obs(self, angle_error=0.0, is_visible=False):
        # 模拟单目视觉：不可见时角度误差为0（无信息）
        obs_angle = angle_error if is_visible else 0.0
        return np.array([
            self.uav_vel,
            0.0,  # 预留omega位
            obs_angle,
            1.0 if is_visible else 0.0
        ], dtype=np.float32)

    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def render(self):
        if self.render_mode != 'human': return

        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_aspect('equal')

        self.ax.cla()
        limit = cfg.FIELD_SIZE / 2 + 2
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_title("Monocular UAV Interception")

        # 绘制目标
        self.ax.add_patch(patches.Circle(self.target_pos, 0.3, color='red', label='Target'))

        # 绘制 FOV 扇形
        dx = self.target_pos[0] - self.uav_state[0]
        dy = self.target_pos[1] - self.uav_state[1]
        dist = np.hypot(dx, dy)
        angle_err = self._wrap_angle(np.arctan2(dy, dx) - self.uav_state[2])
        visible = (dist < cfg.MAX_SENSE_DIST) and (abs(angle_err) < cfg.FOV / 2)

        wedge = patches.Wedge(
            (self.uav_state[0], self.uav_state[1]), cfg.MAX_SENSE_DIST,
            np.degrees(self.uav_state[2] - cfg.FOV / 2),
            np.degrees(self.uav_state[2] + cfg.FOV / 2),
            alpha=0.2, color='green' if visible else 'orange'
        )
        self.ax.add_patch(wedge)

        # 绘制轨迹
        self.ax.plot(self.uav_traj_x, self.uav_traj_y, 'b-', alpha=0.5)
        self.ax.plot(self.target_traj_x, self.target_traj_y, 'r--', alpha=0.5)

        # 绘制 UAV 图片或代用图形
        if self.uav_img is not None:
            sz = 1.5
            tr = transforms.Affine2D().rotate_deg_around(
                self.uav_state[0], self.uav_state[1],
                np.degrees(self.uav_state[2]) - 90
            ) + self.ax.transData
            self.ax.imshow(
                self.uav_img, transform=tr,
                extent=[self.uav_state[0] - sz / 2, self.uav_state[0] + sz / 2,
                        self.uav_state[1] - sz / 2, self.uav_state[1] + sz / 2],
                zorder=10
            )
        else:
            self.ax.add_patch(patches.Circle((self.uav_state[0], self.uav_state[1]), 0.5, color='blue'))

        # 方向箭头
        self.ax.arrow(
            self.uav_state[0], self.uav_state[1],
            1.0 * np.cos(self.uav_state[2]), 1.0 * np.sin(self.uav_state[2]),
            head_width=0.3, fc='blue', ec='blue'
        )

        plt.pause(0.01)

    def close(self):
        if self.fig: plt.close(self.fig)