"""
三个功能：
1.提供用于录制训练过程的单个episode的接口
2.提供用于录制测试过程的接口
3.提供用于绘制学习曲线的接口
"""

import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo


class RLrender:
    def __init__(self, env_name, env_params, folder_name="videos"):
        self.env_name = env_name
        self.env_params = env_params
        self.folder_name = folder_name
        os.makedirs(folder_name, exist_ok=True)

    def pe(self, policy_func, gamma, video_name):
        """针对policy函数录制单个episode进行评估"""
        env = RecordVideo(
            gym.make(self.env_name, **self.env_params, render_mode="rgb_array"),
            video_folder=self.folder_name,
            episode_trigger=lambda _: True,
            name_prefix=video_name,
            disable_logger=True
        )

        state, _ = env.reset()
        terminated = False
        truncated = False
        g = 0
        cnt = 0
        while not (terminated or truncated):
            action = policy_func(state)
            state, reward, terminated, truncated, _ = env.step(action)
            g += float(reward) * gamma ** cnt
            cnt += 1
        env.close()
        return g

    @staticmethod
    def plot_learning(g_history, save_path="learning_curve.png"):
        """绘制并保存学习曲线(采用滑动平均)"""
        plt.figure(figsize=(10, 5))
        plt.plot(np.convolve(g_history, np.ones(100) / 100, mode='valid'))
        plt.title("Average Return(for 100 episodes)")
        plt.xlabel("Episode")
        plt.ylabel("Avg Return")
        plt.savefig(save_path)

