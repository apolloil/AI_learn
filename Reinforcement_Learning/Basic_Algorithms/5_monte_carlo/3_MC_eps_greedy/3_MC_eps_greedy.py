"""
用于离散环境的MC epsilon-greedy算法
"""

import gymnasium as gym
import numpy as np
from utils.timer import timer
from utils.RLrender import RLrender


@timer
def monte_carlo_epsilon_greedy(env, render_env, train_episodes, gamma, initial_epsilon, min_epsilon, video_interval):

    # MC epsilon-greedy算法
    num_state = env.observation_space.n
    num_action = env.action_space.n

    # 初始化Q表和N表
    Q = N = np.zeros((num_state, num_action))
    g_history = []

    for episode in range(train_episodes):
        # 动态调整探索率
        epsilon = initial_epsilon - (initial_epsilon - min_epsilon) * episode / train_episodes

        # 初始化环境
        state = env.reset()[0]
        episode_data = []
        terminated = False
        truncated = False

        tot_g = 0
        cnt = 0
        while not (terminated or truncated):
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            tot_g += float(reward) * gamma ** cnt
            cnt += 1
            episode_data.append((state, action, reward))
            state = next_state

        # 记录g值
        g_history.append(tot_g)

        # 更新Q表
        tem_g = 0
        for t in range(len(episode_data) - 1, -1, -1):
            state, action, reward = episode_data[t]
            tem_g = reward + gamma * tem_g
            N[state, action] += 1
            Q[state, action] += (tem_g - Q[state, action]) / N[state, action]

        # 如果处于录制间隔，则根据当前策略录制视频
        if episode % video_interval == 0:
            render_env.pe(lambda state: np.argmax(Q[state]), gamma, f"train_{episode}")

    return Q, g_history


if __name__ == "__main__":

    # 设置训练环境
    env_name = 'FrozenLake-v1'
    env_params = {
        'map_name': "4x4",
        'is_slippery': False
    }

    # 设置训练参数
    train_episodes = 100000
    gamma = 0.8
    initial_epsilon = 0.6
    min_epsilon = 0.05
    video_interval = 50000  # 录制视频间隔

    # 创建训练环境
    env = gym.make(env_name, **env_params, render_mode=None)
    # 创建render环境
    render_env = RLrender(env_name, env_params)

    # 算法
    Q, g_history = monte_carlo_epsilon_greedy(env, render_env, train_episodes,
                                              gamma, initial_epsilon, min_epsilon, video_interval)

    # 关闭环境
    env.close()

    # 绘制学习曲线
    render_env.plot_learning(g_history)

    # 进行测试
    g_test = render_env.pe(lambda state: np.argmax(Q[state]), gamma, f"test")
    print(f"测试完成！最终奖励为：{g_test}")

