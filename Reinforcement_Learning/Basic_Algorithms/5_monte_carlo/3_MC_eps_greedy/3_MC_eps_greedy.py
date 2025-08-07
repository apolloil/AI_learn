import gymnasium as gym
import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from gymnasium.wrappers import RecordVideo

# 禁用matplotlib弹窗
plt.switch_backend('agg')


def epsilon_greedy_policy(q_table, state, epsilon, n_actions):
    """Epsilon-greedy策略选择动作"""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        return np.argmax(q_table[state])


def create_video_env(env_name, env_params, video_folder, video_prefix):
    """创建视频录制环境"""
    env = gym.make(env_name, **env_params, render_mode='rgb_array')
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda ep_idx: True,  # 录制所有传递到这个环境的episode
        name_prefix=video_prefix,
        disable_logger=True
    )
    return env


def monte_carlo_epsilon_greedy(env, env_params, num_episodes=10000, gamma=0.99,
                               initial_epsilon=0.5, min_epsilon=0.01,
                               video_interval=500, video_folder="videos"):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_table = np.zeros((n_states, n_actions))
    g_sum = defaultdict(lambda: np.zeros(n_actions))
    visit_count = defaultdict(lambda: np.zeros(n_actions))
    success_history = []

    # 创建视频目录
    os.makedirs(video_folder, exist_ok=True)

    # 用于存储每个录制环境的列表
    video_envs = {}

    for episode in range(num_episodes):
        # 动态调整探索率
        epsilon = max(min_epsilon, initial_epsilon * (1 - episode / (num_episodes * 1.2)))

        # 检查是否需要录制当前episode
        if episode % video_interval == 0:
            # 创建新的录制环境 - 只录制当前episode
            video_prefix = f"training_episode_{episode}"

            # 创建录制环境
            video_env = create_video_env(
                'FrozenLake-v1',
                env_params,
                video_folder,
                video_prefix
            )

            print(f"创建录制环境用于episode {episode}")
            # 使用录制环境
            current_env = video_env

            # 存储录制环境以便后续关闭
            video_envs[episode] = video_env
        else:
            # 使用普通训练环境
            current_env = env

        # 重置环境
        state, _ = current_env.reset()
        episode_data = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = epsilon_greedy_policy(q_table, state, epsilon, n_actions)
            next_state, reward, terminated, truncated, _ = current_env.step(action)
            episode_data.append((state, action, reward))
            state = next_state

        # 记录是否成功
        success = 1 if reward > 0 else 0
        success_history.append(success)

        # 如果是录制环境，立即关闭（因为只用于一个episode）
        if episode in video_envs:
            video_envs[episode].close()
            del video_envs[episode]

        # 蒙特卡洛更新（首次访问法）
        G = 0
        visited = set()
        for t in range(len(episode_data) - 1, -1, -1):
            state, action, reward = episode_data[t]
            G = gamma * G + reward

            if (state, action) not in visited:
                visited.add((state, action))
                visit_count[state][action] += 1
                g_sum[state][action] += G
                q_table[state][action] = g_sum[state][action] / visit_count[state][action]

        # 打印进度
        if episode % 500 == 0:
            recent_success = np.mean(success_history[-100:]) if success_history else 0
            print(f"Episode {episode}/{num_episodes} | "
                  f"Success Rate: {recent_success:.2f} | "
                  f"Epsilon: {epsilon:.4f}")

    # 训练结束时关闭任何可能未关闭的录制环境
    for env in video_envs.values():
        env.close()

    return q_table, success_history


def evaluate_policy(env_name, Q, num_episodes=5, video_folder="evaluation_videos", **env_kwargs):
    """评估策略并录制视频"""
    os.makedirs(video_folder, exist_ok=True)

    # 创建视频录制环境
    video_prefix = "evaluation"
    eval_env = create_video_env(
        env_name,
        env_kwargs,
        video_folder,
        video_prefix
    )

    wins = 0
    paths = []

    for ep in range(num_episodes):
        state, _ = eval_env.reset()
        terminated = False
        truncated = False
        path = [state]
        steps = 0

        while not (terminated or truncated):
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = eval_env.step(action)
            path.append(state)
            steps += 1
            time.sleep(0.1)  # 控制视频速度

        if reward > 0:
            wins += 1
            print(f"Episode {ep + 1}: 成功! 步数: {steps} | 路径: {path}")
        else:
            print(f"Episode {ep + 1}: 失败! 最终位置: {state}")

        paths.append(path)

    eval_env.close()
    return wins / num_episodes, paths


if __name__ == "__main__":
    # 设置训练环境参数（super）
    env_params = {
        'map_name': "8x8",
        'is_slippery': True
    }
    # 创建训练环境
    env = gym.make('FrozenLake-v1', **env_params, render_mode=None)

    # 设置训练参数(super)
    num_episodes = 500000
    gamma = 0.8
    initial_epsilon = 0.6
    video_interval = 50000  # 录制视频间隔

    print("=" * 50)
    print("开始训练 - FrozenLake Monte Carlo Epsilon-Greedy")
    print(f"总训练回合: {num_episodes} | 折扣因子: {gamma} | 初始探索率: {initial_epsilon}")
    print(f"将录制episode 0, 500, 1000, 1500...的视频")
    print("=" * 50)

    # 训练模型
    start_time = time.time()
    Q_table, success_history = monte_carlo_epsilon_greedy(
        env,
        env_params=env_params,
        num_episodes=num_episodes,
        gamma=gamma,
        initial_epsilon=initial_epsilon,
        video_interval=video_interval
    )
    end_time = time.time()

    # 评估模型
    print("\n训练完成! 开始可视化测试并录制评估视频...")
    win_rate, paths = evaluate_policy('FrozenLake-v1', Q_table, num_episodes=5, **env_params)

    # 打印结果
    print(f"\n训练用时: {end_time - start_time:.2f}秒")
    print(f"最终成功率: {win_rate * 100:.2f}%")
    print("成功路径示例:", [p for p in paths if p[-1] == 15][:1])

    # 保存学习曲线
    plt.figure(figsize=(10, 5))
    plt.plot(np.convolve(success_history, np.ones(100) / 100, mode='valid'))
    plt.title("训练成功率(100-episodes平均)")
    plt.xlabel("训练回合")
    plt.ylabel("成功率")
    plt.savefig("learning_curve.png")
    print("学习曲线已保存为 learning_curve.png")

    env.close()