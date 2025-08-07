import gymnasium as gym
import time

env_name = 'CartPole-v1'
env = gym.make(env_name, render_mode="human")

state = env.reset()

terminated = False
truncated = False
total_reward = 0

while not (terminated or truncated):
    env.render()
    env.render()
    action = env.action_space.sample()
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    time.sleep(0.2)
print(total_reward)

