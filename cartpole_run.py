import gymnasium as gym
from stable_baselines3 import PPO

# Ortamı render_mode ile oluştur
env = gym.make("CartPole-v1", render_mode="human")

# Modeli PPO algoritmasıyla oluştur
model = PPO("MlpPolicy", env, verbose=1)

# Eğitimi başlat
model.learn(total_timesteps=1000)

# Modeli test et
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()

env.close()
