import matplotlib.pyplot as plt
import numpy as np
import gym
from stable_baselines3 import PPO
from ARPOD_Gymenv import HCW_ARPOD
import os   
import time
from visualize import plot_path
from visualize import plot_target
from os import cpu_count
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from pyramid_LOS import pyramid

n_cores = cpu_count()
print(f'Number of Logical CPU cores: {n_cores}')


X, Y, Z = [], [], []
x0 = np.asarray([10.0, 5.0, 10.0, 0.0, 0.0, 0.0], dtype=np.float64) 
env = HCW_ARPOD(x0)
obs = x0
#obs = env.reset()

obs = x0
"""
X.append(obs[0])
Y.append(obs[1])
Z.append(obs[2])
"""
model_dir ="model_export/ARPODv10.zip"
model = PPO.load(model_dir, env=env)

done = False

while done == False:
    action, _states = model.predict(obs)

    obs, rewards, done, info = env.step(action)

    X.append(obs[0]) 
    Y.append(obs[1]) 
    Z.append(obs[2])

    stats = env.episode_data
    print(stats)

N = stats["current step"]

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
plt.legend(loc="upper right")

region = pyramid(env.theta1, env.theta2, ax)

region.plot_LOS()
plot_target(ax, 0, 0, 1, 1)

for step in range(0,N,5):
    if step == 0:
        pass
    else:
        ax.set_xlabel(f'time {step} seconds')
        plot_path(fig, ax, X, Y, Z, info, env.theta1, env.theta2,step)

print("FINAL OBS")
print(obs)
