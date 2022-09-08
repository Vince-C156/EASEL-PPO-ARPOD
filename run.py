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
import glob
from PIL import Image

n_cores = cpu_count()
print(f'Number of Logical CPU cores: {n_cores}')


X, Y, Z = [], [], []
x0 = np.asarray([-9.0, 8.0, 10.0, 0.0, 0.0, 0.0], dtype=np.float64) 
env = HCW_ARPOD(x0)
#obs = x0
obs = env.reset()

#obs = x0
"""
X.append(obs[0])
Y.append(obs[1])
Z.append(obs[2])
"""
model_name = "ARPODv26"
model_dir =f'model_export/{model_name}.zip'
model = PPO.load(model_dir, env=env)

if os.path.exists(f'{model_name}_images/') == False:
    os.mkdir(f'{model_name}_images/')

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

region = pyramid(env.theta1, env.theta2, ax, False)

region.plot_LOS()
#ax, center_x,center_y,radius,height_z
plot_target(ax, 0, 0, 2, 4)
order = 0

for step in range(0,N,5):
    if step == 0:
        plt.savefig(f'{model_name}_images/{order:008}')
    else:
        ax.set_xlabel(f'time {step} seconds')
        plot_path(fig, ax, X, Y, Z, info, env.theta1, env.theta2,step)
        plt.savefig(f'{model_name}_images/{order:008}')
    order+=1


fp_out = f"{model_name}.gif"
imgs = (Image.open(f) for f in sorted(glob.glob(f'{model_name}_images/*.png')))
img = next(imgs)  # extract first image from iterator
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=2, loop=0)
print("FINAL OBS")
print(obs)
