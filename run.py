import matplotlib.pyplot as plt
import numpy as np
import gym
from stable_baselines3 import PPO
from ARPOD_Gymenv import HCW_ARPOD
import os   
import time
from visualize import plot_path


X, Y, Z = [], [], []

env = HCW_ARPOD([200.0, 100.0, 250.0, 0.0, 0.0, 0.0])
obs = env.reset()

X.append(obs[0])
Y.append(obs[1])
Z.append(obs[2])

model_dir ="model_out/ARPOD_AGENT3.zip"
model = PPO.load(model_dir, env=env)

done = False

while done == False:
    action, _states = model.predict(obs)

    obs, rewards, done, info = env.step(action)

    X.append(obs[0]) 
    Y.append(obs[1]) 
    Z.append(obs[2])

plot_path(X, Y, Z)

print("FINAL OBS")
print(obs)
