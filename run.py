import matplotlib.pyplot as plt
import numpy as np
import gym
from stable_baselines3 import PPO
from ARPOD_Gymenv import HCW_ARPOD
import os   
import time


env = HCW_ARPOD([-200.0, 100.0, 200.0, 0.0, 0.0, 0.0])
obs = env.reset()


model_dir ="model_out/PPO2.zip"
model = PPO.load(model_dir, env=env)


action, _states = model.predict(obs)

obs, rewards, done, info = env.step(action)
