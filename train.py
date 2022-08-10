import gym
from stable_baselines3 import PPO
import os
import time
from ARPOD_Gymenv import HCW_ARPOD
from tensorflow.keras.callbacks import TensorBoard



model_dir = "model_out/ARPODv1.0"
log_dir = "model_out/ARPOD_v1.0/logs"
tensorboard = TensorBoard(log_dir=log_dir)
x0 = [1000.0, 500.0, 550.0, 0.0, 0.0, 0.0] 
env = HCW_ARPOD(x0) 
env.reset()
 
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

for i in range(500):
    model.learn(total_timesteps=13000, reset_num_timesteps=False, tb_log_name="ARPODv1.0")
    model.save(model_dir)

env.close()

#print(os.listdir())
