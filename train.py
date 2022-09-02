import gym
from stable_baselines3 import PPO
import os
from os.path import exists
import time
from ARPOD_Gymenv import HCW_ARPOD
from tensorflow.keras.callbacks import TensorBoard

#os.mkdir("model_export")

path = os.getcwd()
dir_list = os.listdir(path + "/model_export")

print(dir_list)
nozip_list = [filename for filename in dir_list if filename.count('.zip') == 0]
version_n = len(nozip_list) + 1

model_name = f"ARPODv{version_n}"

print(model_name)
os.mkdir(f"model_export/{model_name}")
os.mkdir(f"model_export/{model_name}/logs")

model_dir = f"model_export/{model_name}"
log_dir = f"model_export/{model_name}/logs"
tensorboard = TensorBoard(log_dir=log_dir)
x0 = [1000.0, 500.0, 550.0, 0.0, 0.0, 0.0] 
env = HCW_ARPOD(x0) 
env.reset()
 
model = PPO("MlpPolicy", env, 
             learning_rate=0.03,
             n_epochs=15,
             n_steps=3250,
             batch_size=25, 
             clip_range_vf=None,
             clip_range=0.2,
             ent_coef=0.025,
             vf_coef=0.03,verbose=1, tensorboard_log=log_dir)

info_list = list()
#13000
for i in range(35):
    model.learn(total_timesteps=13000, reset_num_timesteps=False, tb_log_name=model_name)
    info_list.append(env.episode_data)
    model.save(model_dir)

for data in info_list:
    print(data)

env.close()

#print(os.listdir())
