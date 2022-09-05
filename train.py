import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
from os.path import exists
import time
from ARPOD_Gymenv import HCW_ARPOD
from tensorflow.keras.callbacks import TensorBoard
from os import cpu_count
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

n_cores = cpu_count()
print(f'Number of Logical CPU cores: {n_cores}')

n_envs = 8

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
x0 = [8.0, -8.0, 8.0, 0.0, 0.0, 0.0]
x1 = [-10.0, -5.0, 5.0, 0.0, 0.0, 0.0] 
x2 = [10.0, -5.0, -5.0, 0.0, 0.0, 0.0] 
x3 = [10.0, 5.0, -5.0, 0.0, 0.0, 0.0]

X = [x0,x1,x2,x3]
#lambda
#env = SubprocVecEnv([HCW_ARPOD(state) for state in X])

env = HCW_ARPOD(x0)
env.reset()


#v6 n_steps=6500

model = PPO("MlpPolicy", env, 
             learning_rate=0.03,
             n_epochs=5,
             n_steps=1000,
             batch_size=25, 
             clip_range_vf=None,
             clip_range=0.2,
             ent_coef=0.035,
             vf_coef=0.50,verbose=1, tensorboard_log=log_dir)

info_list = list()
#13000
for i in range(5):
    model.learn(total_timesteps=13000, reset_num_timesteps=False, tb_log_name=model_name)
    info_list.append(env.episode_data)
    model.save(model_dir)

for data in info_list:
    print(data)

env.close()

#print(os.listdir())
