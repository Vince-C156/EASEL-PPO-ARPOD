import matplotlib.pyplot as plt
import numpy as np
import gym
from stable_baselines3 import PPO
from ARPOD_Gymenv import HCW_ARPOD
import os   
import time
from visualize import plot_path
import json

n_episodes = 1000

x0 = np.asarray([-1000.0, -2000.0, -1000.0, 0.0, 0.0, 0.0], dtype=np.float64) 
env = HCW_ARPOD(x0)
obs = env.reset()

"""
print(f'Total episodes evaluated: {n_episodes}')
print(f'Total steps over all episodes: {T_eval}')
print(f'Averge steps per episode: {T_eval/n_episodes}')
print(f'Percentage of time within the LOS overall {totalsteps_LOS / T_eval} with {totalsteps_LOS} LOS seconds total')
print(f'Total incompleted missions: {n_failed}')
print(f'Total incompletions due to obstical collision: {n_obstical_collision}')
print(f'Total incompletions due to time: {n_notime}')
print(f'Total incompletions for leaving the mission region: {n_outbounds}')
print(f'Averge fuel per episode: {total_fuel/n_episodes} m/s')
"""

"""
        env.episode_data = {"x0" : self.x0,
                             "last u": self.u,
                             "current step" : 0,
                             "steps in LOS" : 0,
                             "step reward" : 0,
                             "episode reward" : 0,
                             "ending condition" : "running",
                             "mission success" : False,
                             "ending state" : list()}
"""

name = 'ARPODv12'
model_dir = f'model_export/{name}.zip'
model = PPO.load(model_dir, env=env)

totalsteps_LOS = 0
T_eval = 0
n_won = 0
n_failed = 0
n_obstical_collision = 0
n_notime = 0
n_outbounds = 0
total_fuel = 0

for i in range(n_episodes):

    done = False
    while done == False:

        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        stats = env.episode_data

    if stats["mission success"] != True:
        n_failed += 1
        if stats["ending condition"] == "obstacle_collision":
            n_obstical_collision += 1
        elif stats["ending condition"] == "out_bounds":
            n_outbounds += 1
        elif stats["ending condition"] == "step_limit":
            n_notime += 1
        else:
            n_won += 1

        #will add target collision

    T_inLOS = stats["steps in LOS"]
    T_episode = stats["current step"]
    fuel = stats["total fuel"]

    T_eval += T_episode
    totalsteps_LOS += T_inLOS
    total_fuel += fuel

    print('EVALUATION METRICS')
    print()
    print(f'Episode {i+1} stats')
    print('====================')
    print([print(f'{key}: {val}') for key, val in stats.items()])
    print('====================')
    print()

print('Overall Evaluation')
print('====================')
print(f'Total episodes evaluated: {n_episodes}')
print(f'Total steps over all episodes: {T_eval}')
print(f'Averge steps per episode: {T_eval/n_episodes}')
print(f'Percentage of time within the LOS overall {totalsteps_LOS / T_eval} with {totalsteps_LOS} LOS seconds total')
print(f'Total incompleted missions: {n_failed}')
print(f'Total incompletions due to obstical collision: {n_obstical_collision}')
print(f'Total incompletions due to time: {n_notime}')
print(f'Total incompletions for leaving the mission region: {n_outbounds}')
print(f'Averge fuel per episode: {total_fuel/n_episodes} m/s')
print('====================')
print('Model Params')
print('====================')
for key, val in model.get_parameters().items():
    if key == 'param_groups':
        [print(f'{k}: {v}') for k, v in val[0].items()]


#print([print(f'{key}: {val}') for key, val in model.get_parameters().items() if ("policy" in key or "shared_net" in key or "action" in key) != False])
print('====================')


json_dict = {"Total episodes evaluated " : n_episodes, 
             "Total steps over all episodes" : T_eval, 
             "Percentage of time within the LOS overall" : totalsteps_LOS / T_eval,
             "Total incompleted missions" : n_failed,
             "Total incompletions due to obstical collision" : n_obstical_collision,
             "Total incompletions due to time" : n_notime,
             "Total incompletions for leaving the mission region" : n_outbounds, 
             "Averge steps per episode " : T_eval/n_episodes,
             "Averge fuel per episode" : total_fuel/n_episodes,}

#json_object = json.dumps(json_dict, indent = 4)

with open(f'{name}.json', "w") as outfile:
    json.dump(json_dict, outfile)

#plot_path(X, Y, Z, info)

#print("FINAL OBS")
#print(obs)
