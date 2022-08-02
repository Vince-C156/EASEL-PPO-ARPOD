from stable_baselines3.common.env_checker import check_env
from ARPOD_Gymenv import HCW_ARPOD

env = HCW_ARPOD([100.0, 100.0, 100.0, 1.0, 2.0, 0.0])

check_env(env)
