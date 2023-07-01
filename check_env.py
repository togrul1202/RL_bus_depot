import gymnasium
from stable_baselines3.common.env_checker import check_env
import gym_depot

env = gymnasium.make('BusDepot-v0')
check_env(env)
