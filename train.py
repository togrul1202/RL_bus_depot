import gymnasium
import gym_depot
from stable_baselines3 import A2C

env = gymnasium.make('BusDepot-v0')
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

