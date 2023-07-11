import os
import time

import gymnasium
import gym_depot
from stable_baselines3 import PPO

models_dir = f'models/training4/PPO-{int(time.time())}'
log_dir = f'logs/training4/PPO-{int(time.time())}'

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

env = gymnasium.make('BusDepot-v0')
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cpu")

t = time.time()
time_steps = 10000
for i in range(1, 151):
    model.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f'{models_dir}/{time_steps*i}')
print(f"Time spent : {time.time()-t:.2f}s")
