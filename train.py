import os
import time

import gymnasium
import gym_depot
from stable_baselines3 import PPO, A2C

from gym_depot.utils import params

sb3_model = params['sb3_model']
training = params['training']
models_dir = params['models_dir'] + training + sb3_model + '-' + str(int(time.time()))
log_dir = params['logs_dir'] + training + sb3_model + '-' + str(int(time.time()))

os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

if params['tr_render']:
    env = gymnasium.make('BusDepot-v0', render_mode='human')
else:
    env = gymnasium.make('BusDepot-v0')

env.reset()
model_name = A2C if sb3_model == 'A2C' else PPO
model = model_name("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device=params['device'], ent_coef=0.001)

t = time.time()
time_steps = 10000
for i in range(1, 1501):
    model.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name=sb3_model)
    model.save(f'{models_dir}/{time_steps*i}')
print(f"Time spent : {time.time()-t:.2f}s")
