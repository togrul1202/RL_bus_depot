import os
from pathlib import Path
import time

import numpy as np
import gymnasium
import gym_depot

from stable_baselines3 import PPO, A2C
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from gym_depot.utils import params

sb3_model = params['sb3_model']
training = params['training']
ent_coef = params['ent_coef']
tot_ts = params['total_steps']
checkpoint = params['checkpoint']
models_dir = params['models_dir'] + training + sb3_model + '-' + str(int(time.time()))
log_dir = params['logs_dir'] + training + sb3_model + '-' + str(int(time.time()))
if sb3_model == 'A2C':
    model_name = A2C
elif sb3_model == 'PPO':
    model_name = PPO
elif sb3_model == 'MPPO':
    model_name = MaskablePPO

if not checkpoint:
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

if params['tr_render']:
    env = gymnasium.make('BusDepot-v0', render_mode='human')
else:
    env = gymnasium.make('BusDepot-v0')


def mask_fn(env: gymnasium.Env) -> np.ndarray:
    return env.get_action_mask()


if params['mask']:
    env = ActionMasker(env, mask_fn)                        # if mask is not defined MaskablePPO acts as PPO
env.reset()

if checkpoint:
    model = model_name.load(checkpoint)
    model.set_env(env)
    first_step = int(Path(checkpoint).stem)
    models_dir = checkpoint.replace('/' + Path(checkpoint).name, '')
    print(f'Resuming training from the step {first_step}')
else:
    model = model_name("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device=params['device'], ent_coef=ent_coef)
    first_step = 0

t = time.time()
time_steps = 10000
for i in range(1, tot_ts+1):
    model.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name=sb3_model)
    model.save(f'{models_dir}/{first_step + time_steps*i}')
print(f"Time spent : {time.time()-t:.2f}s")
