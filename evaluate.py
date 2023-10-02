import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import time

import gymnasium
import gym_depot
from gym_depot.envs.depot_env import DepotEnv
from stable_baselines3 import PPO, A2C
from sb3_contrib.ppo_mask import MaskablePPO

from gym_depot.utils import params
from manual import req_act
from manual_inv import req_act_inv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if params['ev_render']:
    env = gymnasium.make('BusDepot-v0', render_mode='human')
else:
    env = gymnasium.make('BusDepot-v0')

result_dir = params['results_dir']
name = params['model_path_ev']
name_dir = Path(name)
filename1 = 'steps-and-actions.png'                     # .svg for thesis
filename2 = 'avg_steps-and-fail.png'

if name != 'manual' and name != 'manual_inv':
    model_name = Path(name).parent.name.split('-')[0]
    if model_name == 'A2C':
        model_name = A2C
    elif model_name == 'PPO':
        model_name = PPO
    elif model_name == 'MPPO':
        model_name = MaskablePPO
    model = model_name.load(name, env=env)
    output_dir = os.path.join(result_dir, name_dir.parent.parent.name, name_dir.parent.name, name_dir.stem)
else:
    output_dir = os.path.join(result_dir, name)
if params['save_graph']:
    os.makedirs(output_dir, exist_ok=True)

episodes = params['episode_num']
fail_list = np.zeros(episodes)
it_num = params['iteration_num']
mean_pun_list = []
fail_percent = []
for it in range(it_num):
    t = time.time()
    episodes_up = episodes
    tot_rew = 0
    rew_diff = []
    act_number = []
    rew = []
    seed = 1
    fail = 0
    exceed_time = []
    fail_dict = {'waiting_limit_exceeded': 0, 'no_valid_action': 0, 'same_cs': 0, 'same_fs': 0, 'crash': 0,
                 'lock_crash': 0, 'stuck': 0, 'wrong_fs': 0, 'wrong_cs': 0}
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed)
        req = obs[-1]
        number = 0
        while True:
            number += 1
            if name == 'manual':
                action = req_act(req, obs)
            elif name == 'manual_inv':
                action = req_act_inv(req, obs)
            elif params['ev_mask']:
                mask = env.get_action_mask()
                action, _ = model.predict(obs, action_masks=mask)
            else:
                action, _ = model.predict(obs)
            #print(f'action:{action}')
            obs, reward, terminated, truncated, info = env.step(action)
            req = obs[-1]
            #print(obs, reward)
            tot_rew += reward
            if terminated or truncated:
                #print(tot_rew)
                rew.append(tot_rew)
                rew_diff.append((reward-tot_rew)/params['min_to_ts'])
                if truncated:
                    episodes_up -= 1
                if reward < 0:
                    if params['check_mask']:
                        print('try again')
                        exceed_time.append(info.get("exceeded time") / params['min_to_ts'])
                    fail += 1
                    fail_dict[info.get('info')] += 1
                    fail_list[episode] += 1
                else:
                    exceed_time.append(info.get("exceeded time")/params['min_to_ts'])
                tot_rew = 0
                act_number.append(number)
                #print(f'number: {number}')
                seed += 1
                #print(f'seed: {seed}')
                break
    mean_rew = mean(rew)
    mean_pun = mean(rew_diff)
    mean_ex = mean(exceed_time)
    mean_number = mean(act_number)
    mean_pun_list.append(mean_pun)
    fail_percent.append(fail*100/episodes)
    print(f'mean rew: {mean_rew}\navg time steps waited: {mean_pun}\navg exceed time: {mean_ex}\nnumber of fails: {fail}/{episodes_up}\nfails: {fail_dict}',
          f'mean number: {mean_number}')
    print(f"Time spent : {time.time() - t:.2f}s")

    if it == 0:
        plt.figure()
        seeds = list(range(1, seed))
        plt.subplot(211)
        plt.plot(seeds, rew_diff, 'y')
        plt.ylabel('total waiting time')
        plt.subplot(212)
        plt.plot(seeds, exceed_time, 'b')
        plt.ylabel('time after arrival')
        plt.xlabel('episodes')
        plt.suptitle(name)
        fig = plt.gcf()
        filepath1 = os.path.join(output_dir, filename1)
        fig.set_size_inches(15, 7.5)
        if params['save_graph']:
            fig.savefig(filepath1, dpi=100)                 # .svg for thesis
        if params['show_graph']:
            plt.show()


env.close()

plt.figure()
plt.subplot(211)
plt.plot(mean_pun_list, 'r')
plt.ylabel('avg time steps waited')
plt.subplot(212)
plt.plot(fail_percent, 'k')
plt.ylabel('fail percentage')
plt.xlabel('iterations')
plt.suptitle(name)
fig = plt.gcf()
filepath2 = os.path.join(output_dir, filename2)
fig.set_size_inches(15, 7.5)
if params['save_graph']:
    fig.savefig(filepath2, dpi=100)                  # .svg for thesis
if params['show_graph']:
    plt.show()

# print(f'fail list: {fail_list}\nargs: {np.argwhere(fail_list==it_num) or np.argmax(fail_list)}')
