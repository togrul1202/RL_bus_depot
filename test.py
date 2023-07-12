import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

import gymnasium
import gym_depot
from stable_baselines3 import PPO, A2C

from manual import req_act
from manual_inv import req_act_inv

env = gymnasium.make('BusDepot-v0')

result_dir = 'results/'
name = 'models/training6/PPO-1689090430/13200000'
name_dir = Path(name)
filename1 = 'steps-and-actions.png'                     # .svg for thesis
filename2 = 'avg_steps-and-fail.png'

if name != 'manual' and name != 'manual_inv':
    model = PPO.load(name, env=env)
    output_dir = os.path.join(result_dir, name_dir.parent.parent.name, name_dir.parent.name, name_dir.stem)
else:
    output_dir = os.path.join(result_dir, name)
os.makedirs(output_dir, exist_ok=True)

episodes = 100
fail_list = np.zeros(100)
episodes_up = episodes
it_num = 5
mean_pun_list = []
fail_percent = []
for it in range(it_num):
    tot_rew = 0
    rew_diff = []
    act_number = []
    rew = []
    seed = 1
    fail = 0
    dic = {'same_cs': 0, 'same_fs': 0, 'crash': 0, 'lock_crash': 0, 'stuck': 0, 'wrong_fs': 0}
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
                rew_diff.append(reward-tot_rew)
                if truncated:
                    episodes_up -= 1
                if reward < 0:
                    fail += 1
                    dic[info.get('info')] += 1
                    fail_list[episode] += 1
                tot_rew = 0
                act_number.append(number)
                #print(f'number: {number}')
                #print(f'seed: {seed}')
                seed += 1
                break
    mean_rew = mean(rew)
    mean_pun = mean(rew_diff)
    mean_number = mean(act_number)
    mean_pun_list.append(mean_pun)
    fail_percent.append(fail*100/episodes)
    print(f'mean rew: {mean_rew}\navg time steps waited: {mean_pun}\nnumber of fails: {fail}/{episodes_up}\nfails: {dic}',
          f'mean number: {mean_number}')

    if it == 0:
        plt.figure()
        seeds = list(range(1, seed))
        plt.subplot(211)
        plt.plot(seeds, rew_diff, 'y')
        plt.ylabel('time steps waited')
        plt.subplot(212)
        plt.plot(seeds, act_number, 'b')
        plt.ylabel('number of actions')
        plt.xlabel('seeds')
        plt.suptitle(name)
        fig = plt.gcf()
        filepath1 = os.path.join(output_dir, filename1)
        fig.set_size_inches(15, 7.5)
        fig.savefig(filepath1, dpi=100)                 # .svg for thesis
        # plt.show()


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
fig.savefig(filepath2, dpi=100)                  # .svg for thesis
# plt.show()

print(f'fail list: {fail_list}\nargs: {np.argwhere(fail_list==5)}')
