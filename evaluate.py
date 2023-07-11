import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

import gymnasium
import gym_depot
from stable_baselines3 import PPO, A2C

env = gymnasium.make('BusDepot-v0')

name = 'training2/PPO-1688411857/200000'
model = PPO.load('models/'+name+'.zip', env=env)

episodes = 100
it_num = 10
mean_pun_list = []
fail_percent = []
for _ in range(it_num):
    tot_rew = 0
    rew_diff = []
    act_number = []
    seed = 0
    fail = 0
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed)
        number = 0
        while True:
            number += 1
            action, _ = model.predict(obs)
            #print(f'action:{action}')
            obs, reward, terminated, truncated, info = env.step(action)
            #print(obs, reward)
            tot_rew += reward
            if terminated or truncated:
                #print(tot_rew)
                rew_diff.append(reward-tot_rew)
                if reward < 0:
                    fail += 1
                tot_rew = 0
                act_number.append(number)
                #print(f'number: {number}')
                #print(f'seed: {seed}')
                seed += 1
                break
    mean_pun = mean(rew_diff)
    mean_pun_list.append(mean_pun)
    fail_percent.append(fail*100/episodes)
    print(f'avg time steps waited: {mean_pun}\nnumber of fails: {fail}/{episodes}')

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
plt.show()
