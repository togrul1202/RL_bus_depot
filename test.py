import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

import gymnasium
import gym_depot

env = gymnasium.make('BusDepot-v0', render_mode='human')
obs, _ = env.reset()
i = 0
while True:
    #env.render()
    action = env.action_space.sample()
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation, reward, info)
    if terminated or truncated:
        i += 1
        if i == 2:
            break
        obs, _ = env.reset()
        print(obs)


env.close()


