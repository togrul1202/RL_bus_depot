import gymnasium
import numpy as np
import torch
import gym_depot


env = gymnasium.make('BusDepot-v0')
a = env.reset()
# obs = a[0]
# print(obs[0][6:-1])
i = 0
while True:
    action = env.action_space.sample()
    print(action)
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation, reward)
    if terminated or truncated:
        i += 1
        if i == 3:
            break
        (b) = env.reset()
        print(b)



env.close()