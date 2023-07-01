import gymnasium
import gym_depot
from gym_depot.utils import *


def get_act(obs, n=0):
    action = 0
    for idx, obs in enumerate(obs):
        if not obs:
            action = idx + n
            break
    return action


if __name__ == "__main__":
    cs_num = params['cs_num']
    env = gymnasium.make('BusDepot-v0')
    a = env.reset()
    obs = a[0]
    tot_rew = 0
    print(obs)
    req = obs[-1]

    while True:
        if req == 0:
            action = get_act(obs[1:-1])
        elif req <= cs_num:
            action = get_act(obs[cs_num+1:-1], n=cs_num)
        else:
            action = get_act(obs[1:cs_num+1])
        print(action)
        observation, reward, terminated, truncated, info = env.step(action)
        obs = observation
        req = obs[-1]
        tot_rew += reward
        print(observation, reward)
        if terminated or truncated:
            break

    print(tot_rew)

    env.close()


