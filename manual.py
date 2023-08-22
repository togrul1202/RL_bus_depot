import gymnasium
import gym_depot
from gym_depot.utils import *


def get_act(obs, n=0, inter=0):
    action = None
    for idx, ob in enumerate(obs):
        if not ob:
            if action != None:
                if idx == action+1:
                    break
            action = idx + n
            if action < interlock and not action % 2:
                continue
            break
    return action


def req_act(req, obs):
    if req == 0:
        action = get_act(obs[1:-1])
    elif req <= cs_num:
        action = get_act(obs[cs_num + 1:-1], n=cs_num)
        if action == None or obs[req] == SF:
            action = req-2
    else:
        action = get_act(obs[1:cs_num + 1])
    return action


if __name__ == "__main__":
    cs_num = params['cs_num']
    env = gymnasium.make('BusDepot-v0')
    obs, _ = env.reset(seed=12)
    tot_rew = 0
    print(obs)
    req = obs[-1]
    number = 0
    while True:
        number += 1
        action = req_act(req, obs)
        print(f'action: {action}')
        observation, reward, terminated, truncated, info = env.step(action)
        obs = observation
        req = obs[-1]
        tot_rew += reward
        print(observation, reward)
        if terminated or truncated:
            print(f'number: {number}')
            print(info.get('info'))
            break

    print(f'total reward: {tot_rew}')

    env.close()


