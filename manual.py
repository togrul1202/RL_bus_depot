import gymnasium
import gym_depot


def get_act(obs, n=0):
    action = 0
    for idx, obs in enumerate(obs):
        if not obs:
            action = idx + n
            break
    return action


if __name__ == "__main__":
    env = gymnasium.make('BusDepot-v0')
    a = env.reset()
    obs = a[0]
    tot_rew = 0
    print(a)
    req = obs[0][-1]

    while True:
        if req == 0:
            action = get_act(obs[0][1:-1])
        elif req <= 5:
            action = get_act(obs[0][6:-1], n=5)
        else:
            action = get_act(obs[0][1:6])
        print(action)
        observation, reward, terminated, truncated, info = env.step(action)
        obs = observation
        req = obs[0][-1]
        tot_rew += reward
        print(observation, reward)
        if terminated or truncated:
            break

    print(tot_rew)

    env.close()


