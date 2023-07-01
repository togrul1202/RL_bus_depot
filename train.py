import os

import gymnasium
import gym_depot
from stable_baselines3 import PPO

# model_dir = 'models/PPO'
#
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)

env = gymnasium.make('BusDepot-v0')
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save('best')
del model
model = PPO.load('best', env=env)
a = env.reset()
obs = a[0]


i = 0
tot_rew = 0
ep_num = 10
while True:
    action, _ = model.predict(obs)
    print(action)
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs, reward)
    tot_rew += reward
    if terminated or truncated:
        print(tot_rew)
        tot_rew = 0
        i += 1
        if i == ep_num:
            break
        a = env.reset()
        obs = a[0]
        print(obs)


env.close()

