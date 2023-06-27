import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from gym_depot.utils import *

rng = np.random.default_rng()


class DepotEnv(gym.Env):

    def __init__(self, render_mode=None):
        self.observation_space = spaces.MultiDiscrete([[9, 10, 10, 10, 10, 10, 10, 10, 9]])
        self.action_space = spaces.Discrete(7)

        # TODO: get values from config.yaml
        self.bus_num = 5
        self.dur = 3
        self.charge = 2     # charging time
        self.fuel = 1       # fueling time
        self.cs_num = 5
        self.fs_num = 2
        self.req_order = [2, 1, 0]
        self.seed = 39

        self.av = np.zeros(7)
        self.td = np.zeros(8)
        self.ts = np.zeros(8)
        self.req = 8
        self.ent = 0
        self.cs = np.zeros(5)
        self.fs = np.zeros(2)
        self.ent_config = ent_array(self.bus_num, self.dur, self.seed)

    def _get_obs(self):
        obs = np.hstack([self.ent, self.cs, self.fs, self.req])
        obs = np.array([obs])
        return obs

    def _get_info(self):
        return {"info": '__'}

    def _get_state(self):
        if self.req == 0:
            state_info = self.ent
        elif self.req <= 5 and self.req != 0:
            state_info = self.cs[self.req-1]
        elif self.req > 5 and self.req != 8:
            state_info = self.fs[self.req-6]
        else:
            state_info = 0
        return state_info

    def _goto(self, station):
        if station <= 4:
            cs_assign = station
            self.cs[cs_assign] = self._get_state()
        else:
            fs_assign = station
            self.fs[fs_assign-5] = self._get_state()

        if self.req == 0:
            self.ent = 0
        elif self.req != 0 and self.req <= 5:
            cs_reset = self.req
            self.cs[cs_reset-1] = 0
        elif self.req > 5 and self.req != 8:
            fs_reset = self.req
            self.fs[fs_reset-6] = 0
        self.req = 8

    def _wrong_dec(self):
        self.ent = 0
        self.cs = np.zeros(5)
        self.fs = np.zeros(2)
        self.req = 8

    def _get_reward(self, r):
        if sum(self.td):
            print(self.td)
        rew = r - sum(self.td)
        self.td = np.zeros(8)
        return rew

    def _get_inst_rew(self, action):
        if (self.req == 0 or (action < 5 and self.req > 5) or
                (action >= 5 and self.req <= 5 and self.req !=0)):
            r = 1
        else:
            r = 0
        return r

    def _check_termination(self, action):
        rew = 0
        done = False
        if (self.av[action] != 0 or                                   # crash action
            (action < 5 and self.req <= 5 and self.req != 0) or       # same cs action
            (action >= 5 and self.req > 5)):                          # same fs action
            rew = -10
            done = True
            self._wrong_dec()
        elif self.req == 8 and sum(self.cs) == 45:
            rew = 10
            done = True
            self._goto(action)
        return rew, done

    def _check_entrance(self):
        if len(self.ent_config):
            if self.ent_config[0]:
                self.ent_config, self.ent = ent_update(self.ent_config, self.ent, self.td, self.seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.av = np.zeros(7)
        self.td = np.zeros(8)
        self.ts = np.zeros(8)
        self.req = 8
        self.ent = 0
        self.cs = np.zeros(5)
        self.fs = np.zeros(2)
        self.ent_config = ent_array(self.bus_num, self.dur, self.seed)
        self.ent_config, self.ent = ent_update(self.ent_config, self.ent, self.td, self.seed)
        if self.ent:
            self.req = 0
            observation = self._get_obs()
            info = self._get_info()
            print(self.ent_config)
            return observation, info

    def step(self, action):
        # check termination
        reward, terminated = self._check_termination(action)
        if not terminated:
            # calculate instantaneous reward
            r = self._get_inst_rew(action)
            # calculate states from the action
            self._goto(action)
            # update availability
            self.av = av_state(self.av, self.cs, self.fs)
            # self.av_state()
            # perform the updates(ent_config, cl, fl) and get the next request
            while self.req == 8:
                self._check_entrance()
                (self.req, self.ent, self.ent_config, self.cs, self.fs, self.ts,
                 self.td) = update(self.req_order, self.req, self.av, self.ent, self.ent_config, self.cs, self.fs,
                                   self.ts, self.td, self.charge,self.fuel, self.seed)
                if self.req == 8 and sum(self.cs) == 45:
                    reward = 10
                    terminated = True
                    break
                elif self.req == 8:
                    print('1 ts passed')
            if not terminated:
                reward = self._get_reward(r)
                print(self.ent_config)
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info

    def render(self):
        pass




