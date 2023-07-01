import numpy as np
import pygame
import os

import gymnasium as gym
from gymnasium import spaces

from gym_depot.utils import *

rng = np.random.default_rng()


class DepotEnv(gym.Env):

    def __init__(self, render_mode=None):

        self.observation_space = spaces.Box(low=0, high=value, shape=(shape,), dtype=int)
        self.action_space = spaces.Discrete(total_st)

        self.av = np.zeros(total_st)
        self.td = np.zeros(total)
        self.ts = np.zeros(total)
        self.req = total
        self.ent = 0
        self.cs = np.zeros(cs_num)
        self.fs = np.zeros(fs_num)
        self.ent_config = np.zeros(len(ent_array())).tolist()

    def _get_obs(self):
        obs = np.hstack([self.ent, self.cs, self.fs, self.req])
        obs = np.array(obs, dtype=int)
        return obs

    def _get_info(self):
        return {"info": '__'}

    def _get_state(self):
        if self.req == 0:
            state_info = self.ent
        elif self.req <= cs_num and self.req != 0:
            state_info = self.cs[self.req-1]
        elif self.req > cs_num and self.req != total:
            state_info = self.fs[self.req-(cs_num+1)]
        else:
            state_info = 0
        return state_info

    def _goto(self, station):
        if station < cs_num:
            cs_assign = station
            self.cs[cs_assign] = self._get_state()
        else:
            fs_assign = station
            self.fs[fs_assign-cs_num] = self._get_state()

        if self.req == 0:
            self.ent = 0
        elif self.req != 0 and self.req <= cs_num:
            cs_reset = self.req
            self.cs[cs_reset-1] = 0
        elif self.req > cs_num and self.req != total:
            fs_reset = self.req
            self.fs[fs_reset-(cs_num+1)] = 0
        self.req = total

    def _wrong_dec(self):
        self.ent = 0
        self.cs = np.zeros(cs_num)
        self.fs = np.zeros(fs_num)
        self.req = total

    def _get_delay(self):
        delay = 0
        for idx, val in enumerate(self.td):
            delay += int(val / time_delay)
        return delay

    def _get_reward(self, r):
        #if sum(self.td):
            #print(self.td)
        delay = self._get_delay()
        rew = r - delay
        self.td = np.zeros(total)
        return rew

    def _get_inst_rew(self, action):
        num = cs_num
        if (self.req == 0 or (action < num and self.req > num) or
                (action >= num and self.req <= num and self.req != 0)):
            r = inst
        else:
            r = 0
        return r

    def _check_termination(self, action):
        num = cs_num
        rew = 0
        done = False
        if (self.av[action] != 0 or                                   # crash action
            (action < num and self.req <= num and self.req != 0) or       # same cs action
            (action >= num and self.req > num)):                          # same fs action
            rew = fail
            done = True
            self._wrong_dec()
        elif self.req == total and sum(self.cs) == 9*cs_num:
            rew = win
            done = True
            self._goto(action)
        return rew, done

    def _check_entrance(self):
        if len(self.ent_config):
            if self.ent_config[0]:
                self.ent_config, self.ent = ent_update(self.ent_config, self.ent, self.td)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.av = np.zeros(total_st)
        self.td = np.zeros(total)
        self.ts = np.zeros(total)
        self.req = total
        self.ent = 0
        self.cs = np.zeros(cs_num)
        self.fs = np.zeros(fs_num)
        self.ent_config = ent_array()
        self.ent_config, self.ent = ent_update(self.ent_config, self.ent, self.td)
        if self.ent:
            self.req = 0
            observation = self._get_obs()
            info = self._get_info()
            #print(self.ent_config)
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
            while self.req == total:
                self._check_entrance()
                (self.req, self.ent, self.ent_config, self.cs, self.fs, self.ts,
                 self.td) = update(self.req, self.av, self.ent, self.ent_config, self.cs, self.fs, self.ts, self.td)
                if self.req == total and sum(self.cs) == 9*cs_num:
                    reward = win
                    terminated = True
                    print('success')
                    break
                #elif self.req == self.total:
                    #print('1 ts passed')
            if not terminated:
                reward = self._get_reward(r)
                print(self.ent_config)
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info

    def render(self):
        pass




