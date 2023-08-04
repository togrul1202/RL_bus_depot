import numpy as np
import os
import math

import gymnasium as gym
from gymnasium import spaces

from gym_depot.utils import *

rng = np.random.default_rng()


class DepotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode=None):
        self.size = 5
        self.window_size = 512

        self.observation_space = spaces.Box(low=0, high=value, shape=(shape,), dtype=int)
        self.action_space = spaces.Discrete(total_st)

        self.info = None
        self.ter_num = 0
        self.av = np.zeros(total_st, dtype=int)            # availability
        self.av_emp = 1                         # available employees
        self.tt = np.zeros(total_st, dtype=int)            # travel time
        self.tt_emp = np.zeros(total_st, dtype=int)        # employee in bus
        self.emp_tt = np.zeros(total_st, dtype=int)        # employee travel time
        self.td = np.zeros(total, dtype=int)               # time delay
        self.ts = np.zeros(total, dtype=int)               # time step
        self.req = total                        # request
        self.ent = 0                            # entrance value
        self.cs = np.zeros(cs_num, dtype=int)              # CS values
        self.fs = np.zeros(fs_num, dtype=int)              # FS values
        self.ent_config = np.zeros(len(ent_array())).tolist()

        self.mask_fail = False
        self.emp_loc = np.zeros(total, dtype=int)           # location of available employees [0]: SB
        self.emp_timer = np.zeros(total_st, dtype=int)      # shows how long available emp stayed at last arrived station

        self.req_render = total
        self.tt_render = np.zeros(total_st, dtype=int)
        self.td_render = np.zeros(total, dtype=int)
        self.ent_render = 0
        self.cs_render = np.zeros(cs_num, dtype=int)
        self.fs_render = np.zeros(fs_num, dtype=int)
        self.td_total = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        obs = np.hstack([self.ent, self.cs, self.fs, self.req])
        obs = np.array(obs, dtype=int)
        return obs

    def _get_info(self):
        return {"info": self.info}

    def _get_state(self):
        if not self.req:
            state_info = self.ent
        elif self.req <= cs_num:
            state_info = self.cs[self.req-1]
        elif self.req > cs_num and self.req != total:
            state_info = self.fs[self.req-(cs_num+1)]
        else:
            state_info = 0
        return state_info

    def _goto(self, station):
        # assign stations
        if station < cs_num:
            cs_assign = station
            self.cs[cs_assign] = self._get_state()
            self.tt[cs_assign], self.tt_emp[cs_assign] = get_tt(self.req, cs_assign)
        else:
            fs_assign = station
            self.fs[fs_assign-cs_num] = self._get_state()
            self.tt[fs_assign], self.tt_emp[fs_assign] = get_tt(self.req, fs_assign)

        # reset stations
        if not self.req:
            self.ent = 0
        elif self.req <= cs_num:
            cs_reset = self.req-1               # cs val to zero
            if emp_num and emp_time:
                time, self.emp_loc = find_closest_emp(cs_reset, self.emp_loc)
                self.emp_tt[cs_reset] = time
                self.emp_tt[station] = -cs_reset-1
            else:
                self.cs[cs_reset] = 0
        elif self.req > cs_num and self.req != total:
            fs_reset = self.req-1               # fs val to zero
            if emp_num and emp_time:
                time, self.emp_loc = find_closest_emp(fs_reset, self.emp_loc)
                self.emp_tt[fs_reset] = time
                self.emp_tt[station] = -fs_reset-1
            else:
                self.fs[fs_reset-cs_num] = 0
        self.req = total

    def _wrong_dec(self):
        self.ent = 0
        self.cs = np.zeros(cs_num, dtype=int)
        self.fs = np.zeros(fs_num, dtype=int)
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
        return rew

    def _get_inst_rew(self, action):
        num = cs_num
        if action < num < self.req:
            r = inst
        elif action >= num >= self.req != 0 and self.cs[self.req-1] != 9:
            r = inst
        else:
            r = 0
        return r

    def _check_termination(self, action):
        num = cs_num
        rew = 0
        done = False
        repeat = False
        state = self._get_state()
        lock, lock_crash, stuck = check_interlock(action, self.req, self.av, self.cs, state)
        crash = self.cs[action] != 0 if action < cs_num else self.fs[action-cs_num] != 0
        same_cs = action < num and self.req <= num and self.req != 0 and not lock and self.cs[self.req - 1] != 9    # even if there is av fs value 9 doesnt lead same_cs
        same_fs = action >= num and self.req > num
        wrong_fs = action >= num >= self.req != 0 and self.cs[self.req - 1] == 9    # wrong fs iff cs value is 9
        dic = {'no_valid_action': self.mask_fail, 'same_cs': same_cs, 'same_fs': same_fs, 'crash': crash,
               'lock_crash': lock_crash, 'stuck': stuck, 'wrong_fs': wrong_fs}
        for key, val in dic.items():
            if val:
                self.info = key
                break
        if self.mask_fail or crash or same_cs or same_fs or lock_crash or stuck or wrong_fs:
            rew = fail - self._get_delay()
            if self.ter_num < rep_num:
                self.ter_num += 1
                repeat = True
            else:
                done = True
                self._wrong_dec()
        return rew, done, repeat

    def _check_entrance(self):
        if len(self.ent_config):
            if self.ent_config[0]:
                self.ent_config, self.ent = ent_update(self.ent_config, self.ent, self.td)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.info = None
        self.ter_num = 0
        self.av = np.zeros(total_st, dtype=int)
        if emp_num:
            self.av_emp = emp_num
        self.tt = np.zeros(total_st, dtype=int)
        self.tt_emp = np.zeros(total_st, dtype=int)
        self.emp_tt = np.zeros(total_st, dtype=int)
        self.td = np.zeros(total, dtype=int)
        self.td_total = 0
        self.ts = np.zeros(total, dtype=int)
        self.req = total
        self.ent = 0
        self.cs = np.zeros(cs_num, dtype=int)
        self.fs = np.zeros(fs_num, dtype=int)
        self.ent_config = ent_array(seed)
        self.ent_config, self.ent = ent_update(self.ent_config, self.ent, self.td)

        self.mask_fail = False
        self.emp_loc = np.zeros(total, dtype=int)
        self.emp_timer = np.zeros(total_st, dtype=int)
        if emp_num:
            self.emp_loc[0] = emp_num          # employees at the SB

        if self.ent:
            self.req = 0
            observation = self._get_obs()
            info = self._get_info()
            self.metadata["render_fps"] = params['render_reset']
            if self.render_mode == "human":
                self._render_frame()
            #print(self.ent_config)
            return observation, info

    def step(self, action):
        step = 0
        truncated = False
        # check termination
        reward, terminated, repeat = self._check_termination(action)
        if terminated or repeat:
            self.metadata["render_fps"] = params['render_slow']
            if self.render_mode == "human":
                self._render_frame(action, failed=True)
        else:
            self.info = None
            self.metadata["render_fps"] = params['render_fast']
            if self.render_mode == "human":
                self._render_frame(action)
            # calculate instant reward
            r = self._get_inst_rew(action)
            # calculate states from the action
            self._goto(action)
            # update availability
            self.av = av_state(self.av, self.cs, self.fs)
            # perform the updates(ent_config, cl, fl) and get the next request
            while self.req == total:
                if step >= 10000:
                    truncated = True
                    reward = fail
                    print('truncated')
                    print(f'tt: {self.tt}')
                    break
                else:
                    truncated = False
                    step += 1
                self._check_entrance()
                (self.req, self.av, self.ent, self.ent_config, self.cs, self.fs, self.ts,
                 self.td, self.tt, self.tt_emp, self.av_emp, self.emp_tt, self.emp_loc, self.emp_timer) = update(self.req,
                    self.av, self.ent, self.ent_config, self.cs, self.fs, self.ts, self.td, self.tt, self.tt_emp,
                    self.av_emp, self.emp_tt, self.emp_loc, self.emp_timer)
                if self.render_mode == "human":
                    self._render_frame()
                print(f'emp_loc: {self.emp_loc}')
                if self.req == total and sum(self.cs) == 9*bus_num and not self.emp_tt.any() and not sum(self.fs):
                    reward = win - self._get_delay()
                    terminated = True
                    self.metadata["render_fps"] = params['render_slow']
                    if self.render_mode == "human":
                        self._render_frame(success=True)
                    #print('success')
                    break
                # elif self.req == total:
                #     print('1 ts passed')
                #     print(f'cs: {self.cs}\nfs: {self.fs}\ntd: {self.td}')
            if not terminated:
                reward = self._get_reward(r)
                self.ter_num -= 1 if self.ter_num and rep_per_action else 0
                #print(self.ent_config)
        observation = self._get_obs()
        info = self._get_info()
        # if self.render_mode == "human":
        #     self._render_frame()
        self.td = np.zeros(total, dtype=int)
        # print(f'employee: {self.av_emp}\ntt: {self.tt}\nemp_tt: {self.emp_tt}')

        return observation, reward, terminated, truncated, info

    def get_action_mask(self):
        array = np.array([[self.check_action_validity(a) for a in range(total_st)]])
        if not array.any():
            self.mask_fail = True
            array = np.array([[True for a in range(total_st)]])
        return array

    def check_action_validity(self, action):
        obs = self._get_obs()
        state = self._get_state()
        lock, lock_crash, stuck = check_interlock(action, self.req, self.av, self.cs, state)
        wrong_fs = action >= cs_num >= self.req != 0 and self.cs[self.req - 1] == 9
        if obs[action + 1]:
            return False                                    # avoid crash
        if lock_crash:
            return False
        if stuck:
            return False
        if not self.req:
            return True
        elif self.req <= cs_num and action < cs_num and not lock and self.cs[self.req - 1] != 9:
            return False                                    # avoid same_cs
        elif self.req > cs_num and action >= cs_num:
            return False                                    # avoid same_fs
        elif wrong_fs:
            return False
        else:
            return True

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self, action=None, failed=False, success=False, check=False):

        size = (800, 800)
        # colors
        white = (255, 255, 255)
        red = (255, 0, 0)
        pink = (255, 100, 255)
        green = (0, 255, 0)
        blue = (5, 213, 250)
        yellow = (255, 255, 0)
        grey = (200, 200, 200)
        brown = (101, 67, 33)
        black = (0, 0, 0)

        if not self.info:
            self.req_render = self.req
            self.tt_render = self.tt
            self.td_render = self.td
            self.ent_render = self.ent
            self.cs_render = self.cs
            self.fs_render = self.fs
            if self.req != total:
                self.td_total += sum(self.td)

        # color info
        colors = ['green', 'grey', 'pink', 'blue', 'yellow', 'red', 'brown']
        color_info = ['empty', 'bus not arrived', 'employee not arrived', 'charging/fueling', 'request to go',
                      'request delayed', 'charging and fueling finished']

        # value info
        rows = ['info', 'SoC', 'FL']

        # actions
        if action is not None:
            if action < cs_num:
                action = 'CS ' + str(action+1)
            else:
                action = 'FS ' + str(action-cs_num+1)

        # stations
        station = ''
        ent_color = green
        ent_loc = (550, 720, 25, 50)
        ent_text = ''
        ent_text_loc = (560, 730)
        ent_name = 'Entrance'
        ent_name_loc = (550, 777)

        cs_color = [green] * cs_num

        fs_color = [green] * fs_num
        fs_loc = [(220, 740, 50, 25), (25, 425, 25, 50)]
        fs_text_loc = [(220, 750), (35, 435)]
        fs_name_loc = [(220, 767), (55, 467)]

        if self.window is None and (self.render_mode == "human" or check):
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(size)
        if self.clock is None and (self.render_mode == "human" or check):
            self.clock = pygame.time.Clock()

        screen = pygame.Surface(size)
        screen.fill(white)

        font_size = 20
        font_color = black
        font = pygame.font.Font(None, font_size)
        info_x = 550

        # time
        time = 'time passed: ' + str(max(self.ts)) + ' minutes'
        time_loc = (550, 320)
        time_surface = font.render(time, True, brown)
        screen.blit(time_surface, time_loc)

        # employee
        if emp_num:
            emp = 'number of available employees: ' + str(self.av_emp)
            emp_loc = (550, 340)
            emp_surface = font.render(emp, True, brown)
            screen.blit(emp_surface, emp_loc)

        # color info
        for idx, color in enumerate(colors):
            color_text = color + ': ' + color_info[idx]
            color_text_loc = (info_x, 17*idx)
            color_text_surface = font.render(color_text, True, font_color)
            screen.blit(color_text_surface, color_text_loc)
        last_loc = 19*(idx+1)

        # value info
        for i, txt in enumerate(rows):
            txt_loc = (info_x+i*34, last_loc)
            txt_surface = font.render(txt, True, font_color)
            screen.blit(txt_surface, txt_loc)
            for j in range(9):
                text_loc = (info_x + i * 34, last_loc + 17 * (j+1))
                if txt == 'info':
                    j += 1
                elif txt == 'SoC':
                    j = math.floor(j/3)
                elif txt == 'FL':
                    j %= 3
                text_surface = font.render(str(j), True, font_color)
                screen.blit(text_surface, text_loc)

        font_size = 16
        font_color = black
        font = pygame.font.Font(None, font_size)

        # Ent
        if self.td_render[0]:
            ent_color = red
            ent_text = 'info: ' + str(self.ent_render) + ', delay: ' + str(self.td_render[0])
        elif self.ent_render:
            ent_color = yellow
            ent_text = 'info: ' + str(self.ent_render)
        pygame.draw.rect(
            screen,
            ent_color,
            ent_loc,
        )
        ent_text_surface = font.render(ent_text, True, font_color)
        ent_text_surface = pygame.transform.rotate(ent_text_surface, 90)
        screen.blit(ent_text_surface, ent_text_loc)

        if action and not self.req_render:
            station = ent_name
            ent_name += '  go to ' + str(action)
            if failed:
                ent_name += ' -> ' + self.info
        ent_name_surface = font.render(ent_name, True, font_color)
        screen.blit(ent_name_surface, ent_name_loc)

        # CS
        ki = 0
        for idx, val in enumerate(self.cs_render):
            cs_name = 'CS ' + str(idx + 1)
            if val:
                cs_text = 'info: ' + str(val)
            else:
                cs_text = ''
            if action and self.req_render == idx+1:
                station = cs_name
                cs_name += '  go to ' + str(action)
                if failed:
                    cs_name += ' -> ' + self.info
            if self.td_render[idx + 1]:
                if self.emp_tt[idx]:
                    cs_color[idx] = grey if self.tt_render[idx] else pink
                elif not self.cs[idx]:
                    cs_color[idx] = green
                else:
                    cs_color[idx] = grey if self.tt_render[idx] else red
                cs_text = cs_text + ', delay: ' + str(self.td_render[idx+1]) if not success else ''
            elif self.tt_render[idx]:
                cs_color[idx] = grey
            elif val == 9:
                if self.req_render == idx+1:
                    cs_color[idx] = yellow
                elif self.emp_tt[idx]:
                    cs_color[idx] = pink
                else:
                    cs_color[idx] = brown
            elif val >= 7:
                cs_color[idx] = yellow
            elif val:
                cs_color[idx] = blue
            if not interlock:
                loc = ((175, 670 - idx * 40, 50, 25), (175, 680 - idx * 40))
            else:
                if idx < interlock:
                    ki = math.floor(idx / 2)
                j = idx % 2 if idx+1 < interlock else 1
                k = ki if idx < interlock else idx-interlock+ki+1
                loc = ((175+j*90, 670-k*40, 50, 25), (175+j*90, 680-k*40))
            cs_loc, cs_text_loc = loc
            pygame.draw.rect(
                screen,
                cs_color[idx],
                cs_loc,
            )
            cs_text_surface = font.render(cs_text, True, font_color)
            screen.blit(cs_text_surface, cs_text_loc)
            cs_name_loc = (cs_loc[0], cs_loc[1]+25)
            cs_name_surface = font.render(cs_name, True, font_color)
            screen.blit(cs_name_surface, cs_name_loc)

        # FS
        for idx, val in enumerate(self.fs_render):
            if val:
                fs_text = 'info: ' + str(val)
            else:
                fs_text = ''
            if self.td_render[idx+1+cs_num]:
                if self.emp_tt[idx+cs_num]:
                    fs_color[idx] = pink
                elif not self.fs[idx]:
                    fs_color[idx] = green
                else:
                    fs_color[idx] = red
                fs_text = fs_text + ', delay: ' + str(self.td_render[idx + cs_num + 1]) if not success else ''
            elif self.tt_render[idx+cs_num]:
                fs_color[idx] = grey
            elif val % 3:
                fs_color[idx] = blue
            elif val:
                fs_color[idx] = yellow
            pygame.draw.rect(
                screen,
                fs_color[idx],
                fs_loc[idx],
            )
            fs_text_surface = font.render(fs_text, True, font_color)
            if idx:
                fs_text_surface = pygame.transform.rotate(fs_text_surface, 90)
            screen.blit(fs_text_surface, fs_text_loc[idx])
            fs_name = 'FS ' + str(idx+1)
            if action and self.req_render == idx+cs_num+1:
                station = fs_name
                fs_name += '  go to ' + str(action)
                if failed:
                    fs_name += ' -> ' + self.info
            fs_name_surface = font.render(fs_name, True, font_color)
            screen.blit(fs_name_surface, fs_name_loc[idx])

        # fail and success
        if self.info or success:
            if self.info:
                text = 'FAILED: ' + self.info
                info = 'from ' + station + ' to ' + str(action)
                color = red
            if success:
                text = 'SUCCESS'
                self.td_total += sum(self.td)
                info = 'total waiting time: ' + str(self.td_total) + ' minutes'
                color = green
            text_loc = (400, 400)
            info_loc = (400, 450)
            font = pygame.font.Font(None, 50)
            fail_surface = font.render(text, True, color)
            screen.blit(fail_surface, text_loc)
            font = pygame.font.Font(None, 30)
            fail_surface = font.render(info, True, color)
            screen.blit(fail_surface, info_loc)

        # The following line copies our drawings from `screen` to the visible window
        self.window.blit(screen, screen.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

