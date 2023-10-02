import numpy as np
import yaml
import pygame
import math


with open('gym_depot/config.yaml') as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

min_to_ts = params['min_to_ts']
level_num = params['level_num']
SF = level_num**2                   # SoC and FL in max level
bus_num = params['bus_num']
bus_time = int(params['bus_time'] * min_to_ts)
min_dur = params['min_duration'] * min_to_ts
max_dur = params['max_duration'] * min_to_ts
charge = params['charge_time'] * min_to_ts if not params['charge_total'] else params['charge_total']/(level_num-1)*min_to_ts
fast_charge = params['fast_charge_time'] * min_to_ts if not params['fast_charge_total'] else \
    params['fast_charge_total']/(level_num-1)*min_to_ts
fuel = params['fuel_time'] * min_to_ts if not params['fuel_total'] else params['fuel_total']/(level_num-1)*min_to_ts
cs_num = params['cs_num']
fast_cs_num = params['fast_cs_num']
fs_num = params['fs_num']
req_order = params['req_order']
inst = params['inst']
fail = params['fail']
win = params['win']
random = params['random']
seed = params['seed']
total_st = cs_num + fs_num
total = total_st + 1
shape = total + 1
if shape <= SF:
    value = SF
else:
    value = shape
time_delay = params['time_delay']
travel_time = [[num * min_to_ts * 2 for num in sublist] for sublist in params['travel_time']]
interlock = params['interlock']
if interlock > cs_num:
    raise Exception('No more pair stations than cs')
elif interlock % 2:
    raise Exception('interlock must be even number')
if bus_num > cs_num:
    raise Exception('no more buses than CS')
if fast_cs_num > cs_num:
    raise Exception('no more FCS than CS')
emp_num = params['emp_num']
emp_time = [[num * min_to_ts for num in sublist] for sublist in params['emp_time']]
rep_num = params['repeat']
rep_per_action = params['repeat_per_action']


def ent_array(ev_seed=None):
    if not random:
        np.random.seed(seed)
    elif ev_seed:
        np.random.seed(ev_seed)
    duration = np.random.random_integers(min_dur, max_dur)
    zeros = duration  # zero = time step
    array = np.array([1] * (bus_num-2) + [0] * zeros)

    # Shuffle the array
    np.random.shuffle(array)
    array = np.concatenate(([1], array, [1]))
    array = array.tolist()
    if bus_time:        # change arriving time of buses at the ent based on bus_time
        array = add_waiting(array, bus_time)
    #print(f'entrance array is: {array}')
    return array


def add_waiting(arr, bt):
    idx = 0
    while True:
        if idx != len(arr) - 1:
            bol = arr[idx + 1]
            for k in range(bt):
                bol = bol or arr[idx + 1 + k]
            if arr[idx] and bol:
                b = arr[idx:]
                for _ in range(bt):
                    if 0 in b:
                        b.remove(0)
                arr = arr[:idx] + b
                for _ in range(bt):
                    arr.insert(idx + 1, 0)
        else:
            break
        idx += 1
    return arr


def ent_update(self):
    if len(self.ent_config):
        if not self.ent:
            if self.ent_config[0]:
                self.ent = np.random.choice([a for a in range(SF-level_num) if a % level_num])  # self.ent_config[0]
                #print(f'self.ent: {self.ent}')
                soc, fl = soc_and_fl(self.ent)
                charge_time = (fast_charge*fast_cs_num+charge*(cs_num-fast_cs_num))/cs_num
                time = int((100-soc)*charge_time + (100-fl)*fuel)+self.ts[0]
                self.ex_time.append(time)
                #print(max(self.ex_time))
            del self.ent_config[0]
        else:
            if 0 in self.ent_config and self.td[0]:
                self.ent_config.remove(0)
    return self.ent_config, self.ent


def cs_update(self):
    for idx, cs in enumerate(self.cs):
        emp_in = False
        if self.emp_tt[idx] < 0:                                         # waiting for employee
            continue
        elif self.tt[idx] or self.tt_emp[idx]:
            if self.tt_emp[idx] == self.tt[idx] != 0:                        # employee in a bus
                emp_in = True
                self.tt_emp[idx] -= 1
                if not self.tt_emp[idx] and self.cs[idx]:
                    if idx < interlock and not idx % 2:
                        if not (self.tt[idx+1] and not self.cs[idx+1]):           # on travel request: no extra emp
                            self.av_emp += 1
                            self.emp_loc[idx + 1] += 1
                    else:
                        self.av_emp += 1
                        self.emp_loc[idx+1] += 1
            if idx < interlock and not idx % 2:
                if not (self.tt[idx+1] and not self.cs[idx+1]):
                    if not self.tt[idx] or self.tt_emp[idx] < 0:
                        print(self.tt_emp[idx])
                    self.tt[idx] -= 1
                else:                                            # on travel request: no extra waiting for next station
                    self.tt[idx] = self.tt[idx+1]
                    self.tt[idx+1] = 0
                    if emp_num:
                        self.tt_emp[idx + 1] = self.tt[idx + 1]
                        if emp_in:
                            self.tt_emp[idx] = self.tt[idx]
                            self.tt_emp[idx] -= 1
                        if not self.tt_emp[idx] and self.cs[idx]:
                            self.av_emp += 1
                            self.emp_loc[idx+1] += 1
                    if not self.tt[idx] or self.tt_emp[idx] < 0:
                        print(self.tt_emp[idx])
                    self.tt[idx] -= 1
            else:
                if not self.tt[idx] or self.tt_emp[idx] < 0:
                    print(self.tt_emp[idx])
                self.tt[idx] -= 1
            self.ts[idx + 1] = 0
        else:
            if cs != 0 and cs <= SF-level_num:
                if idx < fast_cs_num:
                    if self.ts[idx + 1] == fast_charge:
                        self.cs[idx] += level_num
                        self.ts[idx + 1] = 0
                else:
                    if self.ts[idx+1] == charge:
                        self.cs[idx] += level_num
                        self.ts[idx+1] = 0
            else:
                self.ts[idx+1] = 0
    return self.cs, self.ts, self.tt, self.tt_emp, self.av_emp, self.emp_tt, self.emp_loc


def fs_update(self):
    for idx, fs in enumerate(self.fs):
        if self.emp_tt[idx+cs_num] < 0:
            continue
        elif self.tt[idx+cs_num] or self.tt_emp[idx+cs_num]:
            if self.tt_emp[idx+cs_num] == self.tt[idx+cs_num] != 0:                        # employee in a bus
                self.tt_emp[idx+cs_num] -= 1
                if not self.tt_emp[idx+cs_num]:
                    self.av_emp += 1
                    self.emp_loc[idx+1+cs_num] += 1
            self.tt[idx+cs_num] -= 1
            self.ts[idx+1+cs_num] = 0
        else:
            if fs % level_num != 0 and fs != 0:
                if self.ts[idx+1+cs_num] == fuel:
                    self.fs[idx] += 1
                    self.ts[idx+1+cs_num] = 0
            else:
                self.ts[idx+1+cs_num] = 0
    return self.fs, self.ts, self.tt, self.tt_emp, self.av_emp, self.emp_tt, self.emp_loc


def delay_update(self):
    if self.ent:
        mask_arr = np.array([[check_action_validity(self, a, check=True) for a in range(total_st)]]) \
            if params['check_mask'] else np.array([True])
        if 0 not in self.av or not mask_arr.any():
            self.td[0] += 1
    for idx, cs in enumerate(self.cs):
        if cs > SF-level_num and cs != SF:
            if not self.emp_tt[idx] and (0 not in self.av[cs_num:] or not self.av_emp or (idx % 2 and idx < interlock and self.av[idx - 1])):
                self.td[idx + 1] += 1
    for idx, fs in enumerate(self.fs):
        if fs % level_num == 0 and fs != 0 and (0 not in self.av[:cs_num] or not self.av_emp):
            self.td[idx+1+cs_num] += 1


def emp_tt_update(self):
    for idx, t in enumerate(self.emp_tt):                           # on travel req: don't wait for an employee
        if idx < interlock and not idx % 2 and self.tt[idx+1] and self.emp_tt[idx] == -(idx+2):
            self.emp_tt[idx] = 0
            self.emp_tt[idx+1] = 0
            self.cs[idx+1] = 0
        elif t > 0:
            self.emp_tt[idx] -= 1
            if not self.emp_tt[idx]:
                self.tt[self.emp_tt == -(idx + 1)] += 1
                self.tt_emp[self.emp_tt == -(idx + 1)] = self.tt[self.emp_tt == -(idx + 1)]
                self.emp_tt[self.emp_tt == -(idx + 1)] = 0
                if idx < cs_num:
                    self.cs[idx] = 0
                else:
                    self.fs[idx-cs_num] = 0
    #print(f'self.emp_tt: {self.emp_tt}\nself.cs: {self.cs}\nself.fs: {self.fs}')


def find_closest_emp(destination, station, emp_tt, emp_loc):                     # finds emp and send him to the destination
    time_list = []
    idx_list = []
    for idx, emp in enumerate(emp_loc):
        if emp:
            if not idx:                                                     # SB to a station
                time_list.append(emp_time[0][destination])
                idx_list.append(idx)
            elif idx <= cs_num:
                if destination < cs_num:                                       # CS TO CS
                    time_list.append(emp_time[3][1])
                else:
                    time_list.append(emp_time[destination-cs_num][idx-1])      # CS TO FS
                idx_list.append(idx)
            elif idx > cs_num:
                if destination >= cs_num:
                    time_list.append(emp_time[3][0])                            # FS TO FS
                else:
                    time_list.append(emp_time[idx-cs_num-1][destination])       # FS TO CS
                idx_list.append(idx)
    time = min(time_list)
    idx = idx_list[time_list.index(time)]
    emp_loc[idx] -= 1
    emp_tt[destination] = time
    emp_tt[station] = -destination-1
    return emp_tt, emp_loc


def emp_to_sb(self):
    self.emp_timer = np.multiply(self.emp_timer, self.emp_loc[1:])                # reset timer of unavailable employees
    for loc, emp in enumerate(self.emp_loc):
        if loc and emp:
            if self.emp_timer[loc-1] >= emp_time[0][loc-1]/2:           # consider emp at SB if half of the emp_time the last loc to SB passed
                self.emp_loc[loc] -= 1
                self.emp_loc[0] += 1
            else:
                self.emp_timer[loc-1] += 1


def update(self):
    request(self)  # get request
    if self.req == total:
        self.ts += 1
        #print(self.ts)
        delay_update(self)  # update td
        emp_to_sb(self)        # employee goes back to SB
        if emp_num and emp_time:
            emp_tt_update(self)  # update emp_tt and reset cs and fs if zero
            av_state(self)
        ent_update(self)  # update ent_config
        cs_update(self)  # update cl
        fs_update(self)  # update fl
        #print(f'emp_av: {self.av_emp}\nself.emp_tt: {self.emp_tt}\nself.tt: {self.tt}\nself.tt_emp: {self.tt_emp}\nself.td: {td}')


def av_state(self):
    for idx, cs in enumerate(self.cs):
        if cs:
            self.av[idx] = 1
            if idx % 2 and idx < interlock:
                self.av[idx - 1] = 1
        else:
            self.av[idx] = 0
    for idx, fs in enumerate(self.fs):
        self.av[idx+cs_num] = 1 if fs else 0


def request(self):
    for i in req_order:
        if i == 0 and self.req == total and self.ent and 0 in self.av:
            mask_arr = np.array([[check_action_validity(self, a, check=True) for a in range(total_st)]]) \
                if params['check_mask'] else np.array([True])
            if mask_arr.any():
                self.req = 0

        if i == 1 and self.req == total and self.av_emp:
            for idx, cs in enumerate(self.cs):
                if not self.emp_tt[idx]:                        # avoid requesting again while waiting for an employee
                    if idx % 2 and idx < interlock and cs > SF-level_num:
                        if not self.cs[idx-1] and not self.tt[idx-1]:
                            self.req = idx + 1
                            if not self.tt[idx]:                 # on travel request: don't ask for another employee
                                self.av_emp = self.av_emp - 1 if emp_num else 1
                            break
                    elif (cs > SF-level_num and cs != SF) and 0 in self.av[cs_num:]:
                        self.req = idx+1
                        self.av_emp = self.av_emp - 1 if emp_num else 1
                        break

        if i == 2 and self.req == total and self.av_emp:
            for idx, fs in enumerate(self.fs):
                if fs % level_num == 0 and fs != 0 and 0 in self.av[:cs_num] and not self.emp_tt[idx+cs_num]:
                    self.req = idx+1+cs_num
                    self.av_emp = self.av_emp - 1 if emp_num else 1
                    break


def get_tt(req, assign):
    tt_emp = 0
    if not req:
        tt = travel_time[req][assign] if travel_time else 0
    elif interlock and req <= cs_num and assign < cs_num:
        tt = travel_time[3][0]                                                      # travel inside pair stations
        tt_emp = tt if emp_num else 0
    elif (req <= cs_num and assign < cs_num) or (req > cs_num and assign >= cs_num):
        tt = 0            # fail
    elif req <= cs_num:
        tt = travel_time[assign-cs_num+1][req-1] if travel_time else 0
        tt_emp = tt if emp_num else 0
    elif req > cs_num:
        tt = travel_time[req-cs_num][assign] if travel_time else 0
        tt_emp = tt if emp_num else 0
    return tt, tt_emp


def check_interlock(act, req, av, cs_arr, state):
    lock = False
    crash = False
    stuck = False
    if interlock:
        if req:
            if req-1 % 2 and req-1 < interlock and 0 not in av[cs_num:] and act == req-2:
                lock = True

        if act < interlock and not act % 2:
            if av[act+1] and act != req-2:
                crash = True

        if act < interlock and act % 2 and cs_arr[act-1]:
            if state < SF and state % level_num and not cs_arr[act-1] % level_num:
                stuck = True

    return lock, crash, stuck


def soc_and_fl(info):
    soc = (math.floor((info - 1) / level_num)) * 100 / (level_num - 1)
    fl = ((info-1) % level_num) * 100 / (level_num - 1)
    return int(soc), int(fl)


def check_action_validity(self, action, check):
    req = 0 if check else self.req
    obs = np.hstack([self.ent, self.cs, self.fs, self.req])
    obs = np.array(obs, dtype=int)
    if not req:
        state_info = self.ent
    elif req <= cs_num:
        state_info = self.cs[req - 1]
    elif req > cs_num and req != total:
        state_info = self.fs[req - (cs_num + 1)]
    else:
        state_info = 0
    lock, lock_crash, stuck = check_interlock(action, req, self.av, self.cs, state_info)
    wrong_fs = action >= cs_num >= req != 0 and self.cs[req - 1] == SF
    wrong_cs = req <= cs_num and req != 0 and (action < cs_num and action < interlock and action % 2)
    if obs[action + 1]:
        return False                                    # avoid crash
    if lock_crash:
        return False
    if stuck:
        return False
    if not req:
        return True
    elif req <= cs_num and action < cs_num and not lock and self.cs[req - 1] != SF:
        return False                                    # avoid same_cs
    elif req > cs_num and action >= cs_num:
        return False                                    # avoid same_fs
    elif wrong_fs:
        return False
    elif wrong_cs:
        return False
    else:
        return True
