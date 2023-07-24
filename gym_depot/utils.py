import numpy as np
import yaml
import pygame


with open('gym_depot/config.yaml') as stream:
    try:
        params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

bus_num = params['bus_num']
bus_time = params['bus_time']
min_dur = params['min_duration']
max_dur = params['max_duration']
charge = params['charge_time']
fuel = params['fuel_time']
cs_num = params['cs_num']
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
if shape <= 9:
    value = 9
else:
    value = shape
time_delay = params['time_delay']
travel_time = params['travel_time']
interlock = params['interlock']
emp_num = params['emp_num']
emp_time = params['emp_time']

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


def ent_update(ent_config, ent, td):
    if len(ent_config):
        if not ent:
            if ent_config[0]:
                ent = np.random.choice([1, 2, 4, 5])  # self.ent_config[0]
                #print(f'ent: {ent}')
            del ent_config[0]
        else:
            if 0 in ent_config and td[0]:
                ent_config.remove(0)
    return ent_config, ent


def cs_update(cs_arr, ts, tt, tt_emp, av_emp, emp_tt, last_loc):
    for idx, cs in enumerate(cs_arr):
        if emp_tt[idx]:
            emp_tt[idx] -= 1
        elif tt[idx] or tt_emp[idx]:                                 # TODO: add emp_time
            if tt_emp[idx] == tt[idx] != 0:                        # employee in a bus
                tt_emp[idx] -= 1
                if not tt_emp[idx]:
                    av_emp += 1
                    last_loc[idx] = 1
            tt[idx] -= 1
            ts[idx + 1] = 0
        else:
            if cs != 0 and cs < 7:
                if ts[idx+1] == charge:
                    cs_arr[idx] += 3
                    ts[idx+1] = 0
            else:
                ts[idx+1] = 0

    return cs_arr, ts, tt, tt_emp, av_emp, emp_tt, last_loc


def fs_update(fs_arr, ts, tt, tt_emp, av_emp, emp_tt, last_loc):
    for idx, fs in enumerate(fs_arr):
        if emp_tt[idx + cs_num]:
            emp_tt[idx + cs_num] -= 1
        elif tt[idx+cs_num] or tt_emp[idx+cs_num]:                                  # TODO: add emp_time
            if tt_emp[idx+cs_num] == tt[idx+cs_num] != 0:                        # employee in a bus
                tt_emp[idx+cs_num] -= 1
                if not tt_emp[idx+cs_num]:
                    av_emp += 1
                    last_loc[idx+cs_num] = 1
            tt[idx+cs_num] -= 1
            ts[idx+1+cs_num] = 0
        else:
            if fs % 3 != 0 and fs != 0:
                if ts[idx+1+cs_num] == fuel:
                    fs_arr[idx] += 1
                    ts[idx+1+cs_num] = 0
            else:
                ts[idx+1+cs_num] = 0
    return fs_arr, ts, tt, tt_emp, av_emp, emp_tt, last_loc


def delay_update(td, ent, av, cs_arr, fs_arr, av_emp):
    if ent and 0 not in av:
        td[0] += 1

    for idx, cs in enumerate(cs_arr):
        if cs == 7 or cs == 8:
            if not av_emp:
                td[idx + 1] += 1                                    # TODO: and av_emp
            elif 0 not in av[cs_num:]:
                td[idx + 1] += 1
            elif idx % 2 and idx < interlock and av[idx - 1]:
                td[idx + 1] += 1
    for idx, fs in enumerate(fs_arr):
        if fs % 3 == 0 and fs != 0 and (0 not in av[:cs_num] or not av_emp):    # TODO: and av_emp
            td[idx+1+cs_num] += 1
    return td


def update(req, av, ent, ent_config, cs_arr, fs_arr, ts, td, tt, tt_emp, av_emp, emp_tt, last_loc):
    req, av_emp = request(req, av, ent, cs_arr, fs_arr, av_emp)  # get request
    if req == total:
        ts += 1
        #print(ts)
        td = delay_update(td, ent, av, cs_arr, fs_arr, av_emp)  # update td
        ent_config, ent = ent_update(ent_config, ent, td)  # update ent_config
        cs_arr, ts, tt, tt_emp, av_emp, emp_tt, last_loc = cs_update(cs_arr, ts, tt, tt_emp, av_emp, emp_tt, last_loc)  # update cl
        fs_arr, ts, tt, tt_emp, av_emp, emp_tt, last_loc = fs_update(fs_arr, ts, tt, tt_emp, av_emp, emp_tt, last_loc)  # update fl
    return req, ent, ent_config, cs_arr, fs_arr, ts, td, tt, tt_emp, av_emp, emp_tt, last_loc


def av_state(av, cs_arr, fs_arr):
    for idx, cs in enumerate(cs_arr):
        if not cs:
            av[idx] = 0
        else:
            av[idx] = 1
            if idx % 2 and idx < interlock:
                av[idx-1] = 1
    for idx, fs in enumerate(fs_arr):
        av[idx+cs_num] = 0 if fs == 0 else 1
    return av


def request(req, av, ent, cs_arr, fs_arr, av_emp):
    for i in req_order:
        if i == 0 and req == total and ent and 0 in av:
            req = 0

        if i == 1 and req == total and av_emp:
            for idx, cs in enumerate(cs_arr):
                if idx % 2 and idx < interlock and cs >= 7:
                    if not cs_arr[idx-1]:
                        req = idx + 1
                        av_emp = av_emp - 1 if emp_num else 1
                        break
                elif (cs == 7 or cs == 8) and 0 in av[cs_num:]:
                    req = idx+1
                    av_emp = av_emp - 1 if emp_num else 1
                    break

        if i == 2 and req == total and av_emp:
            for idx, fs in enumerate(fs_arr):
                if fs % 3 == 0 and fs != 0 and 0 in av[:cs_num]:
                    req = idx+1+cs_num
                    av_emp = av_emp - 1 if emp_num else 1
                    break
    return req, av_emp


def get_emp_time(req, last_loc):
    tt = np.ones(total_st)*100
    for idx, val in enumerate(last_loc):
        if val == 1:                                    # at a station
            if req <= cs_num <= idx:
                tt[idx] = emp_time[idx-cs_num+1][req-1]
            elif idx < cs_num < req:
                tt[idx] = emp_time[req-cs_num][idx]
        elif val == 2:                                  # at SB
            tt[idx] = emp_time[0][req-1]
        else:
            tt[idx] = 100                               # random high value
    time = min(tt)
    last_loc[np.argmin(tt)] = 0
    return last_loc, time


def get_tt(req, assign, last_loc):
    tt_emp = 0
    emp_tt = 0
    if not req:
        tt = travel_time[req][assign] if travel_time else 0
    elif interlock and req <= cs_num and assign < cs_num:
        tt = 2                                                      # TODO: CS travel inside
        tt_emp = tt if emp_num else 0
        emp_tt = 2 if emp_num and emp_time else 0
    elif (req <= cs_num and assign < cs_num) or (req > cs_num and assign >= cs_num):
        tt = 0            # fail
        emp_tt = 0
    elif req <= cs_num:
        tt = travel_time[assign-cs_num+1][req-1] if travel_time else 0
        tt_emp = tt if emp_num else 0
        if emp_num and emp_time:
            last_loc, emp_tt = get_emp_time(req, last_loc)
    elif req > cs_num:
        tt = travel_time[req-cs_num][assign] if travel_time else 0
        tt_emp = tt if emp_num else 0
        if emp_num and emp_time:
            last_loc, emp_tt = get_emp_time(req, last_loc)
    return tt, tt_emp, last_loc, emp_tt


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

        if act % 2 and cs_arr[act-1] and act < interlock:
            if state < 9 and state % 3 and not cs_arr[act-1] % 3:
                stuck = True

    return lock, crash, stuck

