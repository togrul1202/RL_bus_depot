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


def cs_update(cs_arr, ts, tt, tt_emp, av_emp, emp_tt):
    for idx, cs in enumerate(cs_arr):
        if emp_tt[idx] < 0:                                         # waiting for employee
            continue
        elif tt[idx] or tt_emp[idx]:                                 # TODO: add emp_time
            if tt_emp[idx] == tt[idx] != 0:                        # employee in a bus
                tt_emp[idx] -= 1
                if not tt_emp[idx] and cs_arr[idx]:
                    av_emp += 1
            if idx < interlock and not idx % 2:
                if not (tt[idx+1] and not cs_arr[idx+1]):
                    tt[idx] -= 1
                else:                                            # on travel request: no extra waiting for next station
                    tt[idx] = tt[idx+1]
                    tt[idx+1] = 0
                    tt_emp[idx+1] = tt[idx+1]
                    tt[idx] -= 1
                    tt_emp[idx] -= 1
                    if not tt_emp[idx] and cs_arr[idx]:
                        av_emp += 1
            else:
                tt[idx] -= 1
            ts[idx + 1] = 0
        else:
            if cs != 0 and cs < 7:
                if ts[idx+1] == charge:
                    cs_arr[idx] += 3
                    ts[idx+1] = 0
            else:
                ts[idx+1] = 0
    return cs_arr, ts, tt, tt_emp, av_emp, emp_tt


def fs_update(fs_arr, ts, tt, tt_emp, av_emp, emp_tt):
    for idx, fs in enumerate(fs_arr):
        if emp_tt[idx+cs_num] < 0:
            continue
        elif tt[idx+cs_num] or tt_emp[idx+cs_num]:                                  # TODO: add emp_time
            if tt_emp[idx+cs_num] == tt[idx+cs_num] != 0:                        # employee in a bus
                tt_emp[idx+cs_num] -= 1
                if not tt_emp[idx+cs_num]:
                    av_emp += 1
            tt[idx+cs_num] -= 1
            ts[idx+1+cs_num] = 0
        else:
            if fs % 3 != 0 and fs != 0:
                if ts[idx+1+cs_num] == fuel:
                    fs_arr[idx] += 1
                    ts[idx+1+cs_num] = 0
            else:
                ts[idx+1+cs_num] = 0
    return fs_arr, ts, tt, tt_emp, av_emp, emp_tt


def delay_update(td, ent, av, cs_arr, fs_arr, av_emp, emp_tt):
    if ent and 0 not in av:
        td[0] += 1
    for idx, cs in enumerate(cs_arr):
        if cs == 7 or cs == 8:
            if 0 not in av[cs_num:] or not av_emp or emp_tt[idx] or (idx % 2 and idx < interlock and av[idx - 1]):
                td[idx + 1] += 1
    for idx, fs in enumerate(fs_arr):
        if fs % 3 == 0 and fs != 0 and (0 not in av[:cs_num] or not av_emp or emp_tt[idx+cs_num]):
            td[idx+1+cs_num] += 1
    return td


def emp_tt_update(cs_arr, fs_arr, emp_tt, tt, tt_emp):
    for idx, t in enumerate(emp_tt):                           # on travel req: don't wait for an employee
        if idx < interlock and not idx % 2 and tt[idx+1] and emp_tt[idx] == -(idx+2):
            emp_tt[idx] = 0
            emp_tt[idx+1] = 0
            cs_arr[idx+1] = 0
        elif t > 0:
            emp_tt[idx] -= 1
            if not emp_tt[idx]:
                tt[emp_tt == -(idx + 1)] += 1
                tt_emp[emp_tt == -(idx + 1)] = tt[emp_tt == -(idx + 1)]
                emp_tt[emp_tt == -(idx + 1)] = 0
                if idx < cs_num:
                    cs_arr[idx] = 0
                else:
                    fs_arr[idx-cs_num] = 0
    #print(f'emp_tt: {emp_tt}\ncs_arr: {cs_arr}\nfs_arr: {fs_arr}')
    return cs_arr, fs_arr, emp_tt, tt, tt_emp


def update(req, av, ent, ent_config, cs_arr, fs_arr, ts, td, tt, tt_emp, av_emp, emp_tt):
    req, av_emp = request(req, av, ent, cs_arr, fs_arr, av_emp, emp_tt, tt)  # get request
    if req == total:
        ts += 1
        #print(ts)
        td = delay_update(td, ent, av, cs_arr, fs_arr, av_emp, emp_tt)  # update td
        if emp_num and emp_time:
            cs_arr, fs_arr, emp_tt, tt, tt_emp = emp_tt_update(cs_arr, fs_arr, emp_tt, tt, tt_emp)  # update emp_tt and reset cs and fs if zero
            av = av_state(av, cs_arr, fs_arr)
        ent_config, ent = ent_update(ent_config, ent, td)  # update ent_config
        cs_arr, ts, tt, tt_emp, av_emp, emp_tt = cs_update(cs_arr, ts, tt, tt_emp, av_emp, emp_tt)  # update cl
        fs_arr, ts, tt, tt_emp, av_emp, emp_tt = fs_update(fs_arr, ts, tt, tt_emp, av_emp, emp_tt)  # update fl
        #print(f'emp_av: {av_emp}\nemp_tt: {emp_tt}\ntt: {tt}\ntt_emp: {tt_emp}\ntd: {td}')
    return req, av, ent, ent_config, cs_arr, fs_arr, ts, td, tt, tt_emp, av_emp, emp_tt


def av_state(av, cs_arr, fs_arr):
    for idx, cs in enumerate(cs_arr):
        if cs:
            av[idx] = 1
            if idx % 2 and idx < interlock:
                av[idx - 1] = 1
        else:
            av[idx] = 0
    for idx, fs in enumerate(fs_arr):
        av[idx+cs_num] = 1 if fs else 0
    return av


def request(req, av, ent, cs_arr, fs_arr, av_emp, emp_tt, tt):
    for i in req_order:
        if i == 0 and req == total and ent and 0 in av:
            req = 0

        if i == 1 and req == total and av_emp:
            for idx, cs in enumerate(cs_arr):
                if not emp_tt[idx]:                        # avoid requesting again while waiting for an employee
                    if idx % 2 and idx < interlock and cs >= 7:
                        if not cs_arr[idx-1] and not tt[idx-1]:
                            req = idx + 1
                            if not tt[idx]:                 # on travel request: don't ask for another employee
                                av_emp = av_emp - 1 if emp_num else 1
                            break
                    elif (cs == 7 or cs == 8) and 0 in av[cs_num:]:
                        req = idx+1
                        av_emp = av_emp - 1 if emp_num else 1
                        break

        if i == 2 and req == total and av_emp:
            for idx, fs in enumerate(fs_arr):
                if fs % 3 == 0 and fs != 0 and 0 in av[:cs_num] and not emp_tt[idx+cs_num]:
                    req = idx+1+cs_num
                    av_emp = av_emp - 1 if emp_num else 1
                    break
    return req, av_emp


def get_tt(req, assign):
    tt_emp = 0
    if not req:
        tt = travel_time[req][assign] if travel_time else 0
    elif interlock and req <= cs_num and assign < cs_num:
        tt = 2                                                      # TODO: CS travel inside
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

        if act % 2 and cs_arr[act-1] and act < interlock:
            if state < 9 and state % 3 and not cs_arr[act-1] % 3:
                stuck = True

    return lock, crash, stuck

