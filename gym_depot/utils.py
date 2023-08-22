import numpy as np
import yaml
import pygame
import math


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
fast_charge = params['fast_charge_time']
fuel = params['fuel_time']
level_num = params['level_num']
SF = level_num**2
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
travel_time = params['travel_time']
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
emp_time = params['emp_time']
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


def ent_update(ent_config, ent, td):
    if len(ent_config):
        if not ent:
            if ent_config[0]:
                ent = np.random.choice([a for a in range(SF-level_num) if a % level_num])  # self.ent_config[0]
                #print(f'ent: {ent}')
            del ent_config[0]
        else:
            if 0 in ent_config and td[0]:
                ent_config.remove(0)
    return ent_config, ent


def cs_update(cs_arr, ts, tt, tt_emp, av_emp, emp_tt, emp_loc):
    for idx, cs in enumerate(cs_arr):
        emp_in = False
        if emp_tt[idx] < 0:                                         # waiting for employee
            continue
        elif tt[idx] or tt_emp[idx]:
            if tt_emp[idx] == tt[idx] != 0:                        # employee in a bus
                emp_in = True
                tt_emp[idx] -= 1
                if not tt_emp[idx] and cs_arr[idx]:
                    if idx < interlock and not idx % 2:
                        if not (tt[idx+1] and not cs_arr[idx+1]):           # on travel request: no extra emp
                            av_emp += 1
                            emp_loc[idx + 1] += 1
                    else:
                        av_emp += 1
                        emp_loc[idx+1] += 1
            if idx < interlock and not idx % 2:
                if not (tt[idx+1] and not cs_arr[idx+1]):
                    if not tt[idx] or tt_emp[idx] < 0:
                        print(tt_emp[idx])
                    tt[idx] -= 1
                else:                                            # on travel request: no extra waiting for next station
                    tt[idx] = tt[idx+1]
                    tt[idx+1] = 0
                    if emp_num:
                        tt_emp[idx + 1] = tt[idx + 1]
                        if emp_in:
                            tt_emp[idx] = tt[idx]
                            tt_emp[idx] -= 1
                        if not tt_emp[idx] and cs_arr[idx]:
                            av_emp += 1
                            emp_loc[idx+1] += 1
                    if not tt[idx] or tt_emp[idx] < 0:
                        print(tt_emp[idx])
                    tt[idx] -= 1
            else:
                if not tt[idx] or tt_emp[idx] < 0:
                    print(tt_emp[idx])
                tt[idx] -= 1
            ts[idx + 1] = 0
        else:
            if cs != 0 and cs <= SF-level_num:
                if idx < fast_cs_num:
                    if ts[idx + 1] == fast_charge:
                        cs_arr[idx] += level_num
                        ts[idx + 1] = 0
                else:
                    if ts[idx+1] == charge:
                        cs_arr[idx] += level_num
                        ts[idx+1] = 0
            else:
                ts[idx+1] = 0
    return cs_arr, ts, tt, tt_emp, av_emp, emp_tt, emp_loc


def fs_update(fs_arr, ts, tt, tt_emp, av_emp, emp_tt, emp_loc):
    for idx, fs in enumerate(fs_arr):
        if emp_tt[idx+cs_num] < 0:
            continue
        elif tt[idx+cs_num] or tt_emp[idx+cs_num]:
            if tt_emp[idx+cs_num] == tt[idx+cs_num] != 0:                        # employee in a bus
                tt_emp[idx+cs_num] -= 1
                if not tt_emp[idx+cs_num]:
                    av_emp += 1
                    emp_loc[idx+1+cs_num] += 1
            tt[idx+cs_num] -= 1
            ts[idx+1+cs_num] = 0
        else:
            if fs % level_num != 0 and fs != 0:
                if ts[idx+1+cs_num] == fuel:
                    fs_arr[idx] += 1
                    ts[idx+1+cs_num] = 0
            else:
                ts[idx+1+cs_num] = 0
    return fs_arr, ts, tt, tt_emp, av_emp, emp_tt, emp_loc


def delay_update(td, ent, av, cs_arr, fs_arr, av_emp, emp_tt):
    if ent and 0 not in av:
        td[0] += 1
    for idx, cs in enumerate(cs_arr):
        if cs > SF-level_num and cs != SF:
            if 0 not in av[cs_num:] or not av_emp or emp_tt[idx] or (idx % 2 and idx < interlock and av[idx - 1]):
                td[idx + 1] += 1
    for idx, fs in enumerate(fs_arr):
        if fs % level_num == 0 and fs != 0 and (0 not in av[:cs_num] or not av_emp or emp_tt[idx+cs_num]):
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


def emp_to_sb(emp_loc, emp_timer):
    emp_timer = np.multiply(emp_timer, emp_loc[1:])                # reset timer of unavailable employees
    for loc, emp in enumerate(emp_loc):
        if loc and emp:
            if emp_timer[loc-1] >= emp_time[0][loc-1]/2:           # consider emp at SB if half of the emp_time the last loc to SB passed
                emp_loc[loc] -= 1
                emp_loc[0] += 1
            else:
                emp_timer[loc-1] += 1
    return emp_loc, emp_timer


def update(req, av, ent, ent_config, cs_arr, fs_arr, ts, td, tt, tt_emp, av_emp, emp_tt, emp_loc, emp_timer):
    req, av_emp = request(req, av, ent, cs_arr, fs_arr, av_emp, emp_tt, tt)  # get request
    if req == total:
        ts += 1
        #print(ts)
        td = delay_update(td, ent, av, cs_arr, fs_arr, av_emp, emp_tt)  # update td
        emp_loc, emp_timer = emp_to_sb(emp_loc, emp_timer)        # employee goes back to SB
        if emp_num and emp_time:
            cs_arr, fs_arr, emp_tt, tt, tt_emp = emp_tt_update(cs_arr, fs_arr, emp_tt, tt, tt_emp)  # update emp_tt and reset cs and fs if zero
            av = av_state(av, cs_arr, fs_arr)
        ent_config, ent = ent_update(ent_config, ent, td)  # update ent_config
        cs_arr, ts, tt, tt_emp, av_emp, emp_tt, emp_loc = cs_update(cs_arr, ts, tt, tt_emp, av_emp, emp_tt, emp_loc)  # update cl
        fs_arr, ts, tt, tt_emp, av_emp, emp_tt, emp_loc = fs_update(fs_arr, ts, tt, tt_emp, av_emp, emp_tt, emp_loc)  # update fl
        #print(f'emp_av: {av_emp}\nemp_tt: {emp_tt}\ntt: {tt}\ntt_emp: {tt_emp}\ntd: {td}')
    return req, av, ent, ent_config, cs_arr, fs_arr, ts, td, tt, tt_emp, av_emp, emp_tt, emp_loc, emp_timer


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
                    if idx % 2 and idx < interlock and cs > SF-level_num:
                        if not cs_arr[idx-1] and not tt[idx-1]:
                            req = idx + 1
                            if not tt[idx]:                 # on travel request: don't ask for another employee
                                av_emp = av_emp - 1 if emp_num else 1
                            break
                    elif (cs > SF-level_num and cs != SF) and 0 in av[cs_num:]:
                        req = idx+1
                        av_emp = av_emp - 1 if emp_num else 1
                        break

        if i == 2 and req == total and av_emp:
            for idx, fs in enumerate(fs_arr):
                if fs % level_num == 0 and fs != 0 and 0 in av[:cs_num] and not emp_tt[idx+cs_num]:
                    req = idx+1+cs_num
                    av_emp = av_emp - 1 if emp_num else 1
                    break
    return req, av_emp


def get_tt(req, assign):
    tt_emp = 0
    if not req:
        tt = travel_time[req][assign] if travel_time else 0
    elif interlock and req <= cs_num and assign < cs_num:
        tt = travel_time[3]                                                      # travel inside pair stations
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
