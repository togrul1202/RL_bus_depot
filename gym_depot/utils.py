import numpy as np
import yaml


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

def ent_array():
    if not random:
        np.random.seed(seed)
    duration = np.random.random_integers(min_dur, max_dur)
    zeros = duration  # zero = time step
    array = np.array([1] * (bus_num-2) + [0] * zeros)

    # Shuffle the array
    np.random.shuffle(array)
    array = np.concatenate(([1], array, [1]))
    array = array.tolist()
    print(f'entrance array is: {array}')
    if bus_time:        # change arriving time of buses at the ent based on bus_time
        array = add_waiting(array, bus_time)
    print(f'entrance array is: {array}')
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
                if not random:
                    np.random.seed(seed)
                ent = np.random.choice([1, 2, 4, 5])  # self.ent_config[0]
            del ent_config[0]
        else:
            if 0 in ent_config and td[0]:
                ent_config.remove(0)
                print('oh hi mark')
    return ent_config, ent


def cs_update(cs_val, ts):
    for idx, cs in enumerate(cs_val):
        if cs != 0 and cs < 7:
            if ts[idx+1] == charge:
                cs_val[idx] += 3
                ts[idx+1] = 0
        else:
            ts[idx+1] = 0
    return cs_val, ts


def fs_update(fs_val, ts):
    for idx, fs in enumerate(fs_val):
        if fs % 3 != 0 and fs != 0:
            if ts[idx+1+cs_num] == fuel:
                fs_val[idx] += 1
                ts[idx+1+cs_num] = 0
        else:
            ts[idx+1+cs_num] = 0
    return fs_val, ts


def delay_update(td, ent, av, cs_val, fs_val):
    if ent and 0 not in av:
        td[0] += 1

    for idx, cs in enumerate(cs_val):
        if (cs == 7 or cs == 8) and 0 not in av[cs_num:]:
            td[idx + 1] += 1

    for idx, fs in enumerate(fs_val):
        if fs % 3 == 0 and fs != 0 and 0 not in av[:cs_num]:
            td[idx+1+cs_num] += 1
    return td


def update(req, av, ent, ent_config, cs_val, fs_val, ts, td):
    req = request(req, av, ent, cs_val, fs_val)  # get request
    if req == total:
        ts += 1
        #print(ts)
        td = delay_update(td, ent, av, cs_val, fs_val)  # update td
        ent_config, ent = ent_update(ent_config, ent, td)  # update ent_config
        cs_val, ts = cs_update(cs_val, ts)  # update cl
        fs_val, ts = fs_update(fs_val, ts)  # update fl
    return req, ent, ent_config, cs_val, fs_val, ts, td


def av_state(av, cs_val, fs_val):
    for idx, cs in enumerate(cs_val):
        av[idx] = 0 if cs == 0 else 1
    for idx, fs in enumerate(fs_val):
        av[idx+params['cs_num']] = 0 if fs == 0 else 1
    return av


def request(req, av, ent, cs_val, fs_val):
    for i in req_order:
        if i == 0 and req == total and ent and 0 in av:
            req = 0

        if i == 1 and req == total:
            for idx, cs in enumerate(cs_val):
                if (cs == 7 or cs == 8) and 0 in av[params['cs_num']:]:
                    req = idx+1
                    break

        if i == 2 and req == total:
            for idx, fs in enumerate(fs_val):
                if fs % 3 == 0 and fs != 0 and 0 in av[:params['cs_num']]:
                    req = idx+1+params['cs_num']
                    break
    return req

