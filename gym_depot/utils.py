import numpy as np


def ent_array(bus_num, duration, seed):
    # Set the random seed
    np.random.seed(seed)

    # Create an array with 5 ones and 5 zeros
    array = np.array([1] * bus_num + [0] * duration)

    # Shuffle the array
    np.random.shuffle(array)
    array = array.tolist()

    print(f'entrance array is: {array}')
    return array


def ent_update(ent_config, ent, td, seed):
    if len(ent_config):
        if not ent:
            if ent_config[0]:
                np.random.seed(seed)
                ent = np.random.choice([1, 2, 4, 5])  # self.ent_config[0]
            del ent_config[0]
        else:
            if 0 in ent_config and td[0]:
                ent_config.remove(0)
                print('oh hi mark')
    return ent_config, ent


def cs_update(cs_val, ts, charge):
    for idx, cs in enumerate(cs_val):
        if cs != 0 and cs < 7:
            if ts[idx+1] == charge:
                cs_val[idx] += 3
                ts[idx+1] = 0
        else:
            ts[idx+1] = 0
    return cs_val, ts


def fs_update(fs_val, ts, fuel):
    for idx, fs in enumerate(fs_val):
        if fs % 3 != 0 and fs != 0:
            if ts[idx+6] == fuel:
                fs_val[idx] += 1
                ts[idx+6] = 0
        else:
            ts[idx+6] = 0
    return fs_val, ts


def delay_update(td, ent, av, cs_val, fs_val):
    if ent and 0 not in av:
        td[0] += 1

    for idx, cs in enumerate(cs_val):
        if (cs == 7 or cs == 8) and 0 not in av[5:]:
            td[idx + 1] += 1

    for idx, fs in enumerate(fs_val):
        if fs % 3 == 0 and fs != 0 and 0 not in av[:5]:
            td[idx + 6] += 1
    return td


def update(req_order, req, av, ent, ent_config, cs_val, fs_val, ts, td, charge, fuel, seed):
    req = request(req_order, req, av, ent, cs_val, fs_val)  # get request
    if req == 8:
        ts += 1
        print(ts)
        td = delay_update(td, ent, av, cs_val, fs_val)  # update td
        ent_config, ent = ent_update(ent_config, ent, td, seed)  # update ent_config
        cs_val, ts = cs_update(cs_val, ts, charge)  # update cl
        fs_val, ts = fs_update(fs_val, ts, fuel)  # update fl
    return req, ent, ent_config, cs_val, fs_val, ts, td


def av_state(av, cs_val, fs_val):
    for idx, cs in enumerate(cs_val):
        av[idx] = 0 if cs == 0 else 1
    for idx, fs in enumerate(fs_val):
        av[idx+5] = 0 if fs == 0 else 1
    return av


def request(req_order, req, av, ent, cs_val, fs_val):
    for i in req_order:
        if i == 0 and req == 8 and ent and 0 in av:
            req = 0

        if i == 1 and req == 8:
            for idx, cs in enumerate(cs_val):
                if (cs == 7 or cs == 8) and 0 in av[5:]:
                    req = idx+1
                    break

        if i == 2 and req == 8:
            for idx, fs in enumerate(fs_val):
                if fs % 3 == 0 and fs != 0 and 0 in av[:5]:
                    req = idx+6
                    break
    return req
