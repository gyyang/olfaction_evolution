import copy
import numpy as np
from scipy import stats as sstats
from collections import defaultdict
import itertools


def filter(res, filter_dict=None):
    '''
    Filter results to only include entries containing the key, value pairs in select_dict
    For example, select_dict = {'day', 0} will filter all data from res whose key 'day' == 0
    :param res: flattened dict of results
    :param filter_dict:
    :return: a copy of res with filter applied
    '''
    if filter_dict is None:
        return res
    out = copy.copy(res)
    list_of_ixs = []
    for key, vals in filter_dict.items():
        membership = np.isin(res[key], vals)
        list_of_ixs.append(membership)
    select_ixs = np.all(list_of_ixs, axis=0)
    for key, value in res.items():
        out[key] = value[select_ixs]
    return out


def exclude(res, exclude_dict=None):
    if exclude_dict is None:
        return res
    out = copy.copy(res)
    list_of_ixs = []
    for key, vals in exclude_dict.items():
        membership = np.isin(res[key], vals)
        list_of_ixs.append(membership)
    exclude_ixs = np.all(list_of_ixs, axis=0)
    select_ixs = np.invert(exclude_ixs)

    for key, value in res.items():
        out[key] = value[select_ixs]
    return out

def filter_reduce(res, filter_keys, reduce_key):
    #TODO: have not tested behavior
    out = defaultdict(list)
    if isinstance(filter_keys, str):
        filter_keys = [filter_keys]
    unique_combinations, ixs = retrieve_unique_entries(res, filter_keys)
    for v in unique_combinations:
        filter_dict = {filter_key: val for filter_key, val in zip(filter_keys, v)}
        cur_res = filter(res, filter_dict)
        temp_res = reduce_by_mean(cur_res, reduce_key)
        chain_defaultdicts(out, temp_res)
    for key, val in out.items():
        out[key] = np.array(val)
    return out

def reduce_by_mean(res, key):
    #TODO: have not tested behavior
    data = res[key]
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    sem = sstats.sem(data, axis=0)

    out = defaultdict(list)
    for k, v in res.items():
        if k == key:
            out[k] = mean
            out[k + '_std'] = std
            out[k + '_sem'] = sem
        else:
            if len(set(v)) == 1:
                out[k] = v[0]
            else:
                list_elements = ','.join([str(x) for x in np.unique(v)])
                print('Took the mean of non-unique list elements: ' + list_elements)
    return out

def retrieve_unique_entries(res, loop_keys):
    unique_entries_per_loopkey = []
    for x in loop_keys:
        a = res[x]
        indexes = np.unique(a, return_index=True)[1]
        unique_entries_per_loopkey.append([a[index] for index in sorted(indexes)])

    unique_entry_combinations = list(itertools.product(*unique_entries_per_loopkey))
    list_of_ind = []
    for x in range(len(unique_entry_combinations)):
        list_of_ixs = []
        cur_combination = unique_entry_combinations[x]
        for i, val in enumerate(cur_combination):
            list_of_ixs.append(val == res[loop_keys[i]])
        ind = np.all(list_of_ixs, axis=0)
        ind_ = np.where(ind)[0]
        list_of_ind.append(ind_)
    return unique_entry_combinations, list_of_ind

def chain_defaultdicts(dictA, dictB):
    for k in dictB.keys():
        dictA[k] = list(itertools.chain(dictA[k], dictB[k]))
    for key, val in dictA.items():
        dictA[key] = np.array(val)