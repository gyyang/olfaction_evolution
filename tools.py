import os
import json
import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import configs


def save_config(config, save_path):
    """Save config."""
    # config_dict = {k: getattr(config, k) for k in dir(config) if k[0] != '_'}
    config_dict = config.__dict__
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config_dict, f)


def load_config(save_path):
    """Load config."""
    with open(os.path.join(save_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)

    config = configs.BaseConfig()
    for key, val in config_dict.items():
        setattr(config, key, val)
    return config


def varying_config(experiment, i):
    """Training a specific hyperparameter settings.

    Args:
        experiment: function handle for the experiment, has to take an integer
            as input
        i: integer, indexing the specific hyperparameter setting to be used

       hp['a']=[0,1], hp['b']=[0,1], hp['c']=[0,1], there are 8 possible combinations

    Return:
        config: new configuration
    """
    # Ranges of hyperparameters to loop over
    config, hp_ranges = experiment()

    # Unravel the input index
    keys = hp_ranges.keys()
    dims = [len(hp_ranges[k]) for k in keys]
    n_max = np.prod(dims)
    indices = np.unravel_index(i % n_max, dims=dims)

    if i >= n_max:
        return False

    # Set up new hyperparameter
    for key, index in zip(keys, indices):
        setattr(config, key, hp_ranges[key][index])
    return config


def varying_config_sequential(experiment, i):
    """Training specific combinations of hyper-parameter settings

    Args:
        experiment: function handle for the experiment, has to take an integer as input
        i: integer, indexing the specific hyperparameter settings to be used

       unlike varying_config, this function does not iterate through all possible
       hyper-parameter combinations.

       hp['a']=[0,1], hp['b']=[0,1], hp['c']=[0,1].
       possible combinations are {'a':0,'b':0,'c':0}, and {'a':1,'b':1,'c':1}

    Returns:
        config: new configuration
    """
    config, hp_ranges = experiment()

    # Unravel the input index
    keys = hp_ranges.keys()
    dims = [len(hp_ranges[k]) for k in keys]
    n_max = dims[0]

    if i >= n_max:
        return False

    for key in keys:
        setattr(config, key, hp_ranges[key][i])
    return config


def load_all_results(rootpath, argLast= True):
    """Load results from path.

    Args:
        rootpath: root path of all models loading results from

    Returns:
        res: dictionary of numpy arrays, containing information from all models
    """
    dirs = [os.path.join(rootpath, n) for n in os.listdir(rootpath)]

    from collections import defaultdict
    res = defaultdict(list)

    for i, d in enumerate(dirs):
        log_name = os.path.join(d, 'log.pkl')
        with open(log_name, 'rb') as f:
            log = pickle.load(f)
        config = load_config(d)
        for key, val in log.items():
            if argLast:
                res[key].append(val[-1])  # store last value in log
            else:
                res[key].append(val)
        for k in dir(config):
            if k[0] != '_':
                res[k].append(getattr(config, k))
    # TODO: exclude models that didn't finish training
    for key, val in res.items():
        res[key] = np.array(val)
    return res


nicename_dict = {
        'ORN_NOISE_STD': 'Noise level',
        'N_KC': 'Number of KCs',
        'N_ORN_DUPLICATION': 'ORNs per type',
        'kc_inputs': 'Number of KC Claws',
        'glo_score': 'GloScore',
        'val_acc': 'Accuracy',
        'val_loss': 'Loss',
        'epoch': 'Epoch',
        }


def nicename(name):
    """Return nice name for publishing."""
    try:
        return nicename_dict[name]
    except KeyError:
        return name


def _reshape_worn(w_orn, unique_orn, mode='tile'):
    """Reshape w_orn."""
    n_orn, n_pn = w_orn.shape
    w_orn_by_pn = abs(w_orn)
    n_duplicate_orn = n_orn // unique_orn
    if mode == 'repeat':
        w_orn_by_pn = np.reshape(w_orn_by_pn,
                                 (unique_orn, n_duplicate_orn, n_pn))
        w_orn_by_pn = np.swapaxes(w_orn_by_pn, 0, 1)
    elif mode == 'tile':
        w_orn_by_pn = np.reshape(w_orn_by_pn,
                                 (n_duplicate_orn, unique_orn, n_pn))
    else:
        raise ValueError('Unknown mode' + str(mode))
    return w_orn_by_pn


def compute_glo_score(w_orn, unique_orn, mode='tile'):
    """Compute the glomeruli score in numpy.

    This function returns the glomeruli score, a number between 0 and 1 that
    measures how close the connectivity is to glomeruli connectivity.

    For one glomeruli neuron, first we compute the average connections from
    each ORN group. Then we sort the absolute connection weights by ORNs.
    The glomeruli score is simply:
        (Max weight - Second max weight) / (Max weight + Second max weight)

    Args:
        w_orn: numpy array (n_orn, n_pn). This matrix has to be organized
        in the following ways:
        In the mode=='repeat'
            neurons from the same orn type are indexed consecutively
            for example, neurons from the 0-th type would be 0, 1, 2, ...
        In the mode=='tile'
            neurons from the same orn type are spaced by the number of types,
            for example, neurons from the 0-th type would be 0, 50, 100, ...
        unique_orn: int, the number of unique ORNs
        mode: the way w_orn is organized

    Return:
        avg_glo_score: scalar, average glomeruli score
        glo_scores: numpy array (n_pn,), all glomeruli scores
    """
    n_orn, n_pn = w_orn.shape
    w_orn_by_pn = _reshape_worn(w_orn, unique_orn, mode)
    w_orn_by_pn = w_orn_by_pn.mean(axis=0)

    glo_scores = list()
    for i in range(n_pn):
        w_tmp = w_orn_by_pn[:, i]  # all projections to the i-th PN neuron
        indsort = np.argsort(w_tmp)[::-1]
        w_max = w_tmp[indsort[0]]
        w_second = w_tmp[indsort[1]]
        glo_score = (w_max - w_second) / (w_max + w_second)
        glo_scores.append(glo_score)

    avg_glo_score = np.mean(glo_scores)
    return avg_glo_score, glo_scores


def compute_sim_score(w_orn, unique_orn, mode='tile'):
    """Compute the similarity score in numpy.

    This function returns the glomeruli score, a number between 0 and 1 that
    measures how close the connectivity is to glomeruli connectivity.

    For one glomeruli neuron, first we compute the average connections from
    each ORN group. Then we sort the absolute connection weights by ORNs.
    The glomeruli score is simply:
        (Max weight - Second max weight) / (Max weight + Second max weight)

    Args:
        w_orn: numpy array (n_orn, n_pn). This matrix has to be organized
        in the following ways:
        In the mode=='repeat'
            neurons from the same orn type are indexed consecutively
            for example, neurons from the 0-th type would be 0, 1, 2, ...
        In the mode=='tile'
            neurons from the same orn type are spaced by the number of types,
            for example, neurons from the 0-th type would be 0, 50, 100, ...
        unique_orn: int, the number of unique ORNs
        mode: the way w_orn is organized

    Return:
        avg_glo_score: scalar, average glomeruli score
        glo_scores: numpy array (n_pn,), all glomeruli scores
    """
    n_orn, n_pn = w_orn.shape
    w_orn_by_pn = _reshape_worn(w_orn, unique_orn, mode)
    n_duplicate_orn = n_orn // unique_orn
    if n_duplicate_orn == 1:
        return 0, [0]*unique_orn

    sim_scores = list()
    for i in range(unique_orn):
        w_tmp = w_orn_by_pn[:, i, :]
        sim_tmp = cosine_similarity(w_tmp)
        sim_scores.append(sim_tmp.mean())

    avg_sim_score = np.mean(sim_scores)
    return avg_sim_score, sim_scores
