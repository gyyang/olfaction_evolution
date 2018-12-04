import os
import json
import numpy as np
import train


def save_config(config, save_path):
    """Save config."""
    config_dict = {k: getattr(config, k) for k in dir(config) if k[0] != '_'}
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config_dict, f)


def load_config(save_path):
    """Load config."""
    with open(os.path.join(save_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)

    class Config():
        pass

    config = Config()
    for key, val in config_dict.items():
        setattr(config, key, val)
    return config

def varying_config(experiment, i):
    # Ranges of hyperparameters to loop over
    config, hp_ranges = experiment(i)

    # Unravel the input index
    keys = hp_ranges.keys()
    dims = [len(hp_ranges[k]) for k in keys]
    n_max = np.prod(dims)
    indices = np.unravel_index(i % n_max, dims=dims)

    if i >= n_max:
        return

    # Set up new hyperparameter
    for key, index in zip(keys, indices):
        setattr(config, key, hp_ranges[key][index])
    train.train(config)

def compute_glo_score(w_orn):
    """Compute the glomeruli score in numpy.

    This function returns the glomeruli score, a number between 0 and 1 that
    measures how close the connectivity is to glomeruli connectivity.

    For one glomeruli neuron, first we compute the average connections from
    each ORN group. Then we sort the absolute connection weights by ORNs.
    The glomeruli score is simply:
        (Max weight - Second max weight) / (Max weight + Second max weight)

    Args:
        w_orn: numpy array (n_orn, n_pn). This matrix has to be organized
        in the following way: the n_orn neurons are grouped into n_pn groups
        index 0, ..., n_pn_per_orn - 1 is the 0-th group, and so on

    Return:
        avg_glo_score: scalar, average glomeruli score
        glo_scores: numpy array (n_pn,), all glomeruli scores
    """
    unique_orn = 50
    n_orn, n_pn = w_orn.shape
    w_orn_by_pn = abs(w_orn)
    n_duplicate_orn = n_orn // unique_orn
    w_orn_by_pn = np.reshape(w_orn_by_pn, (unique_orn, n_duplicate_orn, n_pn))
    w_orn_by_pn = w_orn_by_pn.mean(axis=1)
    w_orn_by_pn = abs(w_orn_by_pn)

    # this code does **not** work for arbitrary orn / pn sizes
    n_pn_per_orn = n_orn // n_pn
    # w_orn_by_pn = np.reshape(w_orn, (n_pn, n_pn_per_orn, n_pn))
    # w_orn_by_pn = w_orn_by_pn.mean(axis=1)
    # w_orn_by_pn = abs(w_orn_by_pn)

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
