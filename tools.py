"""Tools for project."""

import os
import json
import pickle
from pathlib import Path
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors


rootpath = os.path.dirname(os.path.abspath(__file__))
FIGPATH = os.path.join(rootpath, 'figures')

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'


def get_figname(save_path, figname=''):
    # For backward compatability
    if isinstance(save_path, str):
        save_name = os.path.split(save_path)[-1]
    else:
        # save_path is a list of model paths
        print(save_path[0])
        # ugly hack to get experiment name
        save_name = os.path.split(os.path.split(save_path[0])[-2])[-1]
        print(save_name)

    path = os.path.join(FIGPATH, save_name)
    os.makedirs(path, exist_ok=True)
    figname = os.path.join(path, save_name + figname)
    return figname


def save_fig(save_path, figname='', dpi=300, pdf=True, show=False):
    figname = get_figname(save_path, figname)
    plt.savefig(os.path.join(figname + '.png'), dpi=dpi)
    print('Figure saved at: ' + figname)

    if pdf:
        plt.savefig(os.path.join(figname + '.pdf'), transparent=True)
        # plt.savefig(os.path.join(figname + '.svg'), transparent=True, format='svg')
    if show:
        plt.show()
    # plt.close()


def save_config(config, save_path, also_save_as_text = True):
    """Save config."""
    config_dict = config.__dict__
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config_dict, f)

    if also_save_as_text:
        with open(os.path.join(save_path, 'config.txt'), "w") as f:
            for k, v in config_dict.items():
                f.write(str(k) + ' >>> ' + str(v) + '\n\n')


def load_config(save_path):
    """Load config."""
    import configs
    with open(os.path.join(save_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    model_type = config_dict.get('model', None)
    if model_type == 'full':
        if 'meta_lr' in config_dict:
            config = configs.MetaConfig()
        else:
            config = configs.FullConfig()
    elif model_type == 'rnn':
        config = configs.RNNConfig()
    else:
        config = configs.BaseConfig()

    for key, val in config_dict.items():
        setattr(config, key, val)

    try:
        config.n_trueclass_ratio = config.n_trueclass / config.N_CLASS
    except AttributeError:
        pass

    return config


def vary_config(base_config, config_ranges, mode):
    """Return configurations.

    Args:
        base_config: dict, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }
        mode: str, can take 'combinatorial', 'sequential', and 'control'

    Return:
        configs: a list of config dict [config1, config2, ...]
    """
    if mode == 'combinatorial':
        _vary_config = _vary_config_combinatorial
    elif mode == 'sequential':
        _vary_config = _vary_config_sequential
    elif mode == 'control':
        _vary_config = _vary_config_control
    else:
        raise ValueError('Unknown mode {}'.format(str(mode)))
    configs, config_diffs = _vary_config(base_config, config_ranges)
    # Automatic set names for configs
    # configs = autoname(configs, config_diffs)
    for i, config in enumerate(configs):
        config.model_name = str(i).zfill(6)  # default name
    return configs


# def autoname(configs, config_diffs):
#     """Helper function for automatically naming models based on configs."""
#     new_configs = list()
#     for config, config_diff in zip(configs, config_diffs):
#         name = 'model'
#         for key, val in config_diff.items():
#             name += '_' + str(key) + str(val)
#         config['save_path'] = Path(config['save_path']) / name
#         new_configs.append(config)
#     return new_configs


def _vary_config_combinatorial(base_config, config_ranges):
    """Return combinatorial configurations.

    Args:
        base_config: dict, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }

    Return:
        configs: a list of config dict [config1, config2, ...]
            Loops over all possible combinations of hp1, hp2, ...
        config_diffs: a list of config diff from base_config
    """
    # Unravel the input index
    keys = config_ranges.keys()
    dims = [len(config_ranges[k]) for k in keys]
    n_max = int(np.prod(dims))

    configs, config_diffs = list(), list()
    for i in range(n_max):
        new_config = deepcopy(base_config)

        config_diff = dict()
        indices = np.unravel_index(i, dims=dims)
        # Set up new config
        for key, index in zip(keys, indices):
            val = config_ranges[key][index]
            setattr(new_config, key, val)
            config_diff[key] = val

        configs.append(new_config)
        config_diffs.append(config_diff)

    return configs, config_diffs


def _vary_config_sequential(base_config, config_ranges):
    """Return sequential configurations.

    Args:
        base_config: dict, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }

    Return:
        configs: a list of config dict [config1, config2, ...]
            Loops over all hyperparameters hp1, hp2 together sequentially
        config_diffs: a list of config diff from base_config
    """
    keys = config_ranges.keys()
    dims = [len(config_ranges[k]) for k in keys]
    n_max = dims[0]

    configs, config_diffs = list(), list()
    for i in range(n_max):
        new_config = deepcopy(base_config)
        config_diff = dict()
        for key in keys:
            val = config_ranges[key][i]
            setattr(new_config, key, val)
            config_diff[key] = val

        configs.append(new_config)
        config_diffs.append(config_diff)

    return configs, config_diffs


def _vary_config_control(base_config, config_ranges):
    """Return control configurations.

    Each config_range is gone through sequentially. The base_config is
    trained only once.

    Args:
        base_config: dict, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }

    Return:
        configs: a list of config dict [config1, config2, ...]
            Loops over all hyperparameters hp1, hp2 independently
        config_diffs: a list of config diff from base_config
    """
    keys = list(config_ranges.keys())
    # Remove the baseconfig value from the config_ranges
    new_config_ranges = {}
    for key, val in config_ranges.items():
        base_config_val = getattr(base_config, key)
        new_config_ranges[key] = [v for v in val if v != base_config_val]

    # Unravel the input index
    dims = [len(new_config_ranges[k]) for k in keys]
    n_max = int(np.sum(dims))

    configs, config_diffs = list(), list()
    configs.append(deepcopy(base_config))
    config_diffs.append({})

    for i in range(n_max):
        new_config = deepcopy(base_config)

        index = i
        for j, dim in enumerate(dims):
            if index >= dim:
                index -= dim
            else:
                break

        config_diff = dict()
        key = keys[j]

        val = new_config_ranges[key][index]
        setattr(new_config, key, val)
        config_diff[key] = val

        configs.append(new_config)
        config_diffs.append(config_diff)

    return configs, config_diffs


def _islikemodeldir(d):
    """Check if directory looks like a model directory."""
    try:
        files = os.listdir(d)
    except NotADirectoryError:
        return False
    fs = ['model.ckpt', 'model.pkl', 'model.pt', 'log.pkl', 'log.npz']
    for f in fs:
        if f in files:
            return True
    return False


def _get_alldirs(dir, model, sort):
    """Return sorted model directories immediately below path.

    Args:
        model: bool, if True find directories containing model files
        sort: bool, if True, sort directories by name
    """
    dirs = os.listdir(dir)
    if model:
        dirs = [d for d in dirs if _islikemodeldir(os.path.join(dir, d))]
        if _islikemodeldir(dir):  # if root is mode directory, return it
            return [dir]
    if sort:
        ixs = np.argsort([int(n) for n in dirs])  # sort by epochs
        dirs = [os.path.join(dir, dirs[n]) for n in ixs]
    return dirs


def select_modeldirs(modeldirs, select_dict=None, acc_min=None):
    """Select model directories.

    Args:
        modeldirs: list of model directories
        select_dict: dict, config must match select_dict to be selected
        acc_min: None or float, minimum validation acc to be included
    """
    new_dirs = []
    for d in modeldirs:
        selected = True
        if select_dict is not None:
            config = load_config(d)  # epoch modeldirs have no configs
            for key, val in select_dict.items():
                if key == 'data_dir':
                    print(config.data_dir)
                    print(val)
                    # If data_dir, only compare last
                    if Path(config.data_dir).name != Path(val).name:
                        selected = False
                        break
                else:
                    if getattr(config, key) != val:
                        selected = False
                        break

        if acc_min is not None:
            log = load_log(d)
            if log['val_acc'][-1] < acc_min:
                selected = False

        if selected:
            new_dirs.append(d)

    return new_dirs


def exclude_modeldirs(modeldirs, exclude_dict=None):
    """Exclude model directories."""
    new_dirs = []
    for d in modeldirs:
        excluded = False
        if exclude_dict is not None:
            config = load_config(d)  # epoch modeldirs have no configs
            for key, val in exclude_dict.items():
                if key == 'data_dir':
                    # If data_dir, only compare last
                    if Path(config.data_dir).name == Path(val).name:
                        excluded = True
                        break
                else:
                    if getattr(config, key) == val:
                        excluded = True
                        break

        if not excluded:
            new_dirs.append(d)

    return new_dirs


def sort_modeldirs(modeldirs, key):
    """Sort modeldirs by value of key."""
    val = []
    for d in modeldirs:
        config = load_config(d)
        val.append(getattr(config, key))
    ind_sort = np.argsort(val)
    modeldirs = [modeldirs[i] for i in ind_sort]
    return modeldirs


def get_modeldirs(path, select_dict=None, exclude_dict=None, acc_min=None):
    dirs = _get_alldirs(path, model=True, sort=True)
    dirs = select_modeldirs(dirs, select_dict=select_dict, acc_min=acc_min)
    dirs = exclude_modeldirs(dirs, exclude_dict=exclude_dict)
    return dirs


def get_experiment_name(model_path):
    """Get experiment name for saving."""
    if _islikemodeldir(model_path):
        config = load_config(model_path)
        experiment_name = config.experiment_name
        if experiment_name is None:
            # model_path is assumed to be experiment_name/model_name
            experiment_name = os.path.normpath(model_path).split(os.path.sep)[-2]
    else:
        # Assume this is path to experiment
        experiment_name = os.path.split(model_path)[-1]

    return experiment_name


def get_model_name(model_path):
    """Get model name for saving."""
    if _islikemodeldir(model_path):
        config = load_config(model_path)
        model_name = config.model_name
        if model_name is None:
            # model_path is assumed to be experiment_name/model_name
            model_name = os.path.split(model_path)[-1]
    else:
        # Assume this is path to experiment
        model_name = os.path.split(model_path)[-1]

    return model_name


def save_pickle(modeldir, obj, epoch=None):
    """Save model weights in numpy.

    Args:
        modeldir: str, model directory
        obj: dictionary of numpy arrays
        epoch: int or None, epoch of training
    """
    if epoch is not None:
        modeldir = os.path.join(modeldir, 'epoch', str(epoch).zfill(4))
    os.makedirs(modeldir, exist_ok=True)
    fname = os.path.join(modeldir, 'model.npz')
    np.savez_compressed(fname, **obj)


def load_pickle(modeldir):
    file_np = os.path.join(modeldir, 'model.npz')
    file_pkl = os.path.join(modeldir, 'model.pkl')
    if os.path.isfile(file_np):
        var_dict = np.load(file_np)
    else:
        with open(file_pkl, 'rb') as f:
            var_dict = pickle.load(f)
    return var_dict


def load_pickles(dir, var):
    """Load pickle by epoch in sorted order."""
    out = []
    dirs = get_modeldirs(dir)
    for i, d in enumerate(dirs):
        var_dict = load_pickle(d)
        try:
            cur_val = var_dict[var]
            out.append(cur_val)
        except:
            print(var + ' is not in directory:' + d)
    return out


def save_log(modeldir, log):
    np.savez_compressed(os.path.join(modeldir, 'log.npz'), **log)


def load_log(modeldir):
    file_np = os.path.join(modeldir, 'log.npz')
    file_pkl = os.path.join(modeldir, 'log.pkl')
    if os.path.isfile(file_np):
        log = np.load(file_np)
    else:
        with open(file_pkl, 'rb') as f:
            log = pickle.load(f)
        save_log(modeldir, log)  # resave with npz
    return log


def has_nobadkc(modeldir, bad_kc_threshold=0.2):
    """Check if model has too many bad KCs."""
    log = load_log(modeldir)
    # After training, bad KC proportion should lower 'bad_kc_threshold'
    return log['bad_KC'][-1] < bad_kc_threshold


def filter_modeldirs_badkc(modeldirs, bad_kc_threshold=0.2):
    """Filter model dirs with too many bad KCs."""
    return [d for d in modeldirs if has_nobadkc(d, bad_kc_threshold)]


def has_singlepeak(modeldir, peak_threshold=None):
    """Check if model has a single peak."""
    # TODO: Use this method throughout to replace similar methods
    log = load_log(modeldir)
    config = load_config(modeldir)
    if peak_threshold is None:
        peak_threshold = 2./config.N_PN  # heuristic

    if config.kc_prune_weak_weights:
        thres = config.kc_prune_threshold
    else:
        thres = log['thres_inferred'][-1]  # last epoch
    if len(log['lin_bins'].shape) == 1:
        bins = log['lin_bins'][:-1]
    else:
        bins = log['lin_bins'][-1, :-1]
    bin_size = bins[1] - bins[0]
    hist = log['lin_hist'][-1]  # last epoch
    # log['lin_bins'] shape (nbin+1), log['lin_hist'] shape (n_epoch, nbin)
    ind_thres = np.argsort(np.abs(bins - thres))[0]
    ind_grace = int(0.01 / bin_size)  # grace distance to start find peak
    hist_abovethres = hist[ind_thres + ind_grace:]
    ind_peak = np.argmax(hist_abovethres)
    # Value at threshold and at peak
    thres_value = hist_abovethres[0]
    peak_value = hist_abovethres[ind_peak]
    if (ind_peak + ind_grace) * bin_size <= peak_threshold or (
            peak_value < 1.3 * thres_value):
        # peak should be at least 'peak_threshold' away from threshold
        return False
    else:
        return True


def filter_modeldirs_badpeak(modeldirs, peak_threshold=None):
    """Filter model dirs without a strong second peak."""
    return [d for d in modeldirs if has_singlepeak(d, peak_threshold)]


def filter_modeldirs(modeldirs, exclude_badkc=False, exclude_badpeak=False):
    """Select model directories.

    Args:
        modeldirs: list of model directories
        exclude_badkc: bool, if True, exclude models with too many bad KCs
        exclude_badpeak: bool, if True, exclude models with bad peaks

    Return:
        modeldirs: list of filtered model directories
    """
    print('Analyzing {} model directories'.format(len(modeldirs)))
    if exclude_badkc:
        modeldirs = filter_modeldirs_badkc(modeldirs)
        print('{} remain after filtering bad kcs'.format(len(modeldirs)))
    if exclude_badpeak:
        modeldirs = filter_modeldirs_badpeak(modeldirs)
        print('{} remain after filtering bad peaks'.format(len(modeldirs)))
    return modeldirs


def load_all_results(path, select_dict=None, exclude_dict=None,
                     argLast=True, ix=None, exclude_early_models=False,
                     none_to_string=True):
    """Load results from path.

    Args:
        path: str or list, if str, root path of all models loading results from
            if list, directories of all models

    Returns:
        res: dictionary of numpy arrays, containing information from all models
    """
    if isinstance(path, str):
        dirs = get_modeldirs(path)
    else:
        dirs = path

    dirs = select_modeldirs(dirs, select_dict=select_dict)
    dirs = exclude_modeldirs(dirs, exclude_dict=exclude_dict)

    from collections import defaultdict
    res = defaultdict(list)
    for i, d in enumerate(dirs):
        log = load_log(d)
        config = load_config(d)

        n_actual_epoch = len(log['val_acc'])
        
        if exclude_early_models and n_actual_epoch < config.max_epoch:
            continue

        # Add logger values
        for key, val in log.items():
            if len(val) == n_actual_epoch:
                if argLast:
                    res[key].append(val[-1])  # store last value in log
                elif ix is not None:
                    res[key].append(val[ix])
                else:
                    res[key].append(val)
            else:
                res[key].append(val)

        k_smart_key = 'K' if config.kc_prune_weak_weights else 'K_inferred'
        if k_smart_key in res.keys():
            res['K_smart'].append(res[k_smart_key][-1])

        # Adding configuration values
        for k in dir(config):
            if k == 'coding_level':  # name conflict with log entry
                res['coding_level_set'].append(config.coding_level)
            elif k[0] != '_':
                v = getattr(config, k)
                if v is None and none_to_string:
                    v = '_none'
                res[k].append(v)

        # Add pn2kc peak information
        _singlepeak = has_singlepeak(d)
        _nobadkc = has_nobadkc(d)
        res['has_singlepeak'].append(_singlepeak)
        res['no_badkc'].append(_nobadkc)
        res['clean_pn2kc'].append(_singlepeak and _nobadkc)

    loss_keys = list()
    for key, val in res.items():
        try:
            res[key] = np.array(val)
        except ValueError:
            print('Cannot turn ' + key +
                  ' into np array, probably non-homogeneous shape')

        if 'loss' in key:
            loss_keys.append(key)

    for key in loss_keys:
        new_key = key[:-4] + 'logloss'
        try:
            res[new_key] = np.log(res[key])
        except AttributeError:
            print('''Could not compute log loss.
                  Most likely models have not finished training.''')

    return res


nicename_dict = {
    '_none': 'None',
    'ORN_NOISE_STD': 'Noise level',
    'N_PN': 'Number of PNs',
    'N_KC': 'Number of KCs',
    'N_ORN_DUPLICATION': 'ORNs per type',
    'kc_inputs': 'PN inputs per KC',
    'glo_score': 'GloScore',
    'or_glo_score': 'OR to ORN GloScore',
    'combined_glo_score': 'OR to PN GloScore',
    'train_acc': 'Training Accuracy',
    'train_loss': 'Training Loss',
    'train_logloss': 'Log Training Loss',
    'val_acc': 'Accuracy',
    'val_loss': 'Loss',
    'val_logloss': 'Log Loss',
    'epoch': 'Epoch',
    'kc_dropout': 'KC Dropout Rate',
    'kc_loss_alpha': r'$\alpha$',
    'kc_loss_beta': r'$\beta$',
    'initial_pn2kc': 'Initial PN-KC Weights',
    'initializer_pn2kc': 'Initializer',
    'mean_claw': 'Average Number of KC Claws',
    'zero_claw': '% of KC with No Input',
    'kc_out_sparse_mean': '% of Active KCs',
    'coding_level': '% of Active KCs',
    'n_trueclass': 'Number of Odor Prototypes',
    'n_trueclass_ratio': 'Odor Prototypes Per Class',
    'weight_perturb': 'Weight Perturb.',
    'lr': 'Learning rate',
    'train_kc_bias': 'Training KC bias',
    'pn_norm_pre': 'PN normalization',
    'kc_norm_pre': 'KC normalization',
    'batch_norm': 'Batch Norm',
    'kc_dropout_rate': 'KC dropout rate',
    'pn_dropout_rate': 'PN dropout rate',
    'K_inferred': 'K',
    'K': 'fixed threshold K',
    'lin_hist_': 'Distribution',
    'lin_bins_': 'PN to KC Weight',
    'lin_hist': 'Distribution',
    'lin_bins': 'PN to KC Weight',
    'kc_prune_threshold': 'KC prune threshold',
    'n_or_per_orn': 'Number of ORs per ORN',
    'K_smart': 'K',
    'kc_prune_weak_weights': 'Prune PN-KC weights',
    'kc_recinh': 'KC recurrent inhibition',
    'kc_recinh_coeff': 'KC rec. inh. strength',
    'kc_recinh_step': 'KC rec. inh. step',
    'orn_corr': 'ORN correlation',
    'w_orn': 'ORN-PN connectivity',
    'w_or': 'OR-ORN connectivity',
    'w_glo': 'PN-KC connectivity',
    'w_combined': 'OR-PN effective connectivity',
    'glo_in': 'PN Input',
    'glo': 'PN Activity',
    'kc_in': 'KC Input',
    'kc': 'KC Activity',
    'sign_constraint_orn2pn': 'Non-negative ORN-PN'
}


def nicename(name, mode='dict'):
    """Return nice name for publishing."""
    if mode in ['lr', 'meta_lr']:
        return np.format_float_scientific(name, precision=0, exp_digits=1)
    elif mode in ['N_KC', 'N_PN']:
        if name >= 1000:
            return '{:.1f}K'.format(name/1000)
        else:
            return name
    elif mode == 'kc_recinh_coeff':
        return '{:0.1f}'.format(name)
    elif mode == 'coding_level':
        return '{:0.2f}'.format(name)
    elif mode == 'n_trueclass_ratio':
        return '{:d}'.format(int(name))
    elif mode == 'data_dir':
        # Right now this is only used for pn_normalization experiment
        if Path(name).name == Path(
            './datasets/proto/concentration').name:
            return 'low'
        elif Path(name).name == Path(
            './datasets/proto/concentration_mask_row_0').name:
            return 'medium'
        elif Path(name).name == Path(
            './datasets/proto/concentration_mask_row_0.6').name:
            return 'high'
        elif name == 'data_dir':
            return 'spread'
        else:
            return '??'
    else:
        return nicename_dict.get(name, name)  # get(key, default value)


# colors from https://visme.co/blog/color-combinations/ # 14
blue = np.array([2,148,165])/255.
red = np.array([193,64,61])/255.
gray = np.array([167, 156, 147])/255.
darkblue = np.array([3, 53, 62])/255.
green = np.array([65,89,57])/255.  # From # 24


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


def _reshape_worn_by_wor(w_orn, w_or):
    ind_max = np.argmax(w_or, axis=0)
    w_orn = w_orn[ind_max,:]
    return w_orn, ind_max


def compute_glo_score(w_orn, unique_ors, mode='tile', w_or = None):
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
        unique_ors: int, the number of unique ORNs
        mode: the way w_orn is organized

    Return:
        avg_glo_score: scalar, average glomeruli score
        glo_scores: numpy array (n_pn,), all glomeruli scores
    """
    n_orn, n_pn = w_orn.shape
    if mode == 'tile' or mode == 'repeat':
        w_orn_by_pn = _reshape_worn(w_orn, unique_ors, mode)
        w_orn_by_pn = w_orn_by_pn.mean(axis=0)
    elif mode == 'matrix':
        _, ind_max = _reshape_worn_by_wor(w_orn, w_or)
        w_orn_by_pn = np.zeros((unique_ors, unique_ors))
        for i in range(unique_ors):
            out = np.mean(w_orn[ind_max == i, :], axis=0)
            out[np.isnan(out)] = 0
            w_orn_by_pn[i, :] = out
    else:
        raise ValueError('reshaping format is not recognized {}'.format(mode))

    glo_scores = list()
    for i in range(n_pn):
        w_tmp = w_orn_by_pn[:, i]  # all projections to the i-th PN neuron
        indsort = np.argsort(w_tmp)[::-1]
        w_max = w_tmp[indsort[0]]
        w_second = w_tmp[indsort[1]]
        glo_score = (w_max - w_second) / (w_max + w_second)
        glo_scores.append(glo_score)

    avg_glo_score = np.round(np.mean(glo_scores),4)
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
    from sklearn.metrics.pairwise import cosine_similarity
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


# def get_colormap():
#     def make_colormap(seq):
#         """Return a LinearSegmentedColormap
#         seq: a sequence of floats and RGB-tuples. The floats should be increasing
#         and in the interval (0,1).
#         """
#
#         seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
#         cdict = {'red': [], 'green': [], 'blue': []}
#         for i, item in enumerate(seq):
#             if isinstance(item, float):
#                 r1, g1, b1 = seq[i - 1]
#                 r2, g2, b2 = seq[i + 1]
#                 cdict['red'].append([item, r1, r2])
#                 cdict['green'].append([item, g1, g2])
#                 cdict['blue'].append([item, b1, b2])
#         return colors.LinearSegmentedColormap('CustomMap', cdict, N=512)
#
#     c = colors.ColorConverter().to_rgb
#     a = 'tomato'
#     b = 'darkred'
#     cmap = make_colormap([c('white'), c(a), .5, c(a), c(b), .8, c(b)])
#     return cmap


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval,
                                            b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap