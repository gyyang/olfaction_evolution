import matplotlib.pyplot as plt
import pickle
import numpy as np
import tools
from pylab import *
import matplotlib.animation as animation
import matplotlib as mpl
import numpy as np
import os
import glob
import standard.analysis as sa
import tools
import matplotlib.pyplot as plt
import task
from dict_methods import *
from scipy.signal import savgol_filter

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

def _get_K(res):
    n_model, n_epoch = res['sparsity'].shape[:2]
    Ks = np.zeros((n_model, n_epoch))
    bad_KC = np.zeros((n_model, n_epoch))
    for i in range(n_model):
        for j in range(n_epoch):
            sparsity = res['sparsity'][i, j]
            Ks[i, j] = sparsity[sparsity>0].mean()
            bad_KC[i,j] = np.sum(sparsity==0)/sparsity.size
    res['K'] = Ks
    res['bad_KC'] = bad_KC

def _filter_peak(res):
    peak_inds = np.zeros_like(res['kc_prune_threshold']).astype(np.bool)
    for i, thres in enumerate(res['kc_prune_threshold']):
        x = np.where(res['lin_bins'][0,:-1] > res['kc_prune_threshold'][i])[0][0]
        peak = np.argmax(res['lin_hist'][i, -1, (x-10):])
        if peak > 20:
            peak_inds[i] = True
        else:
            peak_inds[i] = False
    return peak_inds

def _filter_accuracy(res, threshold = 0.5):
    acc_ind = res['train_acc'][:, -1] > threshold
    return acc_ind

def _filter_badkc(res, threshold = 0.2):
    badkc_ind = res['bad_KC'][:, -1] < threshold
    return badkc_ind

def do_everything(path, filter_peaks = False, redo=False):
    d = os.path.join(path)
    files = glob.glob(d)
    res = defaultdict(list)
    for f in files:
        temp = tools.load_all_results(f, argLast = False)
        chain_defaultdicts(res, temp)

    if redo:
        wglos = tools.load_pickle(path, 'w_glo')
        for i, wglo in enumerate(wglos):
            w = wglo.flatten()
            hist, bins = np.histogram(w, bins=1000, range=[0, 3])
            res['lin_bins'][i] = bins
            res['lin_hist'][i][-1,:] = hist #hack

        _get_K(res)

    badkc_ind = _filter_badkc(res)
    acc_ind = _filter_accuracy(res)
    if filter_peaks:
        peak_ind = _filter_peak(res)
    else:
        peak_ind = np.ones_like(acc_ind)
    ind = badkc_ind * acc_ind * peak_ind
    for k, v in res.items():
        res[k] = v[ind]

    for k in res['lin_bins']:
        res['lin_bins_'].append(k[:-1])
    for k in res['lin_hist']:
        res['lin_hist_'].append(savgol_filter(k[-1], window_length=21, polyorder=0))
    for k, v in res.items():
        res[k] = np.array(res[k])
    return res