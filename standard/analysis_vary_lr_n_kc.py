"""Specifically analyze results from vary_lr_n_kc experiments"""

import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import matplotlib as mpl
from scipy.signal import savgol_filter

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import tools
from tools import nicename
from tools import save_fig
from standard.analysis_pn2kc_training import plot_all_K

mpl.rcParams['font.size'] = 10
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

"""
Previous results
[ 50 100 150 200 300 400]
[ 7.90428212 10.8857362  16.20759494 20.70314843 27.50305499 32.03561644]"""


def move_helper():
    """Temporary function to move new results into old directory."""
    from shutil import copytree, rmtree
    n_orns = [25, 50, 75, 100, 125, 150, 175, 200, 300, 400, 500, 800, 1000]
    for n_orn in n_orns:
        foldername = 'new_vary_prune_pn2kc_init' + str(n_orn)
        path = os.path.join(rootpath, 'files', 'vary_prune_pn2kc_init',
                            foldername)
        newfoldername = 'vary_prune_pn2kc_init' + str(n_orn)
        newpath = os.path.join(rootpath, 'files', 'vary_prune_pn2kc_init',
                               newfoldername)
        if os.path.exists(path):
            files = os.listdir(path)
            
            for file in files:
                if file[:2] == '00':
                    newfile = os.path.join(path, '10' + file[2:])
                    if os.path.exists(newfile):
                        rmtree(newfile)
                    os.rename(os.path.join(path, file), newfile)

            files = os.listdir(path)
            print(files)
            for file in files:
                assert file[:2] == '10'
                newfile = os.path.join(newpath, file)
                if os.path.exists(newfile):
                    rmtree(newfile)
                copytree(os.path.join(path, file), newfile)


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
    return res


def plot2d(path):
    res = tools.load_all_results(path, argLast=False)
    res = _get_K(res)

    xname = 'N_KC'
    yname = 'lr'
    x_val = res[xname][:4]  # specialized code
    y_val = res[yname][::4]

    for vname in ['K', 'lr', 'val_acc', 'N_KC']:
        v = res[vname]
        if len(v.shape) == 2:
            if vname in ['K']:
                v = np.nanmin(v, axis=1)
            else:
                v = v[:, -1]
        v = np.reshape(v, (len(y_val), len(x_val)))

        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_axes([0.3, 0.3, 0.5, 0.5])
        im = ax.imshow(v, origin='lower')

        plt.figure(figsize=(2, 2))
        plt.imshow(v, origin='lower')
        plt.colorbar()
        plt.title(nicename(vname))
        plt.xlabel(nicename('N_KC'))
        plt.ylabel(nicename('lr'))
        plt.xticks(np.arange(len(x_val)), [str(t) for t in x_val])
        plt.yticks(np.arange(len(y_val)), ['{:.1e}'.format(t) for t in y_val])

        ax = fig.add_axes([0.82, 0.3, 0.04, 0.5])
        cb = plt.colorbar(im, cax=ax)
        cb.outline.set_linewidth(0.5)
        
        save_fig(path, vname)


def get_all_K(acc_threshold = 0.75, exclude_start = 48, experiment_folder = 'default'):
    """Get all K from training.
    
    Args:
        acc_threshold: threshold for excluding failed networks
        exclude_start: exclude the beginning number of epochs
        
    Returns:
        n_orns: list of N_ORN values
        Ks: list of arrays, each array is K from many networks
    """
    
    # n_orn_tmps = [50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000]
    n_orn_tmps = [25, 50, 75, 100, 125, 150, 175, 200]
    # n_orns = [200, 500]
    n_orns = list()

    Ks = list()
    badKCs = list()
    for n_orn in n_orn_tmps:
        # TODO: use listdir
        foldername = experiment_folder + str(n_orn)
        path = os.path.join(rootpath, 'files', experiment_folder, foldername)
        if not os.path.exists(path):
            continue
        n_orns.append(n_orn)
        res = tools.load_all_results(path, argLast=False)
        res = _get_K(res)
        
        K = res['K']
        badKC = res['bad_KC']
        acc = res['val_acc']
        if n_orn == 50:
            ind = res['N_KC'] == 2500
            K = K[ind, :]  # only take N_KC=2500
            badKC = badKC[ind, :]
            acc = acc[ind, :]
        if exclude_start:
            K = K[:, exclude_start:]  # after a number of epochs
            badKC = badKC[:, exclude_start:]
            acc = acc[:, exclude_start:]
        
        if acc_threshold:
            # Examine only points with accuracy above threshold
            ind_acc = (acc > acc_threshold).flatten()
            K = K.flatten()
            K = K[ind_acc]
            badKC = badKC.flatten()
            badKC = badKC[ind_acc]
            
        # TODO: Temporary override
        K = res['K'][8, 30:50]
            
        K = K.flatten()
        # remove extreme values
        K = K[~np.isnan(K)]
        K = K[2<K]
        K = K[K<n_orn*0.9]
        Ks.append(K)

        badKC = badKC.flatten()
        badKCs.append(badKC)

    return n_orns, Ks, badKCs


def plot_fraction_badKC(n_orns, data, plot_scatter=False, plot_box=True, path='default'):
    fig = plt.figure(figsize=(3.5, 2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    medians = np.array([np.median(K) for K in data])

    if plot_scatter:
        for n_orn, K, median in zip(n_orns, data, medians):
            ax.scatter(np.log([n_orn] * len(K)), K, alpha=0.01, s=3)
            ax.plot(np.log(n_orn), median, '+', ms=15, color='black')

    if plot_box:
        ax.boxplot(data, positions=np.log(n_orns), widths=0.1,
                   flierprops={'markersize': 3})

    ax.set_xlabel('Number of ORs (N)')
    ax.set_ylabel('Fraction KCs with no input')
    xticks = np.array([50, 100, 200, 400, 1000, 1600])
    ax.set_xticks(np.log(xticks))
    ax.set_xticklabels([str(t) for t in xticks])
    yticks = np.array([0, .5, 1])
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(t) for t in yticks])
    ax.set_xlim(np.log([40, 1700]))
    ax.set_ylim([-.05, 1.05])

    name = 'frac_badKC'
    if plot_scatter:
        name += '_scatter'
    if plot_box:
        name += '_box'
    save_fig(path, name)


def main(experiment_folder):
    n_orns, Ks, badKCs = get_all_K(experiment_folder=experiment_folder)
    plot_all_K(n_orns, Ks, plot_box=True, path=experiment_folder)
    plot_fraction_badKC(n_orns, badKCs, path=experiment_folder)
    # plot_all_K(n_orns, Ks, plot_angle=True, path=experiment_folder)


def get_consensus_K(foldername):
    path = os.path.join(rootpath, 'files', foldername)
    
    dirs = os.listdir(path)
    n_orns = np.sort([int(d[len(foldername):]) for d in dirs])
    # n_orns = [50, 100, 200]
    Ks = list()
    for n_orn in n_orns:
        d = os.path.join(path, foldername+str(n_orn))
        K = _get_consensus_K(d)
        Ks.append(K)
    
    return n_orns, Ks
    
    
def _get_consensus_K(path, plot=False):
    """Get consensus K.
    
    This function is work in progress. The threshold is determined
    by the location that gives least difference across networks.
    """
    
    res = tools.load_all_results(path, argLast=False)
    res = _get_K(res)
    res['density'] = res['hist'] / res['hist'].sum(axis=-1, keepdims=True)
    res['cum_density'] = np.cumsum(res['density'], axis=-1)
    
    n_orn = res['N_ORN'][0]
    
# =============================================================================
#     acc_ind = res['val_acc'][:, -1] > 0.2
#     _ = plt.plot(res['K'][acc_ind, 10:100].T)
#     
#     plt.scatter(np.log(res['thres'][:, -1]), res['K'][:, -1])
#     plt.xlim([-5, 0])
#     plt.ylim([0, 30])
# =============================================================================
        
    bins_center = (res['w_bins'][0][:-1] + res['w_bins'][0][1:])/2
    
    acc_ind = res['val_acc'][:, -1] > 0.75
    nobadkc_ind = [np.std(s)/np.mean(s) for s in res['kc_w_sum'][:, -1]]
    nobadkc_ind = np.array(nobadkc_ind) < 0.1
    ind = acc_ind * nobadkc_ind
    
    density = res['density'][ind, -1]
    cum_density = res['cum_density'][ind, -1]
    std_cum = np.std(cum_density, axis=0)
    mean_cum = np.mean(cum_density, axis=0)
    xind = bins_center > -5
    
    minima = argrelextrema(std_cum[xind], np.less)[0]
    # minima = argrelextrema((std_cum/(1-mean_cum))[xind], np.less)[0]
    print(minima)
    minima = minima[0]
    
    K = (1 - cum_density[:, xind][:, minima]) * n_orn
    
    print('K', K)
    
    if plot:
        plt.figure()
        _ = plt.plot(bins_center, density.T)
        
        plt.figure()
        _ = plt.plot(bins_center[xind], cum_density[:, xind].T)
        
        plt.figure()
        plt.plot(bins_center[xind], std_cum[xind])
        
        plt.plot([bins_center[xind][minima]]*2, [0, std_cum[xind].max()])
    
    return K
  
    
def temp():
    plot = True
    foldername = 'vary_new_lr_n_kc_n_orn'
    path = os.path.join(rootpath, 'files', foldername)
    
    dirs = os.listdir(path)
    n_orns = np.sort([int(d[len(foldername):]) for d in dirs if foldername in d])
    Ks = list()
    n_orn = 75
    path = os.path.join(path, foldername+str(n_orn))
    
    res = tools.load_all_results(path, argLast=False)
    res = _get_K(res)   # TODO: something wrong with this
    res['density'] = res['lin_hist'] / res['lin_hist'].sum(axis=-1, keepdims=True)
    res['cum_density'] = np.cumsum(res['density'], axis=-1)
    
    res['density_sm'] = savgol_filter(res['density'], 51, 3, axis=-1)
    
    n_orn = res['N_ORN'][0]
    
# =============================================================================
#     acc_ind = res['val_acc'][:, -1] > 0.2
#     _ = plt.plot(res['K'][acc_ind, 10:100].T)
#     
#     plt.scatter(np.log(res['thres'][:, -1]), res['K'][:, -1])
#     plt.xlim([-5, 0])
#     plt.ylim([0, 30])
# =============================================================================
        
    bins = (res['lin_bins'][0][:-1] + res['lin_bins'][0][1:])/2
    
    acc_ind = res['val_acc'][:, -1] > 0.5
    nobadkc_ind = [np.std(s)/np.mean(s) for s in res['kc_w_sum'][:, -1]]
    nobadkc_ind = np.array(nobadkc_ind) < 0.1
    ind = acc_ind * nobadkc_ind
    # ind = acc_ind
    
    density = res['density'][ind, -1]
    cum_density = res['cum_density'][ind, -1]
    
# =============================================================================
#     _ = plt.plot(bins, res['density_sm'][8:, -1][:5].T)
#     plt.ylim([0, 0.05])
# =============================================================================
    
    import matplotlib as mpl
    cmap = mpl.cm.get_cmap('cool')

# =============================================================================
#     epochs = [10, 100, 200, 299]
#     for i in epochs:
#         _ = plt.plot(bins, res['density_sm'][8, i],
#                      color=cmap(i/np.max(epochs)))
#     plt.ylim([0, 0.0006])
# =============================================================================

# =============================================================================
#     for i in range(len(res['density_sm'])):
#         plt.figure()
#         _ = plt.plot(bins, res['density_sm'][i, -1].T)
#         plt.ylim([0, 0.003])
#         plt.title(str(i) + '_' + str(res['lr'][i]) + '_' + str(res['N_KC'][i]) + '_' + str(res['val_acc'][i, -1]))
# =============================================================================
        
    
    # TODO: this still needs work
    # Select separable distribution
    ind_epoch = 30
    ind_sep = np.argmax(res['density_sm'][:, ind_epoch, bins>0.1], axis=-1) + 100
    
    net_valid = ind_sep>150
    net_valid = net_valid * acc_ind
    den = res['density_sm'][net_valid, ind_epoch]
    ind_sep = ind_sep[net_valid]
    ind_min = list()
    for i, d in enumerate(den):
        ind_min.append(20+np.argmin(d[20:ind_sep[i]]))  # the 100 is from bins>0.1
    
    infered_ks = res['K'][net_valid, ind_epoch]
    ks = list()
    for i, d in enumerate(den):
        k = np.sum(d[ind_min[i]:]) * n_orn
        ks.append(k)
        
        plt.figure(figsize=(2, 1.5))
        plt.plot(bins, d)
        plt.plot([bins[ind_sep[i]]]*2, [0, 0.003])
        plt.plot([bins[ind_min[i]]]*2, [0, 0.003])
        plt.ylim([0, 0.003])
        plt.title('K: {:0.2f}, {:0.2f}'.format(k, infered_ks[i]))
        # plt.title(str(i) + '_' + str(res['lr'][i]) + '_' + str(res['N_KC'][i]) + '_' + str(res['val_acc'][i, -1]))
    

# =============================================================================
#     std_cum = np.std(cum_density, axis=0)
#     mean_cum = np.mean(cum_density, axis=0)
#     xind = bins > -5
#     
#     minima = argrelextrema(std_cum[xind], np.less)[0]
#     # minima = argrelextrema((std_cum/(1-mean_cum))[xind], np.less)[0]
#     print(minima)
#     minima = minima[0]
#     
#     K = (1 - cum_density[:, xind][:, minima]) * n_orn
#     
#     print('K', K)
#     
#     if plot:
#         plt.figure()
#         _ = plt.plot(bins, density.T)
#         
#         plt.figure()
#         _ = plt.plot(bins[xind], cum_density[:, xind].T)
#         
#         plt.figure()
#         plt.plot(bins[xind], std_cum[xind])
#         
#         plt.plot([bins[xind][minima]]*2, [0, std_cum[xind].max()])
# =============================================================================

# =============================================================================
#     from scipy.signal import savgol_filter
# 
#     file = os.path.join(rootpath, 'files', 'tmp_train', 'log.pkl')
#     with open(file, 'rb') as f:
#         res = pickle.load(f)
#         
#     bins = (res['lin_bins'][:-1]+res['lin_bins'][1:])/2
#     lin_hist = np.array(res['lin_hist'])
#     
#     import matplotlib
# 
#     cmap = matplotlib.cm.get_cmap('cool')
#     
#     smooth_hist = savgol_filter(lin_hist, 21, 3, axis=1)
#     n_epoch = lin_hist.shape[0]
#     for i in range(1, n_epoch):
#         _ = plt.plot(bins, smooth_hist[i], color=cmap(i/n_epoch))
#     plt.ylim(0, 100)
# =============================================================================
    
if __name__ == '__main__':
    move_helper()
# =============================================================================
#     n_orn = 200
#     foldername = 'vary_new_lr_n_kc_n_orn'
#     # foldername = 'vary_init_sparse_lr'
#     path = os.path.join(rootpath, 'files', foldername, foldername+str(n_orn))
# 
#     res = tools.load_all_results(path, argLast=False)
#     res = _get_K(res)   # TODO: something wrong with this
#     res['density'] = res['lin_hist'] / res['lin_hist'].sum(axis=-1, keepdims=True)
#     res['cum_density'] = np.cumsum(res['density'], axis=-1)
#     res['density_sm'] = savgol_filter(res['density'], 51, 3, axis=-1)
#     bins = (res['lin_bins'][0][:-1] + res['lin_bins'][0][1:])/2
#     n_orn = res['N_ORN'][0]
# =============================================================================
    
# =============================================================================
#     net_excludebadkc = res['bad_KC'][:, -1]<0.1
#     net_n_kc = res['N_KC'] == 5000
#     net_plot = net_excludebadkc * net_n_kc
# 
#     plt.figure()
#     _ = plt.plot(res['K'][net_excludebadkc, 3:].T)
#     
#     plt.figure()
#     _ = plt.plot(res['K'][net_plot, 3:].T)
# =============================================================================
