"""Specifically analyze results from vary_lr_n_kc experiments"""

import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import tools
from tools import nicename
from standard.analysis import _easy_save


"""
Previous results
[ 50 100 150 200 300 400]
[ 7.90428212 10.8857362  16.20759494 20.70314843 27.50305499 32.03561644]"""


def move_helper():
    """Temporary function to move new results into old directory."""
    from shutil import copytree, rmtree
    n_orns = [50, 75, 100, 150, 200, 300, 400, 500, 800, 1000]
    for n_orn in n_orns:
        foldername = 'vary_new_lr_n_kc_n_orn' + str(n_orn)
        path = os.path.join(rootpath, 'files', foldername)
        newfoldername = 'vary_lr_n_kc_n_orn' + str(n_orn)
        newpath = os.path.join(rootpath, 'files', newfoldername)
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
    n_model, n_epoch = res['sparsity'].shape
    Ks = np.zeros((n_model, n_epoch))
    for i in range(n_model):
        for j in range(n_epoch):
            sparsity = res['sparsity'][i, j]
            Ks[i, j] = sparsity[sparsity>0].mean()
    res['K'] = Ks
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
        
        _easy_save(path, vname)


def get_all_K(acc_threshold = 0.5, exclude_start = 5):
    """Get all K from training.
    
    Args:
        acc_threshold: threshold for excluding failed networks
        exclude_start: exclude the beginning number of epochs
        
    Returns:
        n_orns: list of N_ORN values
        Ks: list of arrays, each array is K from many networks
    """
    
    n_orns = [50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000]
    # n_orns = [50]
    Ks = list()
    for n_orn in n_orns:
        foldername = 'vary_lr_n_kc_n_orn' + str(n_orn)
        path = os.path.join(rootpath, 'files', foldername)
        res = tools.load_all_results(path, argLast=False)
        res = _get_K(res)
        
        K = res['K']
        acc = res['val_acc']
        if n_orn == 50:
            ind = res['N_KC'] == 2500
            K = K[ind, :]  # only take N_KC=2500
            acc = acc[ind, :]
        if exclude_start:
            K = K[:, exclude_start:]  # after a number of epochs
            acc = acc[:, exclude_start:]
        
        if acc_threshold:
            # Examine only points with accuracy above threshold
            ind_acc = (acc > acc_threshold).flatten()
            K = K.flatten()
            K = K[ind_acc]
            
        K = K.flatten()
        # remove extreme values
        K = K[~np.isnan(K)]
        K = K[2<K]
        K = K[K<n_orn*0.9]
        Ks.append(K)
    
    return n_orns, Ks


def plot_all_K(n_orns, Ks, plot_scatter=False,
               plot_box=False, plot_data=True,
               plot_fit=True, plot_angle=False):
    
    fig = plt.figure(figsize=(3.5, 2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    
    logKs = [np.log(K) for K in Ks]
    med_logKs = np.array([np.median(np.log(K)) for K in Ks])
    
    if plot_scatter:
        for n_orn, K, med_logK in zip(n_orns, Ks, med_logKs):
            ax.scatter(np.log([n_orn]*len(K)), np.log(K), alpha=0.01, s=3)
            ax.plot(np.log(n_orn), med_logK, '+', ms=15, color='black')
    
    if plot_box:
        ax.boxplot(logKs, positions=np.log(n_orns), widths=0.1,
                   flierprops={'markersize': 3})
    
    if plot_fit:
        # x, y = np.log(n_orns)[3:], med_logKs[3:]
        x, y = np.log(n_orns), med_logKs
        x_fit = np.linspace(np.log(50), np.log(1600), 3)    
        model = LinearRegression()
        model.fit(x[:, np.newaxis], y)
        y_fit = model.predict(x_fit[:, np.newaxis])
        ax.plot(x_fit, y_fit)
    
    if plot_data:
        ax.plot(np.log(1000), np.log(100), 'x', color=tools.darkblue)
        ax.text(np.log(900), np.log(120), '[1]', color=tools.darkblue,
                horizontalalignment='center', verticalalignment='bottom')
        
        ax.plot(np.log(1000), np.log(40), 'x', color=tools.darkblue)
        ax.text(np.log(900), np.log(32), '[2]', color=tools.darkblue,
                horizontalalignment='center', verticalalignment='top')
        ax.plot(np.log(50), np.log(7), 'x', color=tools.darkblue)
        ax.text(np.log(50), np.log(6), '[3]', color=tools.darkblue,
                horizontalalignment='left', verticalalignment='top')
    
    if plot_angle:
        fname = os.path.join(rootpath, 'files', 'analytical',
                                 'control_coding_level_summary')
        summary = pickle.load(open(fname, "rb"))
        # summary: 'opt_ks', 'coding_levels', 'conf_ints', 'n_orns'
        ax.boxplot(list(np.log(summary['opt_ks'].T)),
                   positions=np.log(summary['n_orns']),
                   widths=0.1, flierprops={'markersize': 3})
        
    x = [ 50, 100, 150, 200, 300, 400]
    y = [ 7.90428212, 10.8857362,  16.20759494,
         20.70314843, 27.50305499, 32.03561644]
    # ax.plot(np.log(x), np.log(y))
    ax.set_xlabel('Number of ORs (N)')
    ax.set_ylabel('Optimal K')
    xticks = np.array([50, 100, 200, 400, 1000, 1600])
    ax.set_xticks(np.log(xticks))
    ax.set_xticklabels([str(t) for t in xticks])
    yticks = np.array([3, 10, 30, 100])
    ax.set_yticks(np.log(yticks))
    ax.set_yticklabels([str(t) for t in yticks])
    ax.set_xlim(np.log([40, 1700]))
    
    name = 'opt_k'
    if plot_scatter:
        name += '_scatter'
    if plot_box:
        name += '_box'
    if plot_data:
        name += '_data'
    if plot_fit:
        name += '_fit'
    if plot_angle:
        name += 'angle'
    _easy_save('vary_lr_n_kc', name)


def main():
    n_orns, Ks = get_all_K()
    plot_all_K(n_orns, Ks, plot_box=True)
    plot_all_K(n_orns, Ks, plot_angle=True)


if __name__ == '__main__':
    # move_helper()
# =============================================================================
#     foldername = 'vary_lr_n_kc_n_orn1000'
#     path = os.path.join(rootpath, 'files', foldername)
#     plot2d(path)
# =============================================================================
    main()
    

            