"""Specifically analyze results from vary_lr_n_kc experiments"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import tools
from tools import nicename
from standard.analysis import _easy_save


rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)


def main(path):
    res = tools.load_all_results(path, argLast=False)

    n_model, n_epoch = res['sparsity'].shape
    Ks = np.zeros((n_model, n_epoch))
    for i in range(n_model):
        for j in range(n_epoch):
            sparsity = res['sparsity'][i, j]
            Ks[i, j] = sparsity[sparsity>0].mean()
    res['K'] = Ks

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
        plt.figure(figsize=(2, 2))
        plt.imshow(v, origin='lower')
        plt.colorbar()
        plt.title(nicename(vname))
        plt.xlabel(nicename('N_KC'))
        plt.ylabel(nicename('lr'))
        plt.xticks(np.arange(len(x_val)), [str(t) for t in x_val])
        plt.yticks(np.arange(len(y_val)), ['{:.1e}'.format(t) for t in y_val])
        _easy_save(path, vname)


if __name__ == '__main__':
    foldername = 'vary_lr_n_kc_n_orn50'
    path = os.path.join(rootpath, 'files', foldername)
    main(path)