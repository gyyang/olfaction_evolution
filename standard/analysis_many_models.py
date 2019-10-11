#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 13:29:30 2019

@author: robert_yang
"""

"""Specifically analyze results from vary_lr_n_kc experiments"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import tools
from tools import nicename
from standard.analysis import _easy_save
from standard.analysis_pn2kc_training import plot_all_K


mpl.rcParams['font.size'] = 10
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

"""
Previous results
[ 50 100 150 200 300 400]
[ 7.90428212 10.8857362  16.20759494 20.70314843 27.50305499 32.03561644]"""



# if __name__ == '__main__':
def analyze_single_net(n_orn=200, foldername='vary_prune_pn2kc_init'):
    # n_orn = 200
    

    path = os.path.join(rootpath, 'files', foldername, foldername+str(n_orn))
    res = tools.load_all_results(path, argLast=False, ix=slice(0, 50))
    res['bins'] = (res['lin_bins'][0][:-1] + res['lin_bins'][0][1:])/2
    res['n_orn'] = res['N_ORN'][0]
    # res = _get_K(res)   # TODO: something wrong with this
    
    res['density'] = res['lin_hist'] / res['lin_hist'].sum(axis=-1, keepdims=True)
    res['cum_density'] = np.cumsum(res['density'], axis=-1)
    res['density_sm'] = savgol_filter(res['density'], 51, 3, axis=-1) 
    
    # Recompute K if pruning
    if res['kc_prune_weak_weights'][0]:
        n_net, n_epoch = res['density'].shape[:2]
        
        # Recompute bad KC
        bad_KC = np.zeros((n_net, n_epoch))
        for i in range(n_net):
            for j in range(n_epoch):
                bad_KC[i, j] = np.mean(res['kc_w_sum'][i, j] <1e-9)
        
        net_excludesecondpeak = list()
        peaks = list()
        Ks = list()
        for i in range(n_net):
            t = res['kc_prune_threshold'][i]
            K = res['density'][i][:, res['bins']>t].sum(axis=-1)*n_orn
            K /= 1-bad_KC[i] # Do not correct for this yet
            Ks.append(K)
            
            peak = np.argmax(res['density'][i][-1, res['bins']>t])
            net_excludesecondpeak.append(peak>10)
            peaks.append(peak)
        res['K'] = np.array(Ks)  
        res['net_excludesecondpeak'] = np.array(net_excludesecondpeak)
    
    print('Learning rate')
    print(res['lr'])
    
    res['net_excludelowinit'] = res['initial_pn2kc']>res['initial_pn2kc'].min()
    res['net_excludebadkc'] = res['bad_KC'][:, -1]<0.1
    res['net_excludelowacc'] = res['val_acc'][:, -1] > 0.4
    
    return res


def plot_single_net(res): 
    plt.figure()
    _ = plt.plot(res['K'][:, 3:].T)
    
    net_plot = (res['net_excludebadkc'] * res['net_excludelowacc'] *
                res['net_excludesecondpeak'])
    # net_plot = np.array([True]*res['K'].shape[0])
    # Plot by different learning rate
    epoch_start = 3
    epoch_end = -1
    x_plot = np.arange(res['K'].shape[1])
    K_plot = res['K'][net_plot, :]
    coding_level_plot = res['coding_level'][net_plot, :]
    n_kc_plot = res['N_KC'][net_plot]
    acc_plot = res['val_acc'][net_plot, :]
    logK_plot = np.log(K_plot)
    lr = res['lr'][net_plot]
    show_label = [True] + [lr[i] != lr[i-1] for i in range(1, len(lr))]
    lr_color = np.log(lr/lr.min())
    lr_color /= lr_color.max()
    
    cmap = mpl.cm.get_cmap('cool')
    for var_plot, ylabel in zip([K_plot, coding_level_plot, acc_plot],
                                ['K', 'Activity density', 'Acc']):
        
        plt.figure()
        for i in range(len(lr)):
            label = '{:0.1E}'.format(lr[i]) if show_label[i] else None
            linestyle = '-' if n_kc_plot[i] == 2500 else '--'
            plt.plot(x_plot[epoch_start:epoch_end],
                     var_plot[i, epoch_start:epoch_end],
                     linestyle, color=cmap(lr_color[i]), label=label)
        plt.title('N={:d}'.format(res['n_orn']))
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend(title='LR')
        
    plt.figure()
    plt.plot(res['bins'], res['density'][net_plot, -1][0])
    plt.ylim([0, 0.003])
    
    plt.figure()
    plt.plot(res['bins'], res['density'][net_plot, -1][-1])
    plt.ylim([0, 0.003])
    
    epoch_plot = 49
    
    thres_plot = res['thres'][net_plot, epoch_plot]
    K_plot = res['K'][net_plot, epoch_plot]
    lr = res['lr'][net_plot]
    lr_color = np.log(lr/lr.min())
    lr_color /= lr_color.max()
    
    plt.figure()
    for i in range(len(thres_plot)):
        plt.scatter(thres_plot[i], K_plot[i],
                    color=cmap(lr_color[i]))
    plt.title('N={:d} Epoch={:d}'.format(res['n_orn'], epoch_plot+1))
    plt.xlabel('Threshold')
    plt.ylabel('K')
    
    lr = res['lr'][net_plot]
    K = res['K'][net_plot]
    KC = res['N_KC'][net_plot]
    val_acc = res['val_acc'][net_plot]
    net_maxlr = lr == np.max(lr)
    # net_maxlr = lr == 1e-3
    plt.figure()
    plt.scatter(np.log(KC[net_maxlr]), K[net_maxlr, 10])
    plt.title('N={:d}'.format(res['n_orn']))
    plt.xlabel('Log(N_KC)')
    plt.ylabel('K')

    epoch_start = 2    
    plt.figure()
    _ = plt.plot(x_plot[epoch_start:],
                 val_acc[net_maxlr].mean(axis=0)[epoch_start:])
    plt.xlabel('Epoch')
    plt.ylabel('Val acc')


def analyze_all_nets(foldername = 'vary_prune_pn2kc_init'):
    path = os.path.join(rootpath, 'files', foldername)
    files = os.listdir(path)
    files = [f for f in files if f[:len(foldername)]==foldername]
    n_orns = [int(f[len(foldername):]) for f in files]
    n_orns = np.sort(n_orns)

    res_all = dict()
    for n_orn, file in zip(n_orns, files):
        path = os.path.join(rootpath, 'files', foldername, file)
        assert os.path.exists(path)
    
        res = analyze_single_net(n_orn, foldername)        
        res_all[n_orn] = res
    return n_orns, res_all


def plot_all_nets(n_orns, res_all):
    """Plot results from all networks.
    
    Args:
        n_orns: list of N_PN
        res_all: a dictionary of results. Each entry is itself a dictionary
        of logged items. Each log is organized as (n_models, n_epochs, ...) or
        (n_models) if the value is fixed across epochs
    """
    Ks = list()
    epoch_plots = list()
    for n_orn in n_orns:
        res = res_all[n_orn]
        net_plot = (res['net_excludebadkc'] * res['net_excludelowacc'] *
                    res['net_excludesecondpeak'])
        lr = res['lr'][net_plot]
        K = res['K'][net_plot]
        N_KC = res['N_KC'][net_plot]
        val_acc = res['val_acc'][net_plot]
        net_maxlr = lr == np.max(lr)
        
        mean_val_acc = val_acc[net_maxlr].mean(axis=0)
        epoch_plot = np.argmax(mean_val_acc)
        
        plt.figure()
        _ = plt.plot(val_acc[net_maxlr][:, 1:].T)
        plt.title('N={:d}, LR={:0.1E}'.format(n_orn, np.max(lr)))
            
        epoch_plots.append(epoch_plot)
        Ks.append(K[net_maxlr, epoch_plot])
        print('N={:d}'.format(n_orn))
        print('Epoch used {:d}'.format(epoch_plot))
        
    new_Ks = np.array([K for K in Ks if len(K)>0])
    new_n_orns = np.array([n for n, K in zip(n_orns, Ks) if len(K)>0])
    plot_all_K(new_n_orns, new_Ks, plot_box=True)
    # plt.title('lr {:0.1E}, Epoch {:d}'.format(lr_plot, epoch_plot+1))
    # plt.title('Highest LR, Epoch {:d}'.format(epoch_plot+1))
    plt.title('Highest LR, Epoch with highest acc')


if __name__ == '__main__':
    # foldername = 'vary_pn2kc_init'
    # foldername = 'vary_prune_pn2kc_init'
    # foldername = 'vary_init_sparse_lr'
    
    # res = analyze_single_net(n_orn=100, foldername='vary_prune_pn2kc_init')
    # plot_single_net(res)
    
    n_orns, res_all = analyze_all_nets(foldername = 'vary_prune_pn2kc_init')
    plot_all_nets(n_orns, res_all)


    

