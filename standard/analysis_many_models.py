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


mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

"""
Previous results
[ 50 100 150 200 300 400]
[ 7.90428212 10.8857362  16.20759494 20.70314843 27.50305499 32.03561644]"""


def expand_res(res, epoch_focus=-1):
    """Expand results."""
    res['bins'] = (res['lin_bins'][0][:-1] + res['lin_bins'][0][1:])/2
    res['n_orn'] = res['N_ORN'][0]
    n_orn = res['n_orn']
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
        res['bad_KC'] = bad_KC  # rewrite the one computed during training
        
        net_excludesecondpeak = list()
        peaks = list()
        Ks = list()
        for i in range(n_net):
            t = res['kc_prune_threshold'][i]
            K = res['density'][i][:, res['bins']>t].sum(axis=-1)*n_orn
            K /= 1-bad_KC[i] # Do not correct for this yet
            Ks.append(K)
            
            peak = np.argmax(res['density'][i][epoch_focus, res['bins']>t])
            net_excludesecondpeak.append(peak>50)
            peaks.append(peak)
        res['K'] = np.array(Ks)  
        res['net_excludesecondpeak'] = np.array(net_excludesecondpeak)
        res['peaks'] = np.array(peaks)
    
    print('Learning rate range')
    print(np.unique(res['lr']))
    
    res['net_excludelowinit'] = res['initial_pn2kc']>res['initial_pn2kc'].min()
    res['net_excludebadkc'] = res['bad_KC'][:, epoch_focus]<0.1
    res['net_excludelowacc'] = res['val_acc'][:, epoch_focus] > 0.5
    
    # Use N_KC closest to N_ORN**2
    unique_nkc = np.unique(res['N_KC'])
    closest_nkc = np.argmin(np.abs(np.log(n_orn**2)-np.log(unique_nkc)))
    closest_nkc = unique_nkc[closest_nkc]
    res['net_useclosestnkc'] = res['N_KC'] == closest_nkc
    res['closest_nkc'] = closest_nkc
    return res


def analyze_single_net(n_orn=200, foldername='vary_prune_pn2kc_init'):
    # Find common prefix
    base = os.path.join(rootpath, 'files', foldername)
    prefix = os.path.commonprefix(os.listdir(base))
    path = os.path.join(base, prefix+str(n_orn))
    res = tools.load_all_results(path, argLast=False, ix=slice(0, 50))

    return res


def analyze_all_nets(foldername = 'vary_prune_pn2kc_init'):
    base = os.path.join(rootpath, 'files', foldername)
    prefix = os.path.commonprefix(os.listdir(base))
    files = os.listdir(base)
    files = [f for f in files if f[:len(prefix)]==prefix]
    n_orns = [int(f[len(prefix):]) for f in files]
    n_orns = np.sort(n_orns)

    res_all = dict()
    for n_orn, file in zip(n_orns, files):
        path = os.path.join(base, file)
        assert os.path.exists(path)
    
        res = analyze_single_net(n_orn, foldername)        
        res_all[n_orn] = res
    return n_orns, res_all


def plot_single_net(res): 
    res = expand_res(res)
    
    plt.figure()
    _ = plt.plot(res['K'][:, 3:].T)
    
    net_plot = (res['net_excludebadkc'] * res['net_excludelowacc'] *
                res['net_excludesecondpeak'] * res['net_useclosestnkc'])
    
    # net_plot = np.array([True]*res['K'].shape[0])
    # Plot by different learning rate
    epoch_start = 2
    epoch_end = -1
    x_plot = np.arange(res['K'].shape[1])
    K_plot = res['K'][net_plot, :]
    kc_prune_threshold_plot = res['kc_prune_threshold'][net_plot]
    coding_level_plot = res['coding_level'][net_plot, :]
    n_kc_plot = res['N_KC'][net_plot]
    acc_plot = res['val_acc'][net_plot, :]
    trainloss_plot = res['train_loss'][net_plot, :]
    valloss_plot = res['val_logloss'][net_plot, :]
    logK_plot = np.log(K_plot)
    lr = res['lr'][net_plot]
    show_label = [True] + [lr[i] != lr[i-1] for i in range(1, len(lr))]
    lr_color = np.log(lr/lr.min())
    lr_color /= lr_color.max()
    
    cmap = mpl.cm.get_cmap('cool')
    for var_plot, ylabel in zip([K_plot, coding_level_plot, acc_plot, valloss_plot, trainloss_plot],
                                ['K', 'Activity density', 'Acc', 'Log Loss', 'Log Train Loss']):
        
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
    
    for i_plot in [0, -1]:
        plt.figure()
        plt.plot(res['bins'], res['density'][net_plot, 10][i_plot])
        plt.ylim([0, 0.003])
        plt.title('LR {:0.1E} Prune Thrs {:0.1E}'.format(
                lr[i_plot], kc_prune_threshold_plot[i_plot]))
    
    epoch_plot = -1
    
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


def plot_all_nets(n_orns, res_all, lr_criterion='max', epoch_name=5):
    """Plot results from all networks.
    
    Args:
        n_orns: list of N_PN
        res_all: a dictionary of results. Each entry is itself a dictionary
        of logged items. Each log is organized as (n_models, n_epochs, ...) or
        (n_models) if the value is fixed across epochs
    """
    Ks = list()
    epoch_plots = list()
    lr_useds = list()
    for n_orn in n_orns:
        if n_orn >= 800:
            continue

        res = res_all[n_orn]
        res = expand_res(res)
        net_plot = (res['net_excludebadkc'] * res['net_excludelowacc'] *
                    res['net_excludesecondpeak'] * res['net_useclosestnkc'])

        lr = res['lr'][net_plot]
        K = res['K'][net_plot]
        N_KC = res['N_KC'][net_plot]
        val_acc = res['val_acc'][net_plot]
        kc_prune_threshold = res['kc_prune_threshold'][net_plot]
        if lr_criterion == 'max':
            lr_used = np.max(lr)
        elif lr_criterion == 'min':
            lr_used = np.min(lr)
        else:
            lr_used = lr_criterion
        net_lr_used = lr == lr_used
        # print(np.min(lr))
        
        mean_val_acc = val_acc[net_lr_used].mean(axis=0)  
        if epoch_name == 'max_acc':
            epoch_plot = np.argmax(mean_val_acc)
        else:
            epoch_plot = epoch_name
        
        print('')
        print('N_ORN ', str(n_orn))
        print('LR used', str(lr_used), ' out of ', np.unique(res['lr']))
        print('Epoch used', str(epoch_plot))
        print('Acc', val_acc[net_lr_used][:, epoch_plot])
        print('N_KC', N_KC[net_lr_used])
        print('K', K[net_lr_used, epoch_plot])
        print('Prune threshold', kc_prune_threshold[net_lr_used], ' out of ',
              np.unique(res['kc_prune_threshold']))
        
        # plt.figure()
        # plt.plot(mean_val_acc[2:])
        
# =============================================================================
#         plt.figure()
#         _ = plt.plot(val_acc[net_maxlr][:, 1:].T)
#         plt.title('N={:d}, LR={:0.1E}'.format(n_orn, np.max(lr)))
# =============================================================================
            
        epoch_plots.append(epoch_plot)
        Ks.append(K[net_lr_used, epoch_plot])
        lr_useds.append(lr_used)
        # print('N={:d}'.format(n_orn))
        # print('Epoch used {:d}'.format(epoch_plot))
        
    new_Ks = np.array([K for K in Ks if len(K)>0])
    new_n_orns = np.array([n for n, K in zip(n_orns, Ks) if len(K)>0])
    plot_all_K(new_n_orns, new_Ks, plot_box=True, path='manymodel')
    plt.title(str(lr_criterion)+' LR, ' + str(epoch_name) + ' Epoch')
    
    plot_all_K(new_n_orns, new_Ks, plot_box=True, plot_angle=True, plot_dim=False,
               path='manymodel')


if __name__ == '__main__':
    # foldername = 'vary_pn2kc_init'
    # foldername = 'vary_prune_pn2kc_init'
    # foldername = 'vary_init_sparse_lr'
    
    # res = analyze_single_net(n_orn=200, foldername='vary_prune_lr_old')
    # plot_single_net(res)
    
    # n_orns, res_all = analyze_all_nets(foldername = 'vary_prune_lr')
    plot_all_nets(n_orns, res_all, epoch_name=10)


    

