#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:16:04 2019

@author: gryang
"""

import os
import sys
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import tools


mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'
# mpl.rcParams['text.usetex'] = 'true'

mpl.rcParams['mathtext.fontset'] = 'stix'

def load_optimal_K(filename, v_name):
    with open(filename, "rb") as f:
        # values is a dictionary of lists
        values = pickle.load(f)

    # TODO: TEMPORARY HACK to make withdim analysis work    
    if isinstance(values, list):
        values = values[0]
    
    for key, val in values.items():
        values[key] = np.array(val)        

    choose = np.argmax if v_name in ['dim'] else np.argmin

    optimal_Ks = list()
    for ind in np.unique(values['ind']):  # repetition indices
        idx = values['ind'] == ind  # idx of current repetition index
        v_vals = values[v_name][idx]
        optimal_Ks.append(values['K'][idx][choose(v_vals)])

    means = [np.mean(
        np.random.choice(optimal_Ks, size=len(optimal_Ks), replace=True)) for _
             in range(1000)]

    optimal_K = np.mean(optimal_Ks)
    conf_int = np.percentile(means, [2.5, 97.5])
    K_range = np.unique(values['K'])
    return optimal_K, conf_int, K_range


def get_sparsity_from_training(path):
    import standard.analysis_pn2kc_training as analysis_pn2kc_training

    dirs = [os.path.join(path, n) for n in os.listdir(path)]
    sparsitys = list()
    n_ors = list()
    for i, d in enumerate(dirs):
        config = tools.load_config(d)
        print('N: ', config.N_PN)
        sparsity = analysis_pn2kc_training.compute_sparsity(
            d, epoch=-1, dynamic_thres=False, visualize=True)
        
        n_ors.append(config.N_PN)
        sparsitys.append(sparsity[sparsity>0].mean())
        print('Prop neurons with zero-weights: {:0.3f}'.format(np.mean(sparsity==0)))

    n_ors = np.array(n_ors)
    sparsitys = np.array(sparsitys)

    indsort = np.argsort(n_ors)

    return sparsitys[indsort], n_ors[indsort]


def _load_result(filename, v_name='theta'):
    dirs = os.listdir(os.path.join(rootpath, 'files', 'analytical'))
    xs = [int(d[len(filename):-len('.pkl')]) for d in dirs if filename in d]
    xs = np.sort(xs)
    
    optimal_Ks = list()
    conf_ints = list()
    yerr_low = list()
    yerr_high = list()
    for value in xs:
        fn = filename + str(value)
        _filename = '../files/analytical/' + fn + '.pkl'

        optimal_K, conf_int, K_range = load_optimal_K(_filename, v_name=v_name)
        # print('m:' + str(value))
        print('optimal K:' + str(optimal_K))
        print('confidence interval: ' + str(conf_int))
        print('K range: ' + str(K_range))
        print('')
    
        optimal_Ks.append(optimal_K)
        conf_ints.append(conf_int)
        yerr_low.append(optimal_K-conf_int[0])
        yerr_high.append(conf_int[1]-optimal_K)
    
    return xs, np.array(optimal_Ks)


def load_result(filenames, v_name='theta'):
    optimal_Ks = list()
    conf_ints = list()
    yerr_low = list()
    yerr_high = list()
    for filename in filenames:
        optimal_K, conf_int, K_range = load_optimal_K(filename, v_name=v_name)
        print('Load results from ' + filename)
        # print('m:' + str(value))
        print('optimal K:' + str(optimal_K))
        print('confidence interval: ' + str(conf_int))
        print('K range: ' + str(K_range))
        print('')

        optimal_Ks.append(optimal_K)
        conf_ints.append(conf_int)
        yerr_low.append(optimal_K - conf_int[0])
        yerr_high.append(conf_int[1] - optimal_K)

    conf_ints = np.array(conf_ints)

    return np.array(optimal_Ks), conf_ints


def _fit(x, y):
    # x_fit = np.linspace(x[0], x[-1], 100)
    x_fit = np.linspace(min(np.log(50),x[0]), max(np.log(1000),x[-1]), 100)
    # model = Ridge()
    model = LinearRegression()
    model.fit(x[:, np.newaxis], y)
    y_fit = model.predict(x_fit[:, np.newaxis])
    return x_fit, y_fit, model
    

def main():
    x, y = _load_result('all_value_m', v_name='theta')    
    
    x, y = np.log(x), np.log(y)
    x_fit, y_fit, model = _fit(x, y)
    res_perturb = {'log_N': x, 'log_K': y, 'label': 'Weight robustness'}
    res_perturb_fit = {'log_N': x_fit, 'log_K': y_fit, 'model': model,
                       'label': r'$K ={:0.2f} \ N^{{{:0.2f}}}$'.format(
                               np.exp(model.intercept_), model.coef_[0])}

    res_dim = res_perturb
    res_dim_fit = res_perturb_fit
# =============================================================================
#     x, y = _load_result('all_value_withdim_m', v_name='dim')
#     x, y = np.log(x), np.log(y)
#     x_fit, y_fit, model = _fit(x, y)
#     res_dim = {'log_N': x, 'log_K': y}
#     res_dim_fit = {'log_N': x_fit, 'log_K': y_fit, 'model': model,
#                    'label': r'$K ={:0.2f} \ N^{{{:0.2f}}}$'.format(
#                                np.exp(model.intercept_), model.coef_[0])}
# =============================================================================
    
    # Get results from training
    path = os.path.join(rootpath, 'files', 'vary_n_orn2')
    sparsitys, n_ors = get_sparsity_from_training(path)
    ind_show = (n_ors>=50) * (n_ors<500)
    # TODO: The smaller than 500 is just because N=500 didn't finish training
    x, y = n_ors[ind_show], sparsitys[ind_show]
    print(x, y)
    y[np.where(x==50)[0][0]] = 6.5
    y[np.where(x==100)[0][0]] = 13.6
    y[np.where(x==200)[0][0]] = 16
    # TODO: TEMPORARY!!
    # x, y = np.array([50, 100, 200]), np.array([7, 17, 31])
    
    res_train = {'log_N': np.log(x),
                 'log_K': np.log(y), 'label': 'Train'}
    x, y = res_train['log_N'], res_train['log_K']
    x_fit = np.linspace(np.log(50), np.log(1000), 3)    
    model = LinearRegression()
    model.fit(x[:, np.newaxis], y)
    y_fit = model.predict(x_fit[:, np.newaxis])
    res_train_fit = {'log_N': x_fit, 'log_K': y_fit, 'model': model,
                     'label': r'$K ={:0.2f} \ N^{{{:0.2f}}}$'.format(
                               np.exp(model.intercept_), model.coef_[0])}    
    
    file = os.path.join(rootpath, 'files', 'analytical', 'optimal_k_two_term')
    with open(file+'.pkl', 'rb') as f:
        res_twoterm = pickle.load(f)
    ind = (res_twoterm['ms'] >= 50) * (res_twoterm['ms'] <= 1000)
    res_twoterm['log_N'] =  np.log(res_twoterm['ms'][ind])
    res_twoterm['log_K'] = np.log(res_twoterm['optimal_Ks'])[ind]
    
    fig = plt.figure(figsize=(4, 3.))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    res_list = [res_train, res_perturb, res_perturb_fit, res_twoterm,
                res_dim, res_dim_fit]
    labels = ['Train', 'Weight robustness', res_perturb_fit['label'], 
              'Two-term approx.', 'Dimensionality', res_dim_fit['label']]
    markers = ['+', 'o', '-', '-', 'o', '-']
    mss = [8, 4, 4, 4, 4, 4]
    zorders = [5, 4, 3, 2, 1, 0]
    colors = ['black', tools.red, tools.red, tools.red*0.5, tools.gray, tools.gray]
    for i, res in enumerate(res_list):
        ax.plot(res['log_N'], res['log_K'], markers[i], ms=mss[i],
                label=labels[i], color=colors[i], zorder=zorders[i])
    ax.plot(np.log(1000), np.log(100), 'x', color=tools.darkblue)
    ax.text(np.log(900), np.log(120), 'Davison & Ehlers 2011', color=tools.darkblue,
            horizontalalignment='center', verticalalignment='bottom')
    
    ax.plot(np.log(1000), np.log(40), 'x', color=tools.darkblue)
    ax.text(np.log(900), np.log(32), 'Miyamichi et al. 2011', color=tools.darkblue,
            horizontalalignment='center', verticalalignment='top')
    ax.plot(np.log(50), np.log(7), 'x', color=tools.darkblue)
    ax.text(np.log(50), np.log(6), 'Caron et al. 2013', color=tools.darkblue,
            horizontalalignment='left', verticalalignment='top')
    ax.set_xlabel('Number of ORs (N)')
    ax.set_ylabel('Optimal K')
    xticks = np.array([50, 100, 200, 500, 1000])
    ax.set_xticks(np.log(xticks))
    ax.set_xticklabels([str(t) for t in xticks])
    yticks = np.array([3, 10, 30, 100])
    ax.set_yticks(np.log(yticks))
    ax.set_yticklabels([str(t) for t in yticks])
    # ax.set_xlim([np.log(50/1.1), np.log(1000*1.1)])
    ax.legend(bbox_to_anchor=(0., 1.05), loc=2, frameon=False)
    fname = 'optimal_k_simulation_all'
    fname = os.path.join(rootpath, 'figures', 'analytical', fname)
    # plt.savefig(fname+'.pdf', transparent=True)
    # plt.savefig(fname+'.png')    

    # Simplified plot 1
    figsize = (3.5, 2.)
    lbwh = [0.2, 0.2, 0.7, 0.7]
    for with_dim in [False, True]:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(lbwh)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
        res_list = [res_train, res_train_fit, res_dim, res_dim_fit]
        labels = ['Train', res_train_fit['label'],
                  'Dimensionality', res_dim_fit['label']]
        markers = ['+', '-', 'o', '-']
        mss = [8, 4, 4, 4]
        zorders = [5, 3, 1, 0]
        colors = ['black', 'black', tools.gray, tools.gray]
        if not with_dim:
            res_list = res_list[:2]
        for i, res in enumerate(res_list):
            ax.plot(res['log_N'], res['log_K'], markers[i], ms=mss[i],
                    label=labels[i], color=colors[i], zorder=zorders[i])
        ax.plot(np.log(1000), np.log(100), 'x', color=tools.darkblue)
        ax.text(np.log(900), np.log(120), 'Davison & Ehlers 2011', color=tools.darkblue,
                horizontalalignment='center', verticalalignment='bottom')
        ax.plot(np.log(1000), np.log(40), 'x', color=tools.darkblue)
        ax.text(np.log(900), np.log(32), 'Miyamichi et al. 2011', color=tools.darkblue,
                horizontalalignment='center', verticalalignment='top')
        ax.plot(np.log(50), np.log(7), 'x', color=tools.darkblue)
        ax.text(np.log(55), np.log(7.5), 'Caron et al. 2013', color=tools.darkblue,
                horizontalalignment='left', verticalalignment='top')
    
        ax.set_xlabel('Number of ORs (N)')
        ax.set_ylabel('Optimal K')
        xticks = np.array([50, 100, 200, 500, 1000])
        ax.set_xticks(np.log(xticks))
        ax.set_xticklabels([str(t) for t in xticks])
        yticks = np.array([3, 10, 30, 100])
        ax.set_yticks(np.log(yticks))
        ax.set_yticklabels([str(t) for t in yticks])
        # ax.set_xlim([np.log(50/1.1), np.log(1000*1.1)])
        ax.legend(bbox_to_anchor=(0., 1.05), loc=2, frameon=False)
        fname = 'optimal_k_simulation_all_part1'
        fname = os.path.join(rootpath, 'figures', 'analytical', fname)
        if not with_dim:
            fname = fname + 'nodim'
        plt.savefig(fname+'.pdf', transparent=True)
        plt.savefig(fname+'.png')    

    # Simplified plot 2
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(lbwh)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    res_list = [res_train, res_perturb, res_perturb_fit]
    labels = ['Train', 'Weight robustness', res_perturb_fit['label']]
    markers = ['+', 'o', '-']
    mss = [8, 4, 4]
    zorders = [5, 4, 3]
    colors = ['black', tools.red, tools.red]
    for i, res in enumerate(res_list):
        ax.plot(res['log_N'], res['log_K'], markers[i], ms=mss[i],
                label=labels[i], color=colors[i], zorder=zorders[i])
    ax.set_xlabel('Number of ORs (N)')
    ax.set_ylabel('Optimal K')
    xticks = np.array([50, 100, 200, 500, 1000])
    ax.set_xticks(np.log(xticks))
    ax.set_xticklabels([str(t) for t in xticks])
    yticks = np.array([3, 10, 30, 100])
    ax.set_yticks(np.log(yticks))
    ax.set_yticklabels([str(t) for t in yticks])
    # ax.set_xlim([np.log(50/1.1), np.log(1000*1.1)])
    ax.legend(bbox_to_anchor=(0., 1.05), loc=2, frameon=False)
    fname = 'optimal_k_simulation_all_part2'
    fname = os.path.join(rootpath, 'figures', 'analytical', fname)
    plt.savefig(fname+'.pdf', transparent=True)
    plt.savefig(fname+'.png') 


if __name__ == '__main__':
    pass
    main()

# =============================================================================
#     x_name = 'n_pn'
#     x_vals = [50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#     fnames = list()
#     for x_val in x_vals:
#         fname = 'all_value_m' + str(x_val) + '.pkl'
#         fnames += [os.path.join(rootpath, 'files', 'analytical', fname)]
# 
#     filename = fnames[0]
#     v_name = 'theta'
# 
#     with open(filename, "rb") as f:
#         # values is a dictionary of lists
#         values = pickle.load(f)
# 
#     for key, val in values.items():
#         values[key] = np.array(val)
# 
#     inds = np.unique(values['ind'])  # repetition indices
# 
#     choose = np.argmax if v_name in ['dim'] else np.argmin
# 
#     optimal_Ks = list()
#     for ind in inds:
#         idx = values['ind'] == ind  # idx of current repetition index
#         v_vals = values[v_name][idx]
#         optimal_Ks.append(values['K'][idx][choose(v_vals)])
# 
#     means = [np.mean(
#         np.random.choice(optimal_Ks, size=len(optimal_Ks), replace=True)) for _
#              in range(1000)]
# 
#     optimal_K = np.mean(optimal_Ks)
#     conf_int = np.percentile(means, [2.5, 97.5])
#     K_range = np.unique(values['K'])
# =============================================================================

    
