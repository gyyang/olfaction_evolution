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
import standard.analysis_pn2kc_training as analysis_pn2kc_training


mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

def get_optimal_K(m, get_dim=False):
    if get_dim:
        fn = 'all_value_withdim_m' + str(m)
    else:
        fn = 'all_value_m' + str(m)
    with open('../files/analytical/'+fn+'.pkl', "rb") as f:
        all_values = pickle.load(f)

    if get_dim:
        v_name = 'dim'
        optimal_Ks = [v['K'][np.argmax(v[v_name])] for v in all_values]
    else:
        # v_name = 'E[norm_dY/norm_Y]'
        v_name = 'theta'
        optimal_Ks = [v['K'][np.argmin(v[v_name])] for v in all_values]
    
    means = [np.mean(np.random.choice(optimal_Ks, size=len(optimal_Ks), replace=True)) for _ in range(1000)]
    
    optimal_K = np.mean(optimal_Ks)
    conf_int = np.percentile(means, [2.5, 97.5])
    K_range = all_values[0]['K']
    return optimal_K, conf_int, K_range


def get_sparsity_from_training(path):
    dirs = [os.path.join(path, n) for n in os.listdir(path)]
    sparsitys = list()
    n_ors = list()
    for i, d in enumerate(dirs):
        config = tools.load_config(d)
        print(config.N_PN)
        sparsity = analysis_pn2kc_training.compute_sparsity(
            d, epoch=-1, dynamic_thres=False, visualize=True)
        
        n_ors.append(config.N_PN)
        sparsitys.append(sparsity[sparsity>0].mean())

    n_ors = np.array(n_ors)
    sparsitys = np.array(sparsitys)

    indsort = np.argsort(n_ors)

    return sparsitys[indsort], n_ors[indsort]


def _load_result(get_dim=False):
    dirs = os.listdir(os.path.join(rootpath, 'files', 'analytical'))
    if get_dim:
        ms = [int(d[len('all_value_withdim_m'):-len('.pkl')])
              for d in dirs if 'all_value_withdim_m' in d]
    else:
        ms = [int(d[len('all_value_m'):-len('.pkl')])
              for d in dirs if 'all_value_m' in d]
    ms = np.sort(ms)
    
    # ms = np.array([50, 150, 1000])
    optimal_Ks = list()
    conf_ints = list()
    yerr_low = list()
    yerr_high = list()
    for m in ms:
        optimal_K, conf_int, K_range = get_optimal_K(m, get_dim=get_dim)
        print('m:' + str(m))
        print('optimal K:' + str(optimal_K))
        print('confidence interval: ' + str(conf_int))
        print('K range: ' + str(K_range))
        print('')
    
        optimal_K = np.log(optimal_K)
        conf_int = np.log(conf_int)
    
        optimal_Ks.append(optimal_K)
        conf_ints.append(conf_int)
        yerr_low.append(optimal_K-conf_int[0])
        yerr_high.append(conf_int[1]-optimal_K)
    
    x = np.log(ms)
    y = optimal_Ks
    
    x_fit = np.linspace(x[0], x[-1], 100)
    
    # model = Ridge()
    model = LinearRegression()
    model.fit(x[:, np.newaxis], y)
    y_fit = model.predict(x_fit[:, np.newaxis])
    
    return x, y, x_fit, y_fit, model
    

def main():
    x, y, x_fit, y_fit, model = _load_result(get_dim=False)
    res_perturb = {'log_N': x, 'log_K': y, 'label': 'Weight robustness'}
    res_perturb_fit = {'log_N': x_fit, 'log_K': y_fit, 'model': model,
                       'label': 'log K = {:0.3f}log N + {:0.3f}'.format(
                               model.coef_[0], model.intercept_)}

    x, y, x_fit, y_fit, model = _load_result(get_dim=True)
    res_dim = {'log_N': x, 'log_K': y}
    res_dim_fit = {'log_N': x_fit, 'log_K': y_fit, 'model': model,
                   'label': 'log K = {:0.3f}log N + {:0.3f}'.format(
                               model.coef_[0], model.intercept_)}
    
    # Get results from training
    path = os.path.join(rootpath, 'files', 'vary_n_orn2')
    sparsitys, n_ors = get_sparsity_from_training(path)
    ind_show = (n_ors>=50) * (n_ors<500)
    # TODO: The smaller than 500 is just because N=500 didn't finish training
    res_train = {'log_N': np.log(n_ors[ind_show]),
                 'log_K': np.log(sparsitys[ind_show]), 'label': 'Train'}        
    
    file = os.path.join(rootpath, 'files', 'analytical', 'optimal_k_two_term')
    with open(file+'.pkl', 'rb') as f:
        res_twoterm = pickle.load(f)
    ind = (res_twoterm['ms'] >= 50) * (res_twoterm['ms'] <= 1000)
    res_twoterm['log_N'] =  np.log(res_twoterm['ms'][ind])
    res_twoterm['log_K'] = np.log(res_twoterm['optimal_Ks'])[ind]
    
    # colors from https://visme.co/blog/color-combinations/ # 14
    color_blue = np.array([2,148,165])/255.
    color_red = np.array([193,64,61])/255.
    fig = plt.figure(figsize=(4, 3.))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    fit_txt = 'log K = {:0.3f}log N + {:0.3f}'.format(model.coef_[0], model.intercept_)
    # ax.scatter(x, y, marker='o', label='Prediction', s=8, color=color_red, zorder=1)
    # ax.plot(x, y, 'o', ms=2, label='Prediction', color=color_red, zorder=1)
    res_list = [res_train, res_perturb, res_perturb_fit, res_twoterm, res_dim, res_dim_fit]
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
    ax.text(np.log(1000), np.log(120), 'Davison & Ehlers 2011', color=tools.darkblue,
            horizontalalignment='center', verticalalignment='bottom')
    
    ax.plot(np.log(1000), np.log(40), 'x', color=tools.darkblue)
    ax.text(np.log(1000), np.log(32), 'Miyamichi et al. 2011', color=tools.darkblue,
            horizontalalignment='center', verticalalignment='top')
    ax.plot(np.log(50), np.log(7), 'x', color=tools.darkblue)
    ax.text(np.log(50), np.log(6), 'Caron et al. 2013', color=tools.darkblue,
            horizontalalignment='left', verticalalignment='top')
    # ax.plot(x_plot, y_plot, label=fit_txt, color=color_red, zorder=1.5)
    # ax.plot(x_plot, x_plot*0.5, '--', label='log K = 0.5log N', color='gray', zorder=2)
    
# =============================================================================
#     ind = (res_twoterm['ms'] >= 50) * (res_twoterm['ms'] <= 1000)
#     ax.plot(np.log(res_twoterm['ms'][ind]),
#             np.log(np.array(res_twoterm['optimal_Ks'])[ind]),
#             label='Two term approx', color='gray', zorder=1.6)
# =============================================================================
    
    # ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt='o', label='simulation', markersize=3)
    
    # ax.scatter(x_training, y_training, marker='+', label='Training', s=40,
    #            color=color_blue, zorder=3)
    ax.set_xlabel('Number of ORs (N)')
    ax.set_ylabel('Optimal K')
    xticks = np.array([50, 100, 200, 500, 1000])
    ax.set_xticks(np.log(xticks))
    ax.set_xticklabels([str(t) for t in xticks])
    yticks = np.array([10, 30, 100])
    ax.set_yticks(np.log(yticks))
    ax.set_yticklabels([str(t) for t in yticks])
    # ax.set_xlim([np.log(50/1.1), np.log(1000*1.1)])
    ax.legend(bbox_to_anchor=(0., 1.05), loc=2, frameon=False)
    fname = 'optimal_k_simulation_all'
    fname = os.path.join(rootpath, 'figures', 'analytical', fname)
    plt.savefig(fname+'.pdf', transparent=True)
    plt.savefig(fname+'.png')
    
if __name__ == '__main__':
    main()
    



