#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Representation stability analysis."""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import pairwise_distances

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

# import model
# from tools import save_fig

N_PN = 50
N_KC = 2500
N_KC_CLAW = 7

def relu(x):
    return x * (x>0.)


def normalize(x):
    """Normalize along axis=1."""
    return (x.T/np.sqrt(np.sum(x**2, axis=1))).T

def _get_M(n_pn, n_kc, n_kc_claw, sign_constraint=True):
    M = model.get_sparse_mask(n_pn, n_kc, n_kc_claw) / np.sqrt(n_kc_claw)
    
    M = perturb(M, 0.5, 'multiplicative')  # pre-perturb
    
    # M = model.get_sparse_mask(n_pn, n_kc, n_kc_claw)
    if not sign_constraint:
        S = (np.random.rand(*M.shape) > 0.5)*2 - 1
        M *= S
    return M


def perturb(M, beta, mode='multiplicative'):
    if mode == 'multiplicative':
        P = np.random.uniform(1-beta, 1+beta, size=M.shape)
        return M * P
    elif mode == 'additive':
        # P = np.random.randn(*M.shape) * beta
        P = np.random.uniform(-beta, beta, size=M.shape) * np.max(M)
        return M + P * (M > 1e-6)  # only applied on connected weights
    else:
        raise ValueError('Unknown perturb mode')


def analyze_perturb(n_pn=N_PN, n_kc=N_KC, n_kc_claw=N_KC_CLAW,
                    coding_level=None, same_threshold=True, n_pts=10,
                    perturb_mode='weight', ff_inh=False, normalize_x=True,
                    n_rep=1):

    X, Y, Y2 = list(), list(), list()
    for i in range(n_rep):
        X0, Y0, Y20 = _analyze_perturb(n_pn, n_kc, n_kc_claw, coding_level,
                                    same_threshold, n_pts, perturb_mode,
                                    ff_inh, normalize_x)
        X.append(X0)
        Y.append(Y0)
        Y2.append(Y20)
    
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    Y2 = np.concatenate(Y2, axis=0)
    
    return X, Y, Y2
    

def _analyze_perturb(n_pn=N_PN, n_kc=N_KC, n_kc_claw=N_KC_CLAW,
                    coding_level=None, same_threshold=True, n_pts=10,
                    perturb_mode='weight', ff_inh=False, normalize_x=True):
    X = np.random.rand(n_pts, n_pn)
    # X = abs(np.random.randn(n_pts, n_pn))  # TODO: TEMP
    if normalize_x:
        X = normalize(X)

    M = _get_M(n_pn, n_kc, n_kc_claw)
    
    # M = np.random.uniform(0.5, 1.5, size=(n_pn, n_kc)) * (np.random.rand(n_pn, n_kc)<n_kc_claw/n_pn)

    if ff_inh:
        # TODO: think how to weight perturb with ff inh
        M = M - M.sum(axis=0).mean()
    
    Y = np.dot(X, M)

    if perturb_mode == 'weight':
        M2 = perturb(M, beta=0.1, mode='multiplicative')
        # M2 = perturb(M, beta=0.1, mode='additive')
        # M2 = _get_M(n_pn, n_kc, n_kc_claw)
        Y2 = np.dot(X, M2)
    elif perturb_mode == 'pn_activity':
        X2 = X + np.random.randn(*X.shape) * 0.1
        Y2 = np.dot(X2, M)
    else:
        raise ValueError('Unknown perturb mode: ' + str(perturb_mode))
    
    if coding_level is not None:
        threshold = np.percentile(Y.flatten(), 100-coding_level)
        Y = Y - threshold
        if not same_threshold:
            threshold = np.percentile(Y2.flatten(), 100-coding_level)
        Y2 = Y2 - threshold            
        
        Y = relu(Y)
        Y2 = relu(Y2)
    
    return X, Y, Y2


def get_perturb_diff(n_pn=N_PN, n_kc=N_KC, n_kc_claw=N_KC_CLAW):
    X, Y, Y2 = analyze_perturb(n_pn, n_kc, n_kc_claw)
    
    # Measure the difference between the representation
    diff = np.mean(abs(Y-Y2))

    return diff


def get_diff_by_n_kc_claw():
    n_kc_claws = np.arange(1, 20)
    diffs = [get_perturb_diff(n_kc_claw=n) for n in n_kc_claws]
    
    plt.figure()
    plt.plot(n_kc_claws, diffs, 'o-')
    plt.xlabel('KC claws')
    plt.ylabel('Absolute perturbation')

    return n_kc_claws, diffs


def subtract111_func(Z):
    vec111 = np.ones(2500)
    vec111 = vec111/np.sqrt(np.sum(vec111**2))
    vec111 = vec111[:, np.newaxis]
    Z = Z  - np.dot(np.dot(Z, vec111), vec111.T)
    return Z


def _compute_relative_distortion(n_kc_claw, subtract111=True, plot_fig=False):
    X, Y, Y2 = analyze_perturb(n_kc_claw=n_kc_claw, coding_level=10, n_pts=100)
    
    if subtract111:
        Y = subtract111_func(Y)
        Y2 = subtract111_func(Y2)    
        
    # Y2 = Y2/(Y2.mean()/Y.mean())
    
    # plt.figure()
    # _ = plt.hist(Y.flatten())
    
    dist = pairwise_distances(Y).flatten()
    dist2 = pairwise_distances(Y2).flatten()
    
    if n_kc_claw in [1, 3, 7, 20, 30] and plot_fig:
        plt.figure(figsize=(2, 2))
        plt.scatter(dist, dist2)
        m = np.max(dist2)
        plt.plot([0, m], [0, m])
        plt.title('KC claw: ' + str(n_kc_claw))
    
    relative_distortion = np.median(abs(dist - dist2) / (dist + dist2 + 1e-10))
    
    return relative_distortion


def compute_relative_distortion(n_kc_claw, n_rep=1):
    dists = [_compute_relative_distortion(n_kc_claw) for i in range(n_rep)]
    return np.mean(dists)


def analyze_pairwise_distance():
    
    # n_kc_claws = [7]
    n_kc_claws = np.arange(1, 40)
    relative_distortions = list()
    for n_kc_claw in n_kc_claws:
        relative_distortion = compute_relative_distortion(n_kc_claw, n_rep=1)
        relative_distortions.append(relative_distortion)
        
    plt.figure()
    plt.plot(n_kc_claws, relative_distortions, 'o-')
    plt.xticks([1, 3, 7, 10, 20, 30])
    # save_fig('analytical', 'relative_distortion')


def _get_proj(n_kc_claw, n_pts=500, n_proj=500, coding_level=100, **kwargs):
    X, Y, Y2 = analyze_perturb(
            n_kc_claw=n_kc_claw, coding_level=coding_level, n_pts=n_pts, **kwargs)
    vec = np.random.randn(N_KC, n_proj)
    b = 0
    # b = np.random.randn(n_proj)
    proj = np.dot(Y, vec) + b
    proj2 = np.dot(Y2, vec) + b
    proj, proj2 = proj.flatten(), proj2.flatten()
    return proj, proj2


def get_proj(n_kc_claw, n_rep=1, **kwargs):
    proj, proj2 = np.array([]), np.array([])
    for i in range(n_rep):
        p, p2 = _get_proj(n_kc_claw, **kwargs)
        proj = np.concatenate((p, proj))
        proj2 = np.concatenate((p2, proj2))
    return proj, proj2


def vary_kc_claw():
    """Main analysis file."""
    perturb_mode = 'weight'
    # perturb_mode = 'pn_activity'
    n_kc_claws = np.arange(1, 50, 1)
    
    projs = list()
    proj2s = list()
    for i, n_kc_claw in enumerate(n_kc_claws):    
        proj, proj2 = get_proj(n_kc_claw, n_rep=5, coding_level=10,
                               n_pn=50, perturb_mode=perturb_mode,
                               ff_inh=True)
        projs.append(proj)
        proj2s.append(proj2)
    
    names = ['projected_signal', 'projected_noise',
             'signal_noise_ratio', 'p_sign_preserve']
    
    from scipy.signal import savgol_filter
    
    x = n_kc_claws
    res = dict()
    for value_name in names:
        values = list()
        for i in range(len(n_kc_claws)):
            proj, proj2 = projs[i], proj2s[i]
            if value_name == 'p_sign_preserve':
                value = np.mean((proj > 0) == (proj2 > 0))
            elif value_name == 'projected_signal':
                value = np.std(proj)
            elif value_name == 'projected_noise':
                value = np.std(proj-proj2)
            elif value_name == 'signal_noise_ratio':
                value = (np.std(proj))/(np.std(proj-proj2))
            else:
                raise ValueError('Unknown value name')
            values.append(value)
        res[value_name] = np.array(values)
    
    for key, val in res.items():
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_axes([0.25, 0.25, 0.7, 0.7])
        ax.plot(x, val, 'o', markersize=1)
        
        if key in ['p_sign_preserve', 'signal_noise_ratio']:
            yhat = savgol_filter(val, 11, 3) # window size 51, polynomial order 3
            ax.plot(x, yhat, '-', linewidth=1)
            ax.set_title('Max at K={:d}'.format(x[np.argmax(yhat)]))
            
        if key == 'p_sign_preserve':
            ypred = 1 - 1/np.pi*np.arctan(1/res['signal_noise_ratio'])
            # ax.plot(x, ypred)
        
        # ax.set_xticks([1, 3, 7, 10, 20, 30])
        ax.set_xlabel('Number of KC claws')
        ax.set_ylabel(key)
        # save_fig('analytical', value_name+'perturb_'+perturb_mode)
    
    
def plot_proj_hist():
    n_kc_claw = 40
    X, Y, Y2 = analyze_perturb(n_kc_claw=n_kc_claw, coding_level=10, n_pts=200)
    n_proj = 200
    vec = np.random.randn(N_KC, n_proj)
    proj = np.dot(Y, vec)
    proj2 = np.dot(Y2, vec)
    proj, proj2 = proj.flatten(), proj2.flatten()
    
    for data in [proj, proj2]:
    
        mu, std = norm.fit(data)
        
        # Plot the histogram.
        plt.figure(figsize=(3, 3))
        plt.hist(data, bins=100, density=True, alpha=0.6, color='g')
        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=1)
        title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
        plt.title(title)
        
        plt.show()
    
    
    plt.figure()
    lim = np.array([[-1, 1], [-1, 1]])*0.2
    lim = None
    _ = plt.hist2d(proj, proj2, bins=70, range=lim)


def plot_proj_hist_varyclaws():
    n_kc_claws = [1, 3, 5, 7, 10, 20, 30, 40]
    projs = list()
    proj2s = list()
    for i, n_kc_claw in enumerate(n_kc_claws):    
        proj, proj2 = get_proj(n_kc_claw, n_rep=2, coding_level=10)
        projs.append(proj)
        proj2s.append(proj2)
    
    
    bins = np.linspace(-5, 5, 500)
    bin_centers = (bins[1:]+bins[:-1])/2
        
    plt.figure(figsize=(3, 3))
    colors = plt.cm.jet(np.linspace(0,1,len(n_kc_claws)))
    for i, n_kc_claw in enumerate(n_kc_claws):
        
        proj, proj2 = projs[i], proj2s[i]
        
        # mu, std = np.mean(proj), np.std(proj)    
        # proj_norm = (proj - mu)/std
        # proj2_norm = (proj2 - mu)/std
        # hist, bin_edges = np.histogram(proj_norm, density=True, bins=bins)
        # hist2, bin_edges = np.histogram(proj2_norm, density=True, bins=bins)
        
        hist, bin_edges = np.histogram(proj, density=True, bins=bins)
        # Plot the histogram.
        plt.plot(bin_centers, hist, label=str(n_kc_claw), color=colors[i])
    plt.xlim(-3, 3)
    plt.legend()
    plt.show()

    
    plt.figure(figsize=(3, 3))
    colors = plt.cm.jet(np.linspace(0,1,len(n_kc_claws)))
    for i, n_kc_claw in enumerate(n_kc_claws):
        
        proj, proj2 = projs[i], proj2s[i]
        proj_diff = proj - proj2
        hist, bin_edges = np.histogram(proj_diff, density=True, bins=bins)
        plt.plot(bin_centers, hist, label=str(n_kc_claw), color=colors[i])
    plt.xlim(-3, 3)
    plt.title('Distribution of randomly projected perturbation')
    plt.legend()
    save_fig('analytical', 'hist_pert_proj')
    

# =============================================================================
# Ks = [1, 3, 5, 7, 10, 12, 15, 20, 30, 40]
# # Ks = [40]
# # Ks = [40]
# # n_kc_claws = [7]
# from collections import defaultdict
# values = defaultdict(list)
# approxs = list()
# ground_truth = list()
# for K in Ks:
#     coding_level = 10
#     n_pts = 500
#     kwargs = {'normalize_x': False}
#     X, Y, Y2 = analyze_perturb(
#                 n_kc_claw=K, coding_level=coding_level, n_pts=n_pts, n_rep=10, **kwargs)
#     
#     norm_Y = np.linalg.norm(Y, axis=1)
#     norm_Y2 = np.linalg.norm(Y2, axis=1)
#     norm_dY = np.linalg.norm(Y2-Y, axis=1)
#     
#     cos_theta = (np.sum(Y * Y2, axis=1) / (norm_Y * norm_Y2))
#     cos_theta = cos_theta[(norm_Y * norm_Y2)>0]
#     theta = np.arccos(cos_theta)/np.pi*180
#     
#     norm_ratio = norm_dY/norm_Y
#     norm_ratio = norm_ratio[norm_Y>0]
#     
#     S = norm_Y**2
#     R = norm_dY**2
#     
#     corr = np.var(S)/np.mean(S)**2
#     mu_S = np.mean(S)
#     mu_R = np.mean(R)
#     first_term = mu_R/mu_S
#     second_term = first_term * (np.mean(S**2)/mu_S**2)
#     third_term = -np.mean(S*R)/mu_S**2
#         
#     approx = np.sqrt(first_term+second_term+third_term)
#     
# # =============================================================================
# #     plt.figure(figsize=(3, 1.0))
# #     _ = plt.hist(theta)
# #     plt.xlim([0, 180])
# #     plt.title('K: {:d} Mean Angle: {:0.2f}, norm ratio {:0.3f}'.format(
# #             K, np.mean(theta), norm_ratio.mean()))
# # =============================================================================
#     
#     print('')
#     print(K)
#     # print(np.mean(theta)/180*np.pi/np.mean(norm_ratio))
#     print(np.mean(norm_ratio))
#     print(np.sqrt(np.mean(norm_ratio**2)))
#     print(np.mean(norm_dY)/np.mean(norm_Y))
#     print(np.sqrt(np.mean(norm_dY**2)/np.mean(norm_Y**2)))
#     
#     print('Approximation')
#     # print(corr)
#     print(approx)
#     
#     values['ground_truth'].append(np.mean(norm_ratio))
#     values['approxs'].append(approx)
#     values['first_term'].append(first_term)
#     values['second_term'].append(second_term)
#     values['third_term'].append(third_term)
# =============================================================================
    
    

# =============================================================================
# plt.figure()
# kk = np.sum(Y>0, axis=1)
# kk2 = np.sum(Y2>0, axis=1)
# plt.scatter(kk, kk2)
# plt.plot([150, 350], [150, 350])
# 
# norm_Y = np.linalg.norm(Y, axis=1)
# norm_dY = np.linalg.norm(Y2 - Y, axis=1)
# =============================================================================

# =============================================================================
# plt.figure()
# plt.scatter(norm_Y, norm_dY)
# plt.xlabel('Norm Pre-perturb Y')
# plt.xlabel('Norm Y perturbation')
# =============================================================================
    
# =============================================================================
# m = 50
# c = 2
# def fun(k):
#     b = -(k/2 + c * np.sqrt(k/3-k**2/(4*m)))
#     return 3*k + 4 - 3*k/m + 12*b + 12*b**2/k
# 
# ks = np.linspace(1, 49)
# plt.plot(ks, fun(ks))
# 
# =============================================================================


# =============================================================================
# plt.figure()
# plt.plot(Ks, values['ground_truth'], label='ground truth')
# plt.plot(Ks, values['approxs'], label='approximation')
# plt.legend()
# 
# plt.figure()
# for key in ['first_term', 'second_term', 'third_term']:
#     plt.plot(Ks, values[key], label=key)
# plt.legend()
# =============================================================================


# plt.figure()
# plt.plot(Ks, mu_R/mu_S, 'o-')
