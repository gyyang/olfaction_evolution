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

import model
from standard.analysis import _easy_save

N_PN = 50
N_KC = 2500
N_KC_CLAW = 7

def relu(x):
    return x * (x>0.)


def normalize(x):
    """Normalize along axis=1."""
    return (x.T/np.sqrt(np.sum(x**2, axis=1))).T


def _get_M(n_pn, n_kc, n_kc_claw, sign_constraint=True):
    M = model.get_sparse_mask(n_pn, n_kc, n_kc_claw) / n_kc_claw
    
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
        P = np.random.randn(*M.shape) * beta
        return M + P
    else:
        raise ValueError('Unknown perturb mode')


def analyze_perturb(n_pn=N_PN, n_kc=N_KC, n_kc_claw=N_KC_CLAW,
                    coding_level=None, same_threshold=True, n_pts=10):
    X = np.random.rand(n_pts, 50)
    
    X = normalize(X)

    M = _get_M(n_pn, n_kc, n_kc_claw)
    Y = np.dot(X, M)

    M2 = perturb(M, beta=0.5, mode='multiplicative')
    # M2 = perturb(M, beta=0.2, mode='additive')
    # M2 = _get_M(n_pn, n_kc, n_kc_claw)
    Y2 = np.dot(X, M2)
    
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
    # _easy_save('analytical', 'relative_distortion')


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

 

def analyze_p_sign_preserve():
    n_kc_claws = np.arange(1, 50)
    
    projs = list()
    proj2s = list()
    for i, n_kc_claw in enumerate(n_kc_claws):    
        proj, proj2 = get_proj(n_kc_claw, n_rep=2, coding_level=10)
        projs.append(proj)
        proj2s.append(proj2)
    
    value_name = 'p_sign_preserve'
    
    values = list()
    for i in range(len(n_kc_claws)):
        proj, proj2 = projs[i], proj2s[i]
        if value_name == 'p_sign_preserve':
            value = np.mean((proj > 0) == (proj2 > 0))
        else:
            raise ValueError('Unknown value name')
        values.append(value)
        
    plt.figure(figsize=(3, 3))
    plt.plot(n_kc_claws, values, 'o-')
    plt.xticks([1, 3, 7, 10, 20, 30])
    plt.ylabel(value_name)
    # _easy_save('analytical', 'p_sign_perserve')
    
    
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
    _easy_save('analytical', 'hist_pert_proj')


# proj, proj2 = _get_proj(n_kc_claw=7, n_pts=500, n_proj=1, coding_level=10)