#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Representation stability analysis."""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

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


def _get_M(n_pn, n_kc, n_kc_claw):
    M = model.get_sparse_mask(n_pn, n_kc, n_kc_claw) / n_kc_claw
    # M = model.get_sparse_mask(n_pn, n_kc, n_kc_claw)
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
                    coding_level=None, n_pts=10):
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
    
    plt.plot(n_kc_claws, diffs, 'o-')

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


def _get_p_sign_preserve(n_kc_claw):
    X, Y, Y2 = analyze_perturb(n_kc_claw=n_kc_claw, coding_level=10, n_pts=100)
    vec = np.random.randn(N_KC, 100)    
    proj = np.dot(Y, vec).flatten()
    proj2 = np.dot(Y2, vec).flatten()
    p_sign_preserve = np.mean((proj > 0) == (proj2 > 0))
    return p_sign_preserve


def get_p_sign_preserve(n_kc_claw, n_rep=1):
    ps = [_get_p_sign_preserve(n_kc_claw) for i in range(n_rep)]
    return np.mean(ps)
            

n_kc_claws = np.arange(1, 50)
p_sign_preserves = list()
for n_kc_claw in n_kc_claws:
    p_sign_preserve = get_p_sign_preserve(n_kc_claw, n_rep=1)
    p_sign_preserves.append(p_sign_preserve)
    
plt.figure()
plt.plot(n_kc_claws, p_sign_preserves, 'o-')
plt.xticks([1, 3, 7, 10, 20, 30])
plt.ylabel('P[sign preserve]')
# _easy_save('analytical', 'p_sign_perserve')
    
