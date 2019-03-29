#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Representation stability analysis."""

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

N_PN = 50
N_KC = 2500
N_KC_CLAW = 7

def relu(x):
    return x * (x>0.)

def get_sparse_mask(nx, ny, non, complex=False, nOR=50):
    """Generate a binary mask.

    The mask will be of size (nx, ny)
    For all the nx connections to each 1 of the ny units, only non connections are 1.

    If complex == True, KCs cannot receive the connections from the same OR from duplicated ORN inputs.
    Assumed to be 'repeat' style duplication.

    Args:
        nx: int
        ny: int
        non: int, must not be larger than nx

    Return:
        mask: numpy array (nx, ny)
    """
    mask = np.zeros((nx, ny))
    mask[:non] = 1
    for i in range(ny):
        np.random.shuffle(mask[:, i])  # shuffling in-place
    return mask.astype(np.float32)

def normalize(x):
    """Normalize along axis=1."""
    return (x.T/np.sqrt(np.sum(x**2, axis=1))).T

def _get_M(n_pn, n_kc, n_kc_claw, sign_constraint=True,
           mode='exact', normalize=True):
    if mode == 'exact':
        M = get_sparse_mask(n_pn, n_kc, n_kc_claw)
    elif mode == 'bernoulli':
        M = (np.random.rand(n_pn, n_kc) < (n_kc_claw/n_pn))
    else:
        raise NotImplementedError
    
    if normalize:
        M = M / np.sqrt(n_kc_claw)
    # M = get_sparse_mask(n_pn, n_kc, n_kc_claw)
    # M = (np.random.rand(n_pn, n_kc) < (n_kc_claw/n_pn)) / np.sqrt(n_kc_claw)
    # M = (np.random.rand(n_pn, n_kc) < (n_kc_claw/n_pn))
    
    M = perturb(M, 0.5, 'multiplicative')  # pre-perturb
    
    # M = model.get_sparse_mask(n_pn, n_kc, n_kc_claw)
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


def _analyze_perturb(n_pn=N_PN, n_kc=N_KC, n_kc_claw=N_KC_CLAW,
                    coding_level=None, same_threshold=True, n_pts=10,
                    perturb_mode='multiplicative', ff_inh=False, normalize_x=True,
                    x_dist='uniform', w_mode='exact', normalize_w=True,
                    b_mode='percentile', use_relu=True):
    if x_dist == 'uniform':
        X = np.random.rand(n_pts, n_pn)
    elif x_dist == 'gaussian':
        X = abs(np.random.randn(n_pts, n_pn))  # TODO: TEMP
    else:
        raise NotImplementedError()

    if normalize_x:
        X = normalize(X)

    M = _get_M(n_pn, n_kc, n_kc_claw, mode=w_mode, normalize=normalize_w)
    
    # M = np.random.uniform(0.5, 1.5, size=(n_pn, n_kc)) * (np.random.rand(n_pn, n_kc)<n_kc_claw/n_pn)

    if ff_inh:
        # TODO: think how to weight perturb with ff inh
        M = M - M.sum(axis=0).mean()
    
    Y = np.dot(X, M)

    M2 = perturb(M, beta=0.01, mode=perturb_mode)
    # M2 = perturb(M, beta=0.1, mode='additive')
    # M2 = _get_M(n_pn, n_kc, n_kc_claw)
    Y2 = np.dot(X, M2)
    
    if coding_level is not None:
        if b_mode == 'percentile':
            b = -np.percentile(Y.flatten(), 100-coding_level)
        elif b_mode == 'gaussian':
            b = -(np.mean(Y) + np.std(Y)*1)
        else:
            raise NotImplementedError()
        Y = Y + b
        if not same_threshold:
            b = -np.percentile(Y2.flatten(), 100-coding_level)
        Y2 = Y2 + b
        
        if use_relu:
            Y = relu(Y)
            Y2 = relu(Y2)
    
    dY = Y2 - Y
    
    return X, Y, Y2, dY


def analyze_perturb(n_kc_claw=N_KC_CLAW, n_rep=1, **kwargs):
    X, Y, Y2, dY = list(), list(), list(), list()
    for i in range(n_rep):
        X0, Y0, Y20, dY0 = _analyze_perturb(n_kc_claw=n_kc_claw, **kwargs)
        X.append(X0)
        Y.append(Y0)
        Y2.append(Y20)
        dY.append(dY0)
    
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    Y2 = np.concatenate(Y2, axis=0)
    dY = np.concatenate(dY, axis=0)
    
    return X, Y, Y2, dY


def simulation(K, **kwargs):
    X, Y, Y2, dY = analyze_perturb(n_kc_claw=K, **kwargs)

    norm_Y = np.linalg.norm(Y, axis=1)
    norm_Y2 = np.linalg.norm(Y2, axis=1)
    norm_dY = np.linalg.norm(dY, axis=1)
    
    cos_theta = (np.sum(Y * Y2, axis=1) / (norm_Y * norm_Y2))
    cos_theta = cos_theta[(norm_Y * norm_Y2)>0]
    theta = np.arccos(cos_theta)/np.pi*180
    
    norm_ratio = norm_dY/norm_Y
    norm_ratio = norm_ratio[norm_Y>0]
    
    S = norm_Y**2
    R = norm_dY**2
    
    corr = np.var(S)/np.mean(S)**2
    mu_S = np.mean(S)
    mu_R = np.mean(R)
    first_term = mu_R/mu_S
    second_term = first_term * (np.mean(S**2)/mu_S**2)
    third_term = np.mean(S*R)/mu_S**2
        
    approx = np.sqrt(first_term+second_term-third_term)
    
    res = dict()
    res['K'] = K
    res['E[norm_dY/norm_Y]'] = np.mean(norm_ratio)
    
    res['mu_S'] = mu_S
    res['mu_R'] = mu_R
    res['theta'] = np.mean(theta)

    res['Three-term approx'] = approx
    res['mu_R/mu_S'] = first_term
    res['E[S^2]mu_R/mu_S^3'] = second_term
    res['E[SR]/mu_S^2'] = third_term
    return res


def analytical(K):
    m = 50
    k = K
    c = 1
    
    b = -(k/2+c*np.sqrt(k/3-k**2/4/m))
    
    c1 = 1/48*k**3/m*(3*m*k+8*m-2*k)
    c2 = 1/6*b*k**2/m*(3*k*m+4*m-k)
    c3 = b**2/6*k*(9*k+4-k/m)
    c4 = 2*b**3*k
    c5 = b**4
    
    d1 = 1/48*k**2/m*(4*k*m+4*m+k)
    d2 = 1/3*b*k**2
    d3 = 1/3*b**2*k
    
    mu_S = k*(1+c**2)*(4-3*k/m)/12
    mu_R = k/3
    # e_s2_tmp = c**2*k**2*(9*c**2*k**2 - 24*c**2*k*m + 16*c**2*m**2 + 6*k**2 - 32*k*m + 32*m**2)/(144*m**2)
    e_s2 = c1+c2+c3+c4+c5
    # e_sr_tmp = (12 - 12*c**2/m + 3*k/m + 16*c**2)/144*k**2
    e_sr = d1 + d2 + d3
    
    # e_sr = k**2/144*(12-12*c**2/m+3*k/m+16*c**2)
    
    first_term = mu_R/mu_S
    second_term = e_s2*mu_R/(mu_S**3)
    third_term = e_sr/mu_S**2
    
    approx = np.sqrt(first_term + second_term - third_term)
    
    res = dict()
    res['K'] = k
    res['mu_S'] = mu_S
    res['mu_R'] = mu_R
    
    res['Three-term approx'] = approx
    res['mu_R/mu_S'] = first_term
    res['E[S^2]mu_R/mu_S^3'] = second_term
    res['E[SR]/mu_S^2'] = third_term
    return res


K_sim = np.array([1, 3, 5, 7, 10, 12, 15, 20, 30, 40])
kwargs = {'x_dist': 'uniform',
          'normalize_x': False,
          # 'w_mode': 'exact',
          'w_mode': 'bernoulli',
          'normalize_w': False,
          'b_mode': 'percentile',
          # 'b_mode': 'gaussian',
          'coding_level': 10,
          'use_relu': True,
          # 'use_relu': False,
          # 'perturb_mode': 'additive',
          'perturb_mode': 'additive',
          'n_pts': 500,
          'n_rep': 3}

values_sim = defaultdict(list)
values_an = defaultdict(list)
for K in K_sim:
    res = simulation(K, **kwargs)
    for key, val in res.items():
        values_sim[key].append(val)

K_an = np.linspace(1, 40, 100)
for K in K_an:
    res = analytical(K)
    for key, val in res.items():
        values_an[key].append(val)
    
values_sim = {key: np.array(val) for key, val in values_sim.items()}
values_an = {key: np.array(val) for key, val in values_an.items()}

values_sim['mu_R_scaled'] = values_sim['mu_R']/values_sim['mu_R'].max()
values_sim['mu_S_scaled'] = values_sim['mu_S']/values_sim['mu_S'].max()
values_sim['Three-term approx scaled'] = \
    values_sim['Three-term approx']/values_sim['Three-term approx'].max()    
values_sim['mu_R/mu_S_scaled'] = values_sim['mu_R/mu_S']/values_sim['mu_R/mu_S'].max()

values_an['mu_R_scaled'] = values_an['mu_R']/values_an['mu_R'].max()
values_an['mu_S_scaled'] = values_an['mu_S']/values_an['mu_S'].max()
values_an['Three-term approx scaled'] = \
    values_an['Three-term approx']/values_an['Three-term approx'].max()
values_an['mu_R/mu_S_scaled'] = values_an['mu_R/mu_S']/values_an['mu_R/mu_S'].max()

plt.figure(figsize=(4, 2))
keys = ['theta']
for i, key in enumerate(keys):
    plt.plot(values_sim['K'], values_sim[key], 'o-', label=key)
plt.legend()


plt.figure(figsize=(4, 3))
keys = ['E[norm_dY/norm_Y]', 'Three-term approx']
for i, key in enumerate(keys):
    plt.plot(values_sim['K'], values_sim[key], 'o-', label=key)
plt.legend()


def _plot_compare(keys):
    colors = ['red', 'green', 'blue', 'orange']
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(4, 4))
    for i, key in enumerate(keys):
        axes[0].plot(values_sim['K'], values_sim[key], 'o-', label=key, color=colors[i])
    for i, key in enumerate(keys):
        axes[1].plot(values_an['K'], values_an[key], label=key, color=colors[i])
    axes[1].set_xlabel('K')
    plt.tight_layout()
    plt.legend()
    
def _plot_compare_overlay(keys):
    colors = ['red', 'green', 'blue', 'orange']
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    for i, key in enumerate(keys):
        ax.plot(values_sim['K'], values_sim[key], 'o-', label=key, color=colors[i])
        ax.plot(values_an['K'], values_an[key], color=colors[i])
    ax.set_xlabel('K')
    plt.legend()
    
def plot_compare(keys, overlay=True):
    if overlay:
        _plot_compare_overlay(keys)
    else:
        _plot_compare(keys)
    
plot_compare(['Three-term approx scaled'], overlay=True)
    
plot_compare(['mu_R/mu_S', 'E[S^2]mu_R/mu_S^3', 'E[SR]/mu_S^2'], overlay=False)

# plot_compare(['mu_R_scaled', 'mu_S_scaled'], overlay=True)
plot_compare(['mu_S_scaled'], overlay=True)

plot_compare(['mu_R_scaled'], overlay=True)

plot_compare(['mu_R/mu_S_scaled'], overlay=True)
