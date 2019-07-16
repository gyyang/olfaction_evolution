#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Representation stability analysis."""

import os
import sys
import time
from collections import defaultdict
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import tools
from tools import nicename
from oracle.evaluatewithnoise import _select_random_directions

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


def perturb(M, beta, mode='multiplicative', dist='uniform', normalize_w=False, K=None):
    if mode == 'multiplicative':
        if dist == 'uniform':
            P = np.random.uniform(1-beta, 1+beta, size=M.shape)            
        else:
            P = np.random.randn(*M.shape) * beta + 1
        return M * P
    elif mode == 'additive':
        if dist == 'uniform':
            P = np.random.uniform(-beta, beta, size=M.shape) * np.max(M)
        else:
            P = np.random.randn(*M.shape) * beta
        if normalize_w:
            P /= K
        return M + P * (M > 1e-6)  # only applied on connected weights
    else:
        raise ValueError('Unknown perturb mode')


def _get_M(n_pn, n_kc, n_kc_claw, sign_constraint=True,
           mode='exact', normalize=True, dist='uniform'):
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
    
    # TEMPORARYLY SKIPPED
# =============================================================================
#     if dist == 'uniform':
#         M = perturb(M, 0.5, 'multiplicative')  # pre-perturb
#     else:
#         M = perturb(M, 0.2, 'multiplicative', 'gaussian')  # pre-perturb
# =============================================================================
    
    # M = model.get_sparse_mask(n_pn, n_kc, n_kc_claw)
    return M


def _analyze_perturb(n_pn=N_PN, n_kc=N_KC, n_kc_claw=N_KC_CLAW,
                    coding_level=None, same_threshold=True, n_pts=10,
                    perturb_mode='multiplicative', ff_inh=False, normalize_x=True,
                    x_dist='uniform', w_mode='exact', normalize_w=True,
                    b_mode='percentile', activation='relu', w_dist='uniform',
                    perturb_dist='uniform'):
    if x_dist == 'uniform':
        X = np.random.rand(n_pts, n_pn)
    elif x_dist == 'gaussian':
        X = np.random.randn(n_pts, n_pn) * 0.5 + 0.5
    else:
        raise NotImplementedError()

    if normalize_x:
        X = normalize(X)

    M = _get_M(n_pn, n_kc, n_kc_claw, mode=w_mode, normalize=normalize_w, dist=w_dist)
    
    # M = np.random.uniform(0.5, 1.5, size=(n_pn, n_kc)) * (np.random.rand(n_pn, n_kc)<n_kc_claw/n_pn)

    if ff_inh:
        # TODO: think how to weight perturb with ff inh
        M = M - M.sum(axis=0).mean()
    
    Y = np.dot(X, M)

    if perturb_mode == 'input':
        dX = np.random.randn(*X.shape)*0.01
        Y2 = np.dot(X+dX, M)
    else:
        M2 = perturb(M, beta=0.01, mode=perturb_mode, dist=perturb_dist,
                     normalize_w=normalize_w, K=n_kc_claw)
        Y2 = np.dot(X, M2)
    
    if coding_level is not None:
        if b_mode == 'percentile':
            b = -np.percentile(Y.flatten(), 100-coding_level)
        elif b_mode == 'gaussian':
            b = -(np.mean(Y) + np.std(Y)*1)
            # b = - (K/2 + np.sqrt(K/4))
            # print(n_kc_claw, np.mean(Y), np.std(Y)**2/K*4)
        else:
            raise NotImplementedError()
        Y = Y + b
        if not same_threshold:
            b = -np.percentile(Y2.flatten(), 100-coding_level)
        Y2 = Y2 + b
        
        if activation == 'relu':
            Y = relu(Y)
            Y2 = relu(Y2)
        elif activation == 'tanh':
            Y = np.tanh(Y)
            Y2 = np.tanh(Y2)
        elif activation == 'retanh':
            Y = np.tanh(relu(Y))
            Y2 = np.tanh(relu(Y2))
        elif activation == 'none':
            pass
        else:
            raise NotImplementedError('Unknown activation')
    
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


def simulation(K, compute_dimension=False, **kwargs):
    print('Simulating K = ' + str(K))
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
    ES2 = np.mean(S**2)
    first_term = mu_R/mu_S
    second_term = first_term * (ES2/mu_S**2)
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
    res['E[S^2]'] = ES2
    
    if compute_dimension:
        Y_centered = Y - Y.mean(axis=0)
        # l, _ = np.linalg.eig(np.dot(Y_centered.T, Y_centered))
        # u, s, vh = np.linalg.svd(Y_centered, full_matrices=False)
        # l = s**2
        # dim = np.sum(l)**2/np.sum(l**2)        
        
        C = np.dot(Y_centered.T, Y_centered) / Y.shape[0]
        C2 = C**2
    
        diag_mask = np.eye(C.shape[0],dtype=bool)
            
        E_C2ij = np.mean(C2[~diag_mask])
        E_C2ii = np.mean(C2[diag_mask])
        E_Cii = np.mean(C[diag_mask])
        
        dim = n_kc * E_Cii**2 / (E_C2ii + (n_kc-1) * E_C2ij)

        res['E_Cii'] = E_Cii
        res['E_C2ii'] = E_C2ii
        res['E_C2ij'] = E_C2ij
        res['dim'] = m * E_Cii**2 / (E_C2ii + (m-1) * E_C2ij)
        
    return res


def simulation_perturboutput(K, **kwargs):
    # Get the representation at the expansion layer
    X, Y, Y2, dY = analyze_perturb(n_kc_claw=K, **kwargs)

    W = np.random.randn(Y.shape[1], 100)/np.sqrt(Y.shape[1])
    W2 = _select_random_directions(W)*0.01
    # W2 = np.random.randn(Y.shape[1], 100)/np.sqrt(Y.shape[1])*0.1
    
    Y_original = Y.copy()
    Y = np.dot(Y_original, W)
    dY = np.dot(Y_original, W2)
    Y2 = Y+dY

    norm_Y = np.linalg.norm(Y, axis=1)
    norm_Y2 = np.linalg.norm(Y2, axis=1)
    norm_dY = np.linalg.norm(dY, axis=1)
    
    cos_theta = (np.sum(Y * Y2, axis=1) / (norm_Y * norm_Y2))
    cos_theta = cos_theta[(norm_Y * norm_Y2)>0]
    theta = np.arccos(cos_theta)/np.pi*180
    
    norm_ratio = norm_dY/norm_Y
    norm_ratio = norm_ratio[norm_Y>0]

    res = dict()
    res['K'] = K
    res['E[norm_dY/norm_Y]'] = np.mean(norm_ratio)
    res['theta'] = np.mean(theta)

    return res


from scipy.integrate import quad, dblquad
def I(k, m):
    rho = k/m
    a = 1
    # return dblquad(lambda x, y: (x-a)**2*(y-a)**2*np.exp(-x**2/2-y**2/2+x*y/(1-rho**2)), a, np.inf, a, np.inf)
    tmp = dblquad(lambda y, x: (x-a)**2*(y-a)**2*np.exp(-(x**2+y**2-2*rho*x*y)/2/(1-rho**2)),
                   a, np.inf, lambda x: a, lambda x: np.inf)[0]
    return tmp / (1-rho**2) * k**2


from scipy.special import erf
def f(r, K):
    mu_x = 0.5
    sigma_x = np.sqrt(0.5)
    mu_r = (K-1)*mu_x
    sigma_r = np.sqrt(K-1)*sigma_x
    c = 1
    b = -(K*mu_x+c*np.sqrt(K)*sigma_x)
    B = (mu_x+mu_r+b + np.sqrt(2)*sigma_r*r)/np.sqrt(2)/sigma_x
    
    tmp1 = np.sqrt(np.pi)/2*(mu_x**2+sigma_x**2)*(erf(B)+1)
    tmp2 = (np.sqrt(2)*sigma_x*mu_x-sigma_x**2*B)*np.exp(-B**2)
    
    tmp = tmp1 + tmp2
    return np.exp(-r**2) * tmp / np.pi



def G1(x, q, K):
    mu_x = 0.5
    sigma_x = np.sqrt(0.5)
    mu_q = (K-2)*mu_x
    sigma_q = np.sqrt(K-2)*sigma_x
    c = 1
    b = -(K*mu_x+c*np.sqrt(K)*sigma_x)
    B = np.sqrt(2)*sigma_x*x+mu_x+np.sqrt(2)*sigma_q*q+mu_q+b
    tmp = np.sqrt(2)*sigma_x*np.exp(-B**2)/2+mu_x/2*np.sqrt(np.pi)*(erf(B)+1)
    return np.exp(-q**2-x**2)*(x*np.sqrt(2)*sigma_x+mu_x)*tmp / np.pi**1.5


def G2(r, K):
    mu_x = 0.5
    sigma_x = np.sqrt(0.5)
    mu_r = (K-1)*mu_x
    sigma_r = np.sqrt(K-1)*sigma_x
    c = 1
    b = -(K*mu_x+c*np.sqrt(K)*sigma_x)
    B = (mu_x+mu_r+b + np.sqrt(2)*sigma_r*r)/np.sqrt(2)/sigma_x
    
    tmp1 = np.sqrt(np.pi)/2*mu_x*(erf(B)+1)
    tmp2 = np.sqrt(2)*sigma_x*np.exp(-B**2)/2
    tmp = tmp1 + tmp2
    return np.exp(-r**2) * tmp / np.pi


def I2(k):
    mu_x = 0.5
    sigma_x = np.sqrt(0.5)
    c = 1
    b = -(k*mu_x+c*np.sqrt(k)*sigma_x)
    
    tmp1 = dblquad(lambda x, q: G1(x, q, k),
                   -np.inf, np.inf, -np.inf, np.inf)[0]
        
    tmp2 = quad(lambda r: G2(r, k), -np.inf, np.inf)[0]
        
    tmp = tmp1 * k**3 + 2*b*k**2*tmp2 + b**2
    return tmp


def P1(C):
    np.exp(((C - mu_A)/sigma_A)**2/2) * (A+b)**2
    

# =============================================================================
# for K in np.arange(1, 40):
#     res = quad(lambda x: f(x, K), -np.inf, np.inf)
#     print(K, res)
# =============================================================================

def analytical(K, m=50):
    k = K
    sigma_x = np.sqrt(0.5)

    mu_S = 1/4 * k
    mu_R = k*quad(lambda x: f(x, K), -np.inf, np.inf)[0]
    
    e_s2 = I(k, m)
    # e_sr = I2(k) * mu_R
    tmp = 1
    e_sr = mu_R * k * (sigma_x**2) * tmp
        
    first_term = mu_R/mu_S
    second_term = e_s2*mu_R/(mu_S**3)
    third_term = e_sr/mu_S**2
    
    # approx = np.sqrt(first_term + second_term - third_term)
    approx = np.sqrt(first_term + second_term)
    
    res = dict()
    res['K'] = k
    res['mu_S'] = mu_S
    res['mu_R'] = mu_R
    res['E[SR]'] = e_sr
    
    res['Three-term approx'] = approx
    res['mu_R/mu_S'] = mu_R/mu_S
    
    res['E[SR]/mu_S^2'] = e_sr/mu_S**2
    res['E[S^2]'] = e_s2
    res['E[S^2]mu_R/mu_S^3'] = res['E[S^2]'] * res['mu_R'] / res['mu_S']**3
    # res['E[S^2]'] = k**2
    return res


def _get_optimal_k(m):
    res = minimize_scalar(lambda k: analytical(k, m)['Three-term approx'],
                          bounds=(1, m), method='bounded')
    optimal_k = res.x
    return optimal_k


def get_optimal_k():
    ms = np.logspace(1, 4, 100, dtype=int)
    optimal_Ks = list()
    for m in ms:
        k = _get_optimal_k(m)
        print('m ', m, ' k ', k)
        optimal_Ks.append(k)
        
    res = {'ms': ms, 'optimal_Ks': optimal_Ks}
    
    file = os.path.join(rootpath, 'files', 'analytical', 'optimal_k_two_term')
    with open(file+'.pkl', 'wb') as f:
        pickle.dump(res, f)

    return ms, optimal_Ks

def plot_optimal_k():
    # ms = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    x = np.log(ms)
    y = np.log(optimal_ks)
    
    x_plot = np.linspace(x[0], x[-1], 100)

    # model = Ridge()
    model = LinearRegression()
    model.fit(x[:, np.newaxis], y)
    y_plot = model.predict(x_plot[:, np.newaxis])
    
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(x, y, 'o', label='analytical')
    fit_txt = '$log k = {:0.2f}log m + {:0.2f}$'.format(model.coef_[0], model.intercept_)
    ax.plot(x_plot, y_plot, label='fit: ' + fit_txt)
    ax.set_xlabel('m')
    ax.set_ylabel('optimal K')
    xticks = np.array([10, 100, 1000, 10000])
    ax.set_xticks(np.log(xticks))
    ax.set_xticklabels([str(t) for t in xticks])
    yticks = np.array([10, 100])
    ax.set_yticks(np.log(yticks))
    ax.set_yticklabels([str(t) for t in yticks])
    ax.legend()
    fname = 'optimal_k'
    # plt.savefig('../figures/analytical/'+fname+'.pdf', transparent=True)
    # plt.savefig('../figures/analytical/'+fname+'.png')


def main_compare(perturb='pn2kc', m=50, activation='relu',
                 compute_dimension=False):
    if m == 50:
        # K_sim = np.array([1, 3, 5, 7, 10, 12, 15, 20, 25, 30])
        K_sim = np.arange(1, 31)
    elif m == 150:
        K_sim = np.array([5, 7, 10, 12, 13, 15, 20])
    elif m == 1000:
        K_sim = np.logspace(0.8, 2.5, 20, dtype='int')
    
    if not compute_dimension:
        n_rep = 50
        n_pts = 500
    else:
        n_rep = 1
        n_pts = 5000
        
    kwargs = {'x_dist': 'gaussian',
              'normalize_x': False,
              'w_mode': 'exact',
              # 'w_mode': 'bernoulli',
              'w_dist': 'gaussian',
              'normalize_w': False,
              'b_mode': 'percentile',
              # 'b_mode': 'gaussian',
              'coding_level': 10,
              # 'coding_level': 70,
              'activation': activation,
              # 'activation': False,
              # 'perturb_mode': 'multiplicative',
              'perturb_mode': 'additive',
              'perturb_dist': 'gaussian',
              'n_pts': n_pts,
              # 'n_rep': 50,
              'n_rep': n_rep,
              'n_pn': m}
    
    if perturb == 'pn2kc':
        values_sim = defaultdict(list)
        values_an = defaultdict(list)
        for K in K_sim:
            res = simulation(K, compute_dimension=compute_dimension, **kwargs)
            for key, val in res.items():
                values_sim[key].append(val)
        
        K_an = np.linspace(K_sim.min(), K_sim.max(), 100)
        for K in K_an:
            res = analytical(K, m)
            for key, val in res.items():
                values_an[key].append(val)
        
        for v, name in zip([values_sim, values_an], ['sim', 'an']):       
            file = os.path.join(rootpath, 'files', 'analytical', name+'_m'+str(m))
            if activation != 'relu':
                file = file + activation
            if compute_dimension:
                file = file + '_dim'
            with open(file+'.pkl', 'wb') as f:
                pickle.dump(v, f)

    elif perturb == 'output':        
        values_sim = defaultdict(list)
        for K in K_sim:
            res = simulation_perturboutput(K, **kwargs)
            for key, val in res.items():
                values_sim[key].append(val)
        
        file = os.path.join(rootpath, 'files', 'analytical', 'sim_perturboutput_m'+str(m))
        if activation != 'relu':
            file = file + activation
        with open(file+'.pkl', 'wb') as f:
            pickle.dump(values_sim, f)
    
    elif perturb == 'input':
        kwargs['perturb_mode'] = 'input'
        values_sim = defaultdict(list)
        for K in K_sim:
            res = simulation(K, **kwargs)
            for key, val in res.items():
                values_sim[key].append(val)
        
        file = os.path.join(rootpath, 'files', 'analytical', 'sim_perturbinput_m'+str(m))
        if activation != 'relu':
            file = file + activation
        with open(file+'.pkl', 'wb') as f:
            pickle.dump(values_sim, f)
            
            
def _plot_compare(values, keys):
    colors = ['red', 'green', 'blue', 'orange']
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(4, 4))
    for i, key in enumerate(keys):
        axes[0].plot(values['sim']['K'], values['sim'][key], 'o-', label=key, color=colors[i])
    for i, key in enumerate(keys):
        axes[1].plot(values['an']['K'], values['an'][key], label=key, color=colors[i])
    axes[1].set_xlabel('K')
    plt.tight_layout()
    plt.legend()
    fname = ''.join(keys)
    fname = fname.replace('/', '')
    fname = os.path.join(rootpath, 'figures', 'analytical', fname)
    plt.savefig(fname+'.pdf', transparent=True)
    plt.savefig(fname+'.png')

    
def _plot_compare_overlay(values, keys):
    colors = [tools.red, 'green', 'blue', 'orange']
    fig = plt.figure(figsize=(1.5, 1.2))
    ax = fig.add_axes([0.3, 0.3, 0.6, 0.5])
    for i, key in enumerate(keys):
        ax.plot(values['sim']['K'], values['sim'][key], 'o', label=key,
                color=colors[i], alpha=0.5, markersize=3)
        ax.plot(values['an']['K'], values['an'][key], color=colors[i])
    # if keys[0] != 'E[S^2]/mu_S^2_scaled':
    if False:
        ax.set_xticklabels('')
    else:
        ax.set_xticks([0, 30])
        ax.set_xlabel('K', labelpad=0)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if len(keys) > 1:
        plt.legend()
    else:
        title = keys[0]
        if title == 'E[S^2]/mu_S^2_scaled':
            title = r'$E[|y|^4]$'
        elif title == 'mu_R/mu_S_scaled':
            title = r'$E[|\Delta|^2]$'
        plt.title(title, fontsize=7, pad=-4)
    
# =============================================================================
#     if 'scaled' in keys[0]:
#         plt.yticks([0, 1])
# =============================================================================
    fname = ''.join(keys)
    fname = fname.replace('/', '')
    fname = os.path.join(rootpath, 'figures', 'analytical', fname)
    plt.savefig(fname+'.pdf', transparent=True)
    plt.savefig(fname+'.png')

    
def plot_compare(values, keys, overlay=True):
    if overlay:
        _plot_compare_overlay(values, keys)
    else:
        _plot_compare(values, keys)


def main_plot(m=50, logK=False, activation='relu'):
    values = list()
    for name in ['sim', 'an']:       
        file = os.path.join(rootpath, 'files', 'analytical', name+'_m'+str(m))
        if activation != 'relu':
            file = file + activation
        with open(file+'.pkl', 'rb') as f:
            values.append(pickle.load(f))
    
    values_sim, values_an = values
    values_sim = {key: np.array(val) for key, val in values_sim.items()}
    values_an = {key: np.array(val) for key, val in values_an.items()}
    
    values_sim['E[S^2]/mu_S^2'] = values_sim['E[S^2]']/(values_sim['mu_S'])**2
    values_an['E[S^2]/mu_S^2'] = values_an['E[S^2]']/(values_an['mu_S'])**2
    
    keys = ['mu_R', 'mu_S', 'Three-term approx', 'mu_R/mu_S', 'E[S^2]',
            'E[S^2]/mu_S^2']
    for key in keys:
        values_sim[key+'_scaled'] = values_sim[key]/values_sim[key].max()
        values_an[key+'_scaled'] = values_an[key]/values_an[key].max()
    
    if m == 50:
        xticks = [3, 7, 15, 30]
    elif m == 1000:
        xticks = [10, 100, 1000]
    else:
        xticks = []
        
    if logK:
        xplot = np.log(values_sim['K'])
    else:
        xplot = values_sim['K']
    
    fig = plt.figure(figsize=(1.5, 1.5))
    ax = fig.add_axes((0.3, 0.3, 0.6, 0.55))
    ax.plot(xplot, values_sim['theta'], 'o-', markersize=2, color=tools.red)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if logK:
        ax.xaxis.set_ticks([np.log(xtick) for xtick in xticks])
        ax.xaxis.set_ticklabels([str(xtick) for xtick in xticks])
    else:
        ax.xaxis.set_ticks(xticks)
    plt.xlabel('K')
    plt.ylabel('Mean perturbation angle')
    if m == 50:
        ax.plot([7, 7], [ax.get_ylim()[0], ax.get_ylim()[-1]], '--', color = 'gray')
    plt.locator_params(axis='y', nbins=2)
    plt.title('N = ' + str(m), fontsize=7)
    figname = os.path.join(rootpath, 'figures', 'analytical',
                           'theta_vs_K_N'+str(m))
    if activation != 'relu':
        figname = figname + activation
    plt.savefig(figname+'.pdf', transparent=True)
    plt.savefig(figname+'.png')

    try:
        fig = plt.figure(figsize=(1.5, 1.5))
        ax = fig.add_axes((0.35, 0.3, 0.6, 0.55))
        ax.plot(values_sim['K'], values_sim['dim'], 'o-', markersize=2, color=tools.red)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks(xticks)
        plt.xlabel(nicename('kc_inputs'))
        plt.ylabel('Dimensionality')
        if m == 50:
            ax.plot([7, 7], [ax.get_ylim()[0], ax.get_ylim()[-1]], '--', color = 'gray')
        plt.locator_params(axis='y', nbins=2)
        figname = os.path.join(rootpath, 'figures', 'analytical', 'dim_vs_K')
        plt.savefig(figname+'.pdf', transparent=True)
        plt.savefig(figname+'.png')
    except KeyError:
        pass
    
    values = {'sim': values_sim, 'an': values_an}
    
    plt.figure(figsize=(4, 3))
    keys = ['E[norm_dY/norm_Y]', 'Three-term approx']
    for i, key in enumerate(keys):
        plt.plot(values_sim['K'], values_sim[key], 'o-', label=key)
    plt.legend()
        
    plot_compare(values, ['Three-term approx_scaled'], overlay=True)
        
    plot_compare(values, ['mu_R/mu_S', 'E[S^2]mu_R/mu_S^3', 'E[SR]/mu_S^2'],
                 overlay=False)
    
    # plot_compare(['mu_R_scaled', 'mu_S_scaled'], overlay=True)
    plot_compare(values, ['mu_S_scaled'], overlay=True)
    
    plot_compare(values, ['mu_R_scaled'], overlay=True)
    
    plot_compare(values, ['mu_R/mu_S_scaled'], overlay=True)
    
    plot_compare(values, ['E[S^2]_scaled'], overlay=True)
    
    plot_compare(values, ['E[S^2]/mu_S^2_scaled'], overlay=True)
    

def compare_dim_plot(m=50, logK=False, activation='relu'):
    """Compare dimensionality relevant variables with weight perturbation."""
    values = list()
    for name in ['', '_dim']:       
        file = os.path.join(rootpath, 'files', 'analytical', 'sim_m'+str(m))
        if activation != 'relu':
            file = file + activation
        file += name
        with open(file+'.pkl', 'rb') as f:
            value = pickle.load(f)
        
        value = {key: np.array(val) for key, val in value.items()}
        values.append(value)
    
    values, values_dim = values
        
    if m == 50:
        xticks = [3, 7, 15, 30]
    elif m == 1000:
        xticks = [10, 100, 1000]
    else:
        xticks = []
        
    if logK:
        xplot = np.log(values_dim['K'])
    else:
        xplot = values_dim['K']
    
    keys = ['dim', 'E_Cii', 'E_C2ij', 'E_C2ii']
    for i, key in enumerate(keys):
        plt.figure(figsize=(4, 3))
        plt.plot(values_dim['K'], values_dim[key], 'o-', label=key)
        plt.legend()
        
    keys = ['E[S^2]']
    for i, key in enumerate(keys):
        plt.figure(figsize=(4, 3))
        plt.plot(values['K'], values[key], 'o-', label=key)
        plt.legend()
    
    
def main_plot_perturb(perturb):
    m = 50
    file = os.path.join(rootpath, 'files', 'analytical',
                        'sim_perturb'+perturb+'_m'+str(m))
    with open(file+'.pkl', 'rb') as f:
        values_sim = pickle.load(f)

    fig = plt.figure(figsize=(1.5, 1.5))
    ax = fig.add_axes((0.35, 0.3, 0.6, 0.55))
    ax.plot(values_sim['K'], values_sim['theta'], 'o-', markersize=2, color=tools.red)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks([3, 7, 15, 30])
    plt.xlabel(nicename('kc_inputs'))
    plt.ylabel('Mean perturbation angle')
    ax.plot([7, 7], [ax.get_ylim()[0], ax.get_ylim()[-1]], '--', color = 'gray')
    plt.locator_params(axis='y', nbins=2)
    figname = os.path.join(rootpath, 'figures', 'analytical', 'theta_vs_K_perturb'+perturb)
    plt.title('Perturbing '+perturb, fontsize=7)
    plt.savefig(figname+'.pdf', transparent=True)
    plt.savefig(figname+'.png')

    
    plt.figure(figsize=(4, 3))
    keys = ['E[norm_dY/norm_Y]']
    for i, key in enumerate(keys):
        plt.plot(values_sim['K'], values_sim[key], 'o-', label=key)
    plt.legend()





def compute_optimalloss_onestepgradient():
    import time
    m = 50
    for K in [1, 3, 7, 10, 20]:
        start = time.time()
        kwargs = {'x_dist': 'gaussian',
                      'normalize_x': False,
                      'w_mode': 'exact',
                      # 'w_mode': 'bernoulli',
                      'w_dist': 'gaussian',
                      'normalize_w': False,
                      'b_mode': 'percentile',
                      # 'b_mode': 'gaussian',
                      'coding_level': 10,
                      'activation': 'relu',
                      'ff_inh': False,
                      # 'perturb_mode': 'multiplicative',
                      'perturb_mode': 'additive',
                      'perturb_dist': 'gaussian',
                      'n_pts': 5000,
                      'n_rep': 1,
                      'n_kc': 2500,
                      'n_pn': m}

        X, Y, Y2, dY = analyze_perturb(n_kc_claw=K, **kwargs)
        
        print('K', K)
        
        # compute_expected_loss_singley(Y)
        # e_L = compute_expected_loss_doubley(Y)
        # e_L = compute_expected_loss_multiy(Y)
        
        n_pts, n_dim = Y.shape
        
        P = 100
        
        # print('time1', time.time() - start)
        
        # z = (np.random.rand(n_pts)>0.5)*1.0
        z = (np.mod(np.arange(n_pts), P) == 0)*1.0  # only one 1 every P
        
        s = np.sum(Y, axis=1)
        s2 = s**2
        
        e_s = np.mean(s)
        e_s2 = np.mean(s2)
        e_z = np.mean(z)
        e_z2 = np.mean(z**2)
        
        # beta = Es*Ez/Es^2
        beta = e_s*e_z/e_s2
        
# =============================================================================
#         d = beta*s-z
#         
#         print('beta', beta)
#         
#         C0 = e_z2 - e_s**2*e_z**2 / e_s2
#         print('C0', C0)
#         
#         Q = np.dot(Y, Y.T)
#         dd = np.outer(d, d)  # (n_pts, n_pts)
#         ddQ = Q * dd
#         offdiag_mask = ~np.eye(ddQ.shape[0],dtype=bool)
# 
#         # C1 = E[d_i d_j Q_ij]
#         # C1_tmp1 E[d_i^2 Q_ii]
#         C1_tmp1 = np.mean(d**2 * Q.diagonal()) / P
#         # C1_tmp2 E[d_i d_j Q_ij | i \neq j]
#         C1_tmp2 = np.mean(ddQ[offdiag_mask]) * (P-1)/P
#         print('C1_tmp1', C1_tmp1)
#         print('C1_tmp2', C1_tmp2)
#         
#         C1 = C1_tmp1 + C1_tmp2
#         print('C1', C1)
#         
#         # C2 = E[d_j d_k q_ji q_ki]
#         # C2_tmp1 E[d_i^2 q_ii^2]
#         C2_tmp1 = np.mean(d**2 * Q.diagonal()**2) / P**2
#         # C2_tmp2 E[d_i d_j q_ij q_ii]
#         C2_tmp2 = np.mean((ddQ*Q.diagonal())[offdiag_mask]) *2*(P-1)/P**2
#         
#         C2 = C2_tmp1 + C2_tmp2
#         
#         
#         print('C2_tmp1', C2_tmp1)
#         print('C2_tmp2', C2_tmp2)
#         
#         print('C2', C2)
#         print('C1^2/C2', C1**2/C2)
#         
#         e_L = C0 - C1**2/C2
#         
#         alpha = C1 / C2
#         print('alpha', alpha)
#            
#         print('Loss', e_L)
# =============================================================================
# =============================================================================
#         old_beta = beta
# 
#         def get_orig_L(beta):
#             w = np.ones(n_dim) * beta
#             d = np.dot(Y, w) - z
#             L = np.mean(d**2)
#             return L
#         
#         res = minimize_scalar(get_orig_L, bounds=(1e-7, 1), method='bounded', 
#                               options={'xatol': 1e-5})
#         print('old beta', old_beta)
#         print(res)
#         L = res.fun
#         beta = res.x
# =============================================================================
        
        d = beta * s - z
        # Q = np.dot(Y, Y.T)
        
        A = np.zeros(n_pts)
        for i in range(n_pts):
            ind = np.mod(np.arange(i, i+P), n_pts)
            A[i] = np.mean(d[ind] * np.dot(Y[ind, :], Y[i]))
        
        B = d
        
        
        alpha = np.mean(A*B) / np.mean(A**2)  # this is correct
        
        # print('alpha prediction', alpha)
        
        w = np.ones(n_dim) * beta
        d = np.dot(Y, w) - z
        L = np.mean(d**2)
        dY = (d*Y.T).T  # d * Y
        
        print('Direct estimate loss0', L)  # this should match C0
        
        def get_L(alpha):
            def _get_L(indices):
                delta_w = -alpha * np.mean(dY[indices, :], axis=0)
                d = np.dot(Y[indices, :], w+delta_w) - z[indices]
                L = np.mean(d**2)
                return L
            
            L = 0
            for i in np.arange(n_pts - P):
                ind = np.arange(i, i+P)
                L_tmp = _get_L(ind)
                L += L_tmp
            L /= n_pts - P
            return L
        
# =============================================================================
#         res = minimize_scalar(get_L, bounds=(1e-7, 100), method='bounded', 
#                               options={'xatol': 1e-5})
#         # L = get_L(alpha/2)
#         L = res.fun
#         print(res)
# =============================================================================
        
        # L = get_L(alpha)
        
        L = np.mean((B-alpha*A)**2)
        
        print('Direct estimate loss', L)  # this should match e_L
        
        print('')
       

if __name__ == '__main__':
    pass
    # main_compare(m=50, activation='relu', compute_dimension=False)
    # main_plot(m=50, logK=False, activation='relu')
    # main_compare(perturb='pn2kc')
    # main_plot_perturboutput()
    # main_compare(perturb='input')
    # main_plot_perturb(perturb='input')
    # get_optimal_K_simulation_participationratio()
    # get_optimal_k()
    # compare_dim_plot()