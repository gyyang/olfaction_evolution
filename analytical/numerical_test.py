#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Representation stability analysis."""

import time
from collections import defaultdict
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression

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


def perturb(M, beta, mode='multiplicative', dist='uniform'):
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
                    b_mode='percentile', use_relu=True, w_dist='uniform',
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

    M2 = perturb(M, beta=0.01, mode=perturb_mode, dist=perturb_dist)

    Y2 = np.dot(X, M2)
    
    if coding_level is not None:
        if b_mode == 'percentile':
            b = -np.percentile(Y.flatten(), 100-coding_level)
        elif b_mode == 'gaussian':
            # b = -(np.mean(Y) + np.std(Y)*1)
            b = - (K/2 + np.sqrt(K/4))
            # print(n_kc_claw, np.mean(Y), np.std(Y)**2/K*4)
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


def get_optimal_k(m):
    res = minimize_scalar(lambda k: analytical(k, m)['Three-term approx'],
                          bounds=(1, m), method='bounded')
    optimal_k = res.x
    return optimal_k


def plot_optimal_k():
    ms = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    optimal_ks = [get_optimal_k(m) for m in ms]
    
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
    plt.savefig('../figures/analytical/'+fname+'.pdf', transparent=True)
    plt.savefig('../figures/analytical/'+fname+'.png')


def main_plot_compare():
    m = 50
    # m = 1000
    if m == 50:
        K_sim = np.array([1, 3, 5, 7, 10, 12, 15, 20])
    elif m == 150:
        K_sim = np.array([5, 7, 10, 12, 13, 15, 20])
    elif m == 1000:
        K_sim = np.array([60, 65, 70, 75])
    kwargs = {'x_dist': 'gaussian',
              'normalize_x': False,
              'w_mode': 'exact',
              # 'w_mode': 'bernoulli',
              'w_dist': 'gaussian',
              'normalize_w': False,
              'b_mode': 'percentile',
              # 'b_mode': 'gaussian',
              'coding_level': 10,
              'use_relu': True,
              # 'use_relu': False,
              # 'perturb_mode': 'multiplicative',
              'perturb_mode': 'additive',
              'perturb_dist': 'gaussian',
              'n_pts': 500,
              'n_rep': 5,
              'n_pn': m}
    
    values_sim = defaultdict(list)
    values_an = defaultdict(list)
    for K in K_sim:
        res = simulation(K, **kwargs)
        for key, val in res.items():
            values_sim[key].append(val)
    
    K_an = np.linspace(K_sim.min(), K_sim.max(), 100)
    for K in K_an:
        res = analytical(K, m)
        for key, val in res.items():
            values_an[key].append(val)
        
    values_sim = {key: np.array(val) for key, val in values_sim.items()}
    values_an = {key: np.array(val) for key, val in values_an.items()}
    
    keys = ['mu_R', 'mu_S', 'Three-term approx', 'mu_R/mu_S', 'E[S^2]']
    for key in keys:
        values_sim[key+'_scaled'] = values_sim[key]/values_sim[key].max()
        values_an[key+'_scaled'] = values_an[key]/values_an[key].max()
    
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
        fname = ''.join(keys)
        fname = fname.replace('/', '')
        plt.savefig('../figures/analytical/'+fname+'.pdf', transparent=True)
        plt.savefig('../figures/analytical/'+fname+'.png')
        
    def _plot_compare_overlay(keys):
        colors = ['red', 'green', 'blue', 'orange']
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        for i, key in enumerate(keys):
            ax.plot(values_sim['K'], values_sim[key], 'o', label=key, color=colors[i], alpha=0.5)
            ax.plot(values_an['K'], values_an[key], color=colors[i])
        ax.set_xlabel('K')
        plt.legend()
        fname = ''.join(keys)
        fname = fname.replace('/', '')
        plt.savefig('../figures/analytical/'+fname+'.pdf', transparent=True)
        plt.savefig('../figures/analytical/'+fname+'.png')
    
        
    def plot_compare(keys, overlay=True):
        if overlay:
            _plot_compare_overlay(keys)
        else:
            _plot_compare(keys)
        
    plot_compare(['Three-term approx_scaled'], overlay=True)
        
    plot_compare(['mu_R/mu_S', 'E[S^2]mu_R/mu_S^3', 'E[SR]/mu_S^2'], overlay=False)
    
    # plot_compare(['mu_R_scaled', 'mu_S_scaled'], overlay=True)
    plot_compare(['mu_S_scaled'], overlay=True)
    
    plot_compare(['mu_R_scaled'], overlay=True)
    
    plot_compare(['mu_R/mu_S_scaled'], overlay=True)
    
    plot_compare(['E[S^2]_scaled'], overlay=True)
    

def get_optimal_K_simulation():
    def guess_optimal_K(m):
        """Get the optimal K based on latest estimation."""
        return np.exp(-0.75)*(m**0.706)
    
    ms = [50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for m in ms:
    # for m in [75]:
        K_mid = int(guess_optimal_K(m))
        K_sim = np.arange(int(K_mid)-5, int(K_mid)+5)
        
        kwargs = {'x_dist': 'gaussian',
                  'normalize_x': False,
                  'w_mode': 'exact',
                  # 'w_mode': 'bernoulli',
                  'w_dist': 'gaussian',
                  'normalize_w': False,
                  'b_mode': 'percentile',
                  # 'b_mode': 'gaussian',
                  'coding_level': 10,
                  'use_relu': True,
                  # 'use_relu': False,
                  # 'perturb_mode': 'multiplicative',
                  'perturb_mode': 'additive',
                  'perturb_dist': 'gaussian',
                  'n_pts': 500,
                  'n_rep': 1,
                  'n_pn': m}
        
        n_rep = 100
        all_values = list()
        for i in range(n_rep):
            start_time = time.time()
            values_sim = defaultdict(list)
            for K in K_sim:
                res = simulation(K, **kwargs)
                for key, val in res.items():
                    values_sim[key].append(val)
            all_values.append(values_sim)
            print('Time taken : {:0.2f}s'.format(time.time() - start_time))
        
        fn = 'all_value_m' + str(m)
        try:
            pickle.dump(all_values, open('./files/analytical/'+fn+'.pkl', "wb"))
        except FileNotFoundError:
            pickle.dump(all_values, open('../files/analytical/'+fn+'.pkl', "wb"))

if __name__ == '__main__':
    main_plot_compare()