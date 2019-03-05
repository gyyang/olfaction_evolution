#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Representation stability analysis."""

import numpy as np
import matplotlib.pyplot as plt

import model


N_PN = 50
N_KC = 2500
N_KC_CLAW = 7

def _get_M(n_pn, n_kc, n_kc_claw):
    M = model.get_sparse_mask(n_pn, n_kc, n_kc_claw) / n_kc_claw
    return M


def perturb(M, beta):
    P = np.random.uniform(1-beta, 1+beta, size=M.shape)
    return M * P


def get_perturb_diff(n_pn=N_PN, n_kc=N_KC, n_kc_claw=N_KC_CLAW):
    n_pts = 1000
    X = np.random.rand(n_pts, 50)
    
    M = _get_M(n_pn, n_kc, n_kc_claw)
    Y = np.dot(X, M)
    
    M2 = perturb(M, beta=0.5)
    Y2 = np.dot(X, M2)
    
    diff = np.mean(abs(Y-Y2))

    return diff


def get_diff_by_n_kc_claw():
    n_kc_claws = np.arange(1, 20)
    diffs = [get_perturb_diff(n_kc_claw=n) for n in n_kc_claws]

    return n_kc_claws, diffs


n_kc_claws, diffs = get_diff_by_n_kc_claw()

plt.plot(n_kc_claws, diffs, 'o-')
plt.xlabel('N KC claws')
plt.ylabel('Distortion after multiplicative noise')
        

