#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Representation stability analysis."""

import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

N_PN = 50
N_KC = 2500
N_KC_CLAW = 7


def relu(x):
    return x * (x > 0.)


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


def _get_M(n_pn, n_kc, n_kc_claw, sign_constraint=True,
                 mode='exact', normalize=True, dist='uniform'):
    if mode == 'exact':
        M = get_sparse_mask(n_pn, n_kc, n_kc_claw)
    elif mode == 'bernoulli':
        raise NotImplementedError
        M = (np.random.rand(n_pn, n_kc) < (n_kc_claw / n_pn))
    else:
        raise NotImplementedError

    if normalize:
        M = M / np.sqrt(n_kc_claw)

    return M


def perturb(M, beta, mode='multiplicative', dist='uniform',
                  normalize_w=False, K=None):
    if mode == 'multiplicative':
        if dist == 'uniform':
            P = torch.rand(M.shape, device=device) * (2 * beta) + (1 - beta)
        else:
            P = torch.randn(M.shape) * beta + 1
        return M * P
    elif mode == 'additive':
        if dist == 'uniform':
            P = (torch.rand(M.shape, device=device) - 0.5) * (
                        2 * beta * torch.max(M))
        else:
            P = torch.randn(M.shape, device=device) * beta
        if normalize_w:
            P /= K
        return M + P * (M > 1e-6).float()  # only applied on connected weights
    else:
        raise ValueError('Unknown perturb mode')


def _analyze_perturb(n_pn=N_PN, n_kc=N_KC, n_kc_claw=N_KC_CLAW,
                           n_pts=10,
                           perturb_mode='multiplicative', ff_inh=False,
                           normalize_x=True,
                           x_dist='uniform', w_mode='exact', normalize_w=True,
                           w_dist='uniform',
                           perturb_dist='uniform', **kwargs):
    if x_dist == 'uniform':
        X = torch.rand(n_pts, n_pn, device=device)
    elif x_dist == 'gaussian':
        X = torch.randn(n_pts, n_pn, device=device) * 0.5 + 0.5
    else:
        raise NotImplementedError()

    if normalize_x:
        raise NotImplementedError
        X = normalize(X)
    M = _get_M(n_pn, n_kc, n_kc_claw, mode=w_mode, normalize=normalize_w,
               dist=w_dist)

    M = torch.from_numpy(M).float().to(device)

    if ff_inh:
        raise NotImplementedError
        # TODO: think how to weight perturb with ff inh
        M = M - M.sum(axis=0).mean()

    Y = torch.mm(X, M)

    if perturb_mode == 'input':
        raise NotImplementedError
        dX = np.random.randn(*X.shape) * 0.01
        Y2 = np.dot(X + dX, M)
    else:
        M2 = perturb(M, beta=0.01, mode=perturb_mode, dist=perturb_dist,
                           normalize_w=normalize_w, K=n_kc_claw)
        Y2 = torch.mm(X, M2)

    return X, Y, Y2


def analyze_perturb(n_kc_claw=N_KC_CLAW, n_rep=1, **kwargs):

    X, Y, Y2 = list(), list(), list()
    for i in range(n_rep):
        X0, Y0, Y20 = _analyze_perturb(n_kc_claw=n_kc_claw, **kwargs)
        X.append(X0)
        Y.append(Y0)
        Y2.append(Y20)

    X = torch.cat(X, axis=0)
    Y = torch.cat(Y, axis=0)
    Y2 = torch.cat(Y2, axis=0)


    return X, Y, Y2


def set_coding_level(Y, Y2, coding_level=None, same_threshold=True,
                           b_mode='percentile', activation='relu', **kwargs):
    if coding_level is not None:
        if b_mode == 'percentile':
            kth = int((100 - coding_level) / 100. * torch.numel(Y))
            b = - torch.kthvalue(torch.flatten(Y), kth).values  # efficient??
        elif b_mode == 'gaussian':
            raise NotImplementedError
            b = -(np.mean(Y) + np.std(Y) * 1)
            # b = - (K/2 + np.sqrt(K/4))
            # print(n_kc_claw, np.mean(Y), np.std(Y)**2/K*4)
        else:
            raise NotImplementedError()
        Y = Y + b

        if not same_threshold:
            raise NotImplementedError
            b = -np.percentile(Y2.flatten(), 100 - coding_level)
        Y2 = Y2 + b

        if activation == 'relu':
            Y = torch.nn.functional.relu(Y)
            Y2 = torch.nn.functional.relu(Y2)
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

    return Y, Y2


def _simulation(Y, Y2, compute_dimension=False):
    n_kc = Y.shape[1]
    #     dY = Y2 - Y
    #     norm_Y = np.linalg.norm(Y, axis=1)
    #     norm_Y2 = np.linalg.norm(Y2, axis=1)
    #     norm_dY = np.linalg.norm(dY, axis=1)

    #     cos_theta = (np.sum(Y * Y2, axis=1) / (norm_Y * norm_Y2))
    #     cos_theta = cos_theta[(norm_Y * norm_Y2) > 0]
    #     theta = np.arccos(cos_theta) / np.pi * 180

    #     norm_ratio = norm_dY / norm_Y
    #     norm_ratio = norm_ratio[norm_Y > 0]

    #     S = norm_Y ** 2
    #     R = norm_dY ** 2

    #     corr = np.var(S) / np.mean(S) ** 2
    #     mu_S = np.mean(S)
    #     mu_R = np.mean(R)
    #     ES2 = np.mean(S ** 2)
    #     first_term = mu_R / mu_S
    #     second_term = first_term * (ES2 / mu_S ** 2)
    #     third_term = np.mean(S * R) / mu_S ** 2

    #     approx = np.sqrt(first_term + second_term - third_term)

    res = dict()
    #     res['E[norm_dY/norm_Y]'] = np.mean(norm_ratio)

    #     res['mu_S'] = mu_S
    #     res['mu_R'] = mu_R
    #     res['theta'] = np.mean(theta)

    #     res['Three-term approx'] = approx
    #     res['mu_R/mu_S'] = first_term
    #     res['E[S^2]mu_R/mu_S^3'] = second_term
    #     res['E[SR]/mu_S^2'] = third_term
    #     res['E[S^2]'] = ES2

    if compute_dimension:
        # Y = torch.from_numpy(Y).float().to(device)
        Y_centered = Y - torch.mean(Y, axis=0)
        # l, _ = np.linalg.eig(np.dot(Y_centered.T, Y_centered))
        # u, s, vh = np.linalg.svd(Y_centered, full_matrices=False)
        # l = s**2
        # dim = np.sum(l)**2/np.sum(l**2)
        C = torch.mm(torch.transpose(Y_centered, 0, 1), Y_centered) / Y.shape[
            0]

        C2 = C ** 2

        E_C2ii = torch.sum(torch.diag(C2))
        E_Cii = torch.mean(torch.diag(C))
        E_C2ij = torch.sum(C2) - E_C2ii
        E_C2ii = (E_C2ii / C.shape[0])
        E_C2ij = (E_C2ij / (C.shape[0] * (C.shape[0] - 1)))

        E_C2ii = E_C2ii.cpu().numpy()
        E_Cii = E_Cii.cpu().numpy()
        E_C2ij = E_C2ij.cpu().numpy()

        dim = n_kc * E_Cii ** 2 / (E_C2ii + (n_kc - 1) * E_C2ij)

        res['E_Cii'] = E_Cii
        res['E_C2ii'] = E_C2ii
        res['E_C2ij'] = E_C2ij
        res['dim'] = dim

    return res


if __name__ == '__main__':
    pass
