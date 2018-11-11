"""Analyze condition number of the network."""

import numpy as np
import matplotlib.pyplot as plt

import model


def _get_cond(q, n_orn, n_pn, n_kc):
    M = np.random.rand(n_orn, n_pn)
    M_new = M * (1-q) + np.eye(n_orn) * q
    
    # J = np.random.rand(N_PN, N_KC) / np.sqrt(N_PN + N_KC)
    # J = np.random.randn(N_PN, N_KC) / np.sqrt(N_PN + N_KC)
    J = np.random.rand(n_pn, n_kc)
    mask = model.get_sparse_mask(n_pn, n_kc, 7)
    J = J * mask

    K = np.dot(M_new, J)
    
    cond = np.linalg.cond(K)
    return cond


def get_logcond(q=1, n_orn=50, n_pn=50, n_kc=2500, n_rep=10):
    conds = [_get_cond(q, n_orn, n_pn, n_kc) for i in range(n_rep)]
    return np.mean(np.log10(conds))



def plot_cond_by_q(n_kc=2500):
    qs = np.linspace(0, 1, 100)
    conds = [get_logcond(q=q, n_kc=n_kc) for q in qs]
    
    plt.figure()
    plt.plot(qs, conds, 'o-')
    plt.title('N_KC: ' + str(n_kc))
    plt.xlabel('frac diagonal')
    plt.ylabel('log condition number')


plot_cond_by_q(n_kc=100)
plot_cond_by_q(n_kc=2500)


n_kcs = np.logspace(1, 4, 10).astype(int)
conds = [get_logcond(n_kc=n_kc) for n_kc in n_kcs]

plt.figure()
plt.plot(np.log10(n_kcs), conds, 'o-')
plt.xticks(np.log10(n_kcs), n_kcs)


n_kcs = np.logspace(1, 4, 10).astype(int)
conds = [get_logcond(n_kc=n_kc, q=0) for n_kc in n_kcs]

plt.figure()
plt.plot(np.log10(n_kcs), conds, 'o-')
plt.xticks(np.log10(n_kcs), n_kcs)