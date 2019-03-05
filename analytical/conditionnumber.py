"""Analyze condition number of the network."""

import numpy as np
import matplotlib.pyplot as plt

import model


def _get_cond(q, n_orn, n_pn, n_kc, n_kc_claw):
    M = np.random.rand(n_orn, n_pn)
    M_new = M * (1-q) + np.eye(n_orn) * q
    
    # J = np.random.rand(N_PN, N_KC) / np.sqrt(N_PN + N_KC)
    # J = np.random.randn(N_PN, N_KC) / np.sqrt(N_PN + N_KC)
    J = np.random.rand(n_pn, n_kc)
    mask = model.get_sparse_mask(n_pn, n_kc, n_kc_claw) / n_kc_claw
    J = J * mask

    K = np.dot(M_new, J)
    
    cond = np.linalg.cond(K)
    return cond


def get_logcond(q=1, n_orn=50, n_pn=50, n_kc=2500, n_kc_claw=7, n_rep=10):
    conds = [_get_cond(q, n_orn, n_pn, n_kc, n_kc_claw) for i in range(n_rep)]
    return np.mean(np.log10(conds))



def plot_cond_by_q(n_kc=2500):
    qs = np.linspace(0, 1, 100)
    conds = [get_logcond(q=q, n_kc=n_kc) for q in qs]
    
    plt.figure()
    plt.plot(qs, conds, 'o-')
    plt.title('N_KC: ' + str(n_kc))
    plt.xlabel('fraction diagonal')
    plt.ylabel('log condition number')
    # plt.savefig('figures/condvsfracdiag_nkc'+str(n_kc)+'.pdf', transparent=True)


def plot_cond_by_n_kc():
    n_kcs = np.logspace(1, 4, 10).astype(int)
    conds_q1 = np.array([get_logcond(n_kc=n_kc, q=1) for n_kc in n_kcs])
    
    plt.figure()
    plt.plot(np.log10(n_kcs), conds_q1, 'o-')
    plt.xticks(np.log10(n_kcs), n_kcs)
    plt.xlabel('N_KC')
    
    
    n_kcs = np.logspace(1, 4, 10).astype(int)
    conds_q0 = np.array([get_logcond(n_kc=n_kc, q=0) for n_kc in n_kcs])
    
    plt.figure()
    plt.plot(np.log10(n_kcs), conds_q0, 'o-')
    plt.xticks(np.log10(n_kcs), n_kcs)
    plt.xlabel('N_KC')
    
    plt.figure()
    plt.plot(np.log10(n_kcs), conds_q1 - conds_q0, 'o-')
    plt.xticks(np.log10(n_kcs), n_kcs)
    plt.ylabel('Log decrease in condition number')
    plt.xlabel('N_KC')
    
    
n_kc_claws = np.arange(1, 21)
conds = np.array([get_logcond(n_kc_claw=n) for n in n_kc_claws])

plt.figure()
plt.plot(n_kc_claws, conds, 'o-')
plt.xticks(n_kc_claws)
plt.xlabel('N_KC_claw')
