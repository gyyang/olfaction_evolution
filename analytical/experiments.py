import os
import sys
from collections import defaultdict
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt


rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

from analytical.numerical_test import simulation
import tools

def get_optimal_K(x_name, x_vals, fnames, y_name='theta', n_rep=100, update_params=None):
    """Get optimal K.

    Args:
        x_name: name of independent variable
        x_vals: values of x
        fnames: filenames
        y_name: name of dependent variable
        n_rep: number of repetitions
    """
    def guess_optimal_K(params):
        """Get the optimal K based on latest estimation."""
        if y_name == 'theta':
            return np.exp(-0.75) * (params['n_pn'] ** 0.706)
        elif y_name == 'dim':
            return np.exp(0.568) * (params['n_pn'] ** 0.2)
        else:
            raise ValueError('Unknown y_name: ' + str(y_name))

    params = {'x_dist': 'gaussian',
              'normalize_x': False,
              'w_mode': 'exact',
              # 'w_mode': 'bernoulli',
              'w_dist': 'gaussian',
              'normalize_w': False,
              'b_mode': 'percentile',
              # 'b_mode': 'gaussian',
              'coding_level': 10,
              'activation': 'relu',
              # 'perturb_mode': 'multiplicative',
              'perturb_mode': 'additive',
              'perturb_dist': 'gaussian',
              'n_pts': 500,
              'n_kc': 2500,
              'n_rep': 1,
              'n_pn': 50}

    if update_params is not None:
        params.update(update_params)

    for x_val, fname in zip(x_vals, fnames):
        params[x_name] = x_val

        K_mid = int(guess_optimal_K(params))
        min_K = max(1, int(K_mid) - 10)
        K_sim = np.arange(min_K, K_mid + 10)

        all_values = list()
        for i in range(n_rep):
            start_time = time.time()
            values_sim = defaultdict(list)
            for K in K_sim:
                if y_name == 'theta':
                    res = simulation(K, **params)
                else:
                    res = simulation(K, compute_dimension=True, **params)
                for key, val in res.items():
                    values_sim[key].append(val)
            all_values.append(values_sim)
            print('Time taken : {:0.2f}s'.format(time.time() - start_time))

        pickle.dump(all_values, open(fname, "wb"))


def get_optimal_K_simulation():
    x_name = 'n_pn'
    x_vals = [50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    fnames = list()
    for x_val in x_vals:
        fname = 'all_value_m' + str(x_val) + '.pkl'
        fnames += [os.path.join(rootpath, 'files', 'analytical', fname)]
    get_optimal_K(x_name, x_vals, fnames, n_rep=100)


def get_optimal_K_simulation_participationratio():
    x_name = 'n_pn'
    x_vals = [50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000]
    update_params = {'n_pts': 5000}
    fnames = list()
    for x_val in x_vals:
        fname = 'all_value_withdim_m' + str(x_val) + '.pkl'
        fnames += [os.path.join(rootpath, 'files', 'analytical', fname)]
    get_optimal_K(x_name, x_vals, fnames, n_rep=10, update_params=update_params)


def control_coding_level():
    m = 1000
    update_params = {'n_pn': m}
    x_name = 'coding_level'
    x_vals = [10, 20, 30, 40, 50, 60, 70, 80]
    fnames = list()
    for x_val in x_vals:
        fname = 'control_coding_level_m ' +str(m ) +'s' + str(x_val) + '.pkl'
        fnames += [os.path.join(rootpath, 'files', 'analytical', fname)]
    get_optimal_K(x_name, x_vals, fnames, n_rep=5, update_params=update_params)


def control_coding_level_vary_n_pn(compute=True):
    x_name = 'n_pn'
    x_vals = np.array([50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    coding_level = 50
    update_params = {'coding_level': coding_level}
    fnames = list()
    for x_val in x_vals:
        fname = 'all_value_s' +str(coding_level) +'_m' + str(x_val) + '.pkl'
        fnames += [os.path.join(rootpath, 'files', 'analytical', fname)]

    if compute:
        get_optimal_K(x_name, x_vals, fnames, n_rep=5, update_params=update_params)
    else:
        from analytical.analyze_simulation_results import load_result, _fit

        y = load_result(fnames, v_name='theta')
        x, y = np.log(x_vals), np.log(y)
        x_fit, y_fit, model = _fit(x, y)
        res = {'log_N': x, 'log_K': y, 'label': 'Weight robustness'}
        res_fit = {'log_N': x_fit, 'log_K': y_fit, 'model': model,
                   'label': r'$K ={:0.2f} \ N^{{{:0.2f}}}$'.format(
                       np.exp(model.intercept_), model.coef_[0])}

        figsize = (3.5, 2.)
        lbwh = [0.2, 0.2, 0.7, 0.7]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(lbwh)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        res_list = [res, res_fit]
        labels = ['Weight robustness', res_fit['label']]
        markers = ['o', '-']
        mss = [4, 4]
        zorders = [1, 0]
        colors = ['black', tools.red]
        for i, res in enumerate(res_list):
            ax.plot(res['log_N'], res['log_K'], markers[i], ms=mss[i],
                    label=labels[i], color=colors[i], zorder=zorders[i])

        ax.set_xlabel('Number of ORs (N)')
        ax.set_ylabel('Optimal K')
        xticks = np.array([50, 100, 200, 500, 1000])
        ax.set_xticks(np.log(xticks))
        ax.set_xticklabels([str(t) for t in xticks])
        yticks = np.array([3, 10, 30, 100])
        ax.set_yticks(np.log(yticks))
        ax.set_yticklabels([str(t) for t in yticks])
        # ax.set_xlim([np.log(50/1.1), np.log(1000*1.1)])
        ax.legend(bbox_to_anchor=(0., 1.05), loc=2, frameon=False)
        fname = 'optimal_k_simulation_all_part2'
        fname = os.path.join(rootpath, 'figures', 'analytical', fname)
        # plt.savefig(fname + '.pdf', transparent=True)
        # plt.savefig(fname + '.png')


if __name__ == '__main__':
    # get_optimal_K_simulation_participationratio()
    # get_optimal_k()
    # compare_dim_plot()
    # control_coding_level()
    control_coding_level_vary_n_pn(compute=False)