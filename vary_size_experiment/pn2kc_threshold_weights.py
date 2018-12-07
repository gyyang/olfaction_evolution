import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools
import utils
from scipy.optimize import curve_fit
from scipy.misc import factorial

param = "kc_inputs"
condition = "test/nodup_loss"

mpl.rcParams['font.size'] = 7
fig_dir = os.path.join(os.getcwd(), condition, 'figures')
dir = os.path.join(os.getcwd(), condition, 'files')

configs, glo_score, val_acc, val_loss, train_loss = utils.load_results(dir)
parameters = [getattr(config, param) for config in configs]
list_of_legends = [param +': ' + str(n) for n in parameters]
data = [glo_score, val_acc, val_loss, train_loss]

wglo = utils.load_pickle(dir, 'w_glo')

def exponential(x, mu, lamb, A):
    return A*np.exp(-(x-mu) * lamb)

def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def poisson(k, lamb, A):
    return A * (lamb**k/ factorial(k)) * np.exp(-lamb)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1) + gauss(x,mu2,sigma2,A2)

def exponential_gauss(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return exponential(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)

def estimate_parameters(model, x, y):
    params, cov = curve_fit(model, x, y)
    sigma = np.sqrt(np.diagonal(cov))
    return params, sigma

range = [0.03, 1]
bins = 100
for w in wglo:
    data = w.flatten()
    y, x = np.histogram(data, bins=bins, range=range)
    x = [(a+b)/2 for a, b in zip(x[:-1],x[1:])]
    guess = [0, 0, 0, 0, 0, 0]
    params, sigma = estimate_parameters(gauss, x, y)
    print(params)
    plt.plot(x,gauss(x,*params[:3]),color='red',lw=3,label='model')
    # plt.plot(x,gauss(x,*params[-3:]),color='green',lw=3,label='model')
    plt.hist(data,bins=bins, range=range)
    plt.show()

#
# nr = 8
# skip = 4
# fig, ax = plt.subplots(nrows=nr, ncols=3)
# thres = 0.15
# for i, (l, cur_w) in enumerate(zip(list_of_legends[::skip], wglo[::skip])):
#     ax[i,0].hist(cur_w.flatten(), bins=100, range= (0, thres))
#     # ax[i,0].set_title(l)
#     ax[i,1].hist(cur_w.flatten(), bins=100, range= (thres, 1))
#     # ax[i,1].set_title(l)
#     sparsity = np.count_nonzero(cur_w > thres, axis=0)
#     ax[i,2].hist(sparsity, bins=20, range= (0, 20))
#     # ax[i,2].set_title('sparsity')
# plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, 'weight_distribution.png'))
