import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools
from tools import nicename
import utils
from scipy.optimize import curve_fit
from scipy.misc import factorial


def plot_sparsity(data, savename, title, xrange=50, yrange= .5):
    fig = plt.figure(figsize=(2.5, 2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    plt.hist(data, bins=xrange, range=[0, xrange], density=True)
    ax.set_xlabel('PN inputs per KC')
    ax.set_ylabel('Fraction of KCs')
    name = title
    ax.set_title(name)

    xticks = [1, 7, 15, 25, 50]
    ax.set_xticks([x - .5 for x in xticks])
    ax.set_xticklabels([str(x) for x in xticks])
    ax.set_yticks(np.linspace(0, yrange, 3))
    plt.ylim([0, yrange])
    plt.xlim([0, xrange])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.tight_layout()
    plt.savefig(savename + '.png', dpi=300)

def plot_distribution(data, savename, title, xrange, yrange):
    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    plt.hist(data, bins=50, range=[0, xrange], density=False)
    ax.set_xlabel('PN to KC Weight')
    ax.set_ylabel('Number of Connections')
    name = title
    ax.set_title(name)

    xticks = [0, .2, .4, .6, .8, 1]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    yticks = [0, 1000, 2000, 3000, 4000, 5000]
    yticklabels = ['0', '1K', '2K', '3K', '4K', '>100K']
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    plt.ylim([0, yrange])
    plt.xlim([0, xrange])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.tight_layout()
    plt.savefig(savename + '.png', dpi=300)

def plot_progress(ys, save_name, legends):
    # Validation accuracy
    figsize = (2, 2)
    rect = [0.3, 0.3, 0.65, 0.5]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)

    cs = ['r','g','b']
    ss = [':','-.','-']
    for y, c, s in zip(ys, cs, ss):
        ax.plot(y, alpha= .75, linestyle=s)

    ax.legend(legends, loc=1, bbox_to_anchor=(1.05, 0.4), fontsize=5)

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Validation accuracy')
    # plt.title('Final accuracy {:0.3f}'.format(y[-1]), fontsize=7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks(np.arange(0, y[-1]+2, 10))
    ax.yaxis.set_ticks([0, 0.5, 1.0])
    ax.set_ylim([0, 1])
    ax.set_xlim([0, len(y)-1])
    plt.tight_layout()
    plt.savefig(save_name + '.png', dpi=300)


condition = "vary_KC_training"
condition = "random"

mpl.rcParams['font.size'] = 7
fig_dir = os.path.join(os.getcwd(), condition, 'figures')
dir = os.path.join(os.getcwd(), condition, 'files')
dirs = [os.path.join(dir, n) for n in os.listdir(dir)]

configs, glo_score, val_acc, val_loss, train_loss = utils.load_results(dir)
save_name = os.path.join(fig_dir, 'val_acc')
plot_progress(val_acc, save_name, ['Trainable, no loss', 'Trainable, with loss', 'Fixed'])

thres = 0.05
titles = ['Before Training', 'After Training']
yrange = [1, 0.5]
for i, d in enumerate(dirs):
    wglo = utils.load_pickle(os.path.join(d,'epoch'), 'w_glo')
    wglo = [wglo[0]] + [wglo[-1]]

    for j, w in enumerate(wglo):
        sparsity = np.count_nonzero(w > thres, axis=0)
        save_name = os.path.join(fig_dir, 'sparsity_' + str(i) + '_' + str(j))
        plot_sparsity(sparsity, save_name, title= titles[j], yrange= yrange[j])
        distribution = w.flatten()
        save_name = os.path.join(fig_dir, 'distribution_' + str(i) + '_' + str(j))
        plot_distribution(distribution, save_name, title= titles[j], xrange= 1.0, yrange = 5000)



def approximate_distribution():
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



# configs, glo_score, val_acc, val_loss, train_loss = utils.load_results(dir)
# parameters = [getattr(config, param) for config in configs]
# list_of_legends = [param +': ' + str(n) for n in parameters]
# data = [glo_score, val_acc, val_loss, train_loss]