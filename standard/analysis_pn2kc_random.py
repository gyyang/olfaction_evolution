import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.patches as patches
from tools import save_fig
import standard.analysis_weight as analysis_weight

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)
fig_dir = os.path.join(rootpath, 'figures')

THRES = .03
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 7
mpl.rcParams['font.family'] = 'arial'

def _shuffle(w_binary, arg):
    '''Shuffles the connections in numpy

    this function returns the shuffled data using different methods

    Args:
        w_binary: connection matrix in binary format (1 = connection, 0 = no connection)
        arg: method of shuffling
        arg == 'random'
            first the overall connection probability P is computed, and every connection has a P
            probability of being a made.
        arg == 'preserve'
            randomly shuffles while preserving the distribution of claw counts and the distribution of
            pns that kcs sample from
        '''
    if arg == 'random':
        P = np.mean(w_binary.flatten())
        shuffled = np.random.uniform(size=[w_binary.shape[0], w_binary.shape[1]]) < P
    elif arg == 'preserve':
        n_pns, n_kcs = w_binary.shape
        connections_per_kc = np.sum(w_binary, axis=0)
        probability_per_pn = np.sum(w_binary, axis=1) / np.sum(w_binary)

        shuffled = np.zeros_like(w_binary)
        j = 0
        for i in range(n_kcs):
            n_connections = connections_per_kc[i]
            ix_pns = np.random.choice(n_pns, size=n_connections, replace=False, p=probability_per_pn)
            shuffled[ix_pns, i] = 1
            j+= n_connections
    else:
        raise ValueError('Unknown shorting method {:s}'.format(arg))
    return shuffled

def _extract_paircounts(mat):
    n_pn = mat.shape[0]
    n_kc = mat.shape[1]
    counts_matrix = np.zeros((n_pn, n_pn))
    for kc in range(n_kc):
        vec = mat[:, kc]
        ix = np.nonzero(vec)[0]
        for i in ix:
            for j in ix:
                counts_matrix[i, j] += 1
    lower = np.tril(counts_matrix, k=-1)
    counts = lower[lower>0]
    return counts, counts_matrix

def _get_claws(dir, ix = 0):
    dirs = [os.path.join(dir, n) for n in os.listdir(dir)]
    wglos = tools.load_pickle(os.path.join(dirs[ix], 'epoch'), 'w_glo')
    wglo_binaries = []
    for i, wglo in enumerate(wglos):
        if i == 0:
            thres = THRES
        else:
            thres = analysis_weight.infer_threshold(wglos[i])

        wglo[np.isnan(wglo)] = 0
        wglo_binaries.append(wglo > thres)
        wglos[i] = wglo
    return wglo_binaries, wglos

#frequency of identical pairs vs shuffled
def pair_distribution(dir, dir_ix, shuffle_arg):
    bin_range = 150
    wglo_binaries, _ = _get_claws(dir, dir_ix)
    wglo_binary = wglo_binaries[-1]

    trained_counts, trained_counts_matrix = _extract_paircounts(wglo_binary)

    n_shuffle = 100
    shuffled_counts_matrix = np.zeros((n_shuffle, bin_range))
    for i in range(n_shuffle):
        shuffled_wglo_binary = _shuffle(wglo_binary, arg= shuffle_arg)
        shuffled_counts, _ = _extract_paircounts(shuffled_wglo_binary)
        y, _ = np.histogram(shuffled_counts, bins=bin_range, range=[0,bin_range], density=True)
        shuffled_counts_matrix[i,:] = y

    shuffled_mean = np.mean(shuffled_counts_matrix, axis=0)
    shuffled_std = np.std(shuffled_counts_matrix, axis=0)

    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    xrange = bin_range
    yrange = 0.1
    xticks = np.arange(0, bin_range, 20)
    yticks = np.linspace(0, yrange, 3)
    legends = ['Trained','Shuffled']

    plt.hist(trained_counts, bins=bin_range, range=[0,bin_range], density=True, alpha = .5)
    plt.errorbar(range(bin_range), shuffled_mean, shuffled_std, elinewidth=.75, linewidth=.75)
    ax.legend(legends, loc=1, bbox_to_anchor=(1.05, 0.4), fontsize=5)
    ax.set_xlabel('Number of KCs')
    ax.set_ylabel('Fraction of Pairs')
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    plt.ylim([0, yrange])
    plt.xlim([0, xrange])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    save_fig(dir, '_' + str(dir_ix) + '_pair_distribution_' + shuffle_arg)

# distribution of connections is not a bernoulli distribution, but is more compact
def claw_distribution(dir, dir_ix, shuffle_arg):
    wglo_binaries, _ = _get_claws(dir, dir_ix)
    wglo_binary = wglo_binaries[-1]
    sparsity = np.count_nonzero(wglo_binary > 0, axis= 0)

    shuffle_factor = 50
    shuffled = []
    for i in range(shuffle_factor):
        shuffled_wglo_binary = _shuffle(wglo_binary, arg=shuffle_arg)
        shuffled.append(shuffled_wglo_binary)
    shuffled = np.concatenate(shuffled, axis= 1)
    shuffled_sparsity = np.count_nonzero(shuffled > 0, axis= 0)

    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    xrange = 20
    yrange = .4
    xticks = [1, 5, 7, 10, 15, 20]
    yticks = np.linspace(0, yrange, 3)
    legends = ['Trained','Shuffled']

    plt.hist(sparsity, bins=xrange, range= (0, xrange), alpha= .5, density=True, align='left')
    plt.hist(shuffled_sparsity,bins=xrange, range= (0, xrange), alpha=.5, density=True, align='left')
    ax.legend(legends, loc=1, bbox_to_anchor=(1.05, 0.4), fontsize=5)
    ax.set_xlabel('Claws per KC')
    ax.set_ylabel('Fraction of KCs')
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    plt.ylim([0, yrange])
    plt.xlim([0, xrange])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    save_fig(dir, '_' + str(dir_ix) + '_claw_distribution_' + shuffle_arg)


#all PNs make the same number of connections onto KCs
def plot_distribution(dir, dir_ix):
    wglo_binaries, _ = _get_claws(dir, dir_ix)
    wglo_binary = wglo_binaries[-1]

    weights_per_pn = np.mean(wglo_binary, axis=1)
    p_connection = np.mean(wglo_binary.flatten())

    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    xrange = len(weights_per_pn)
    yrange = 0.2
    plt.bar(range(xrange), weights_per_pn)
    ax.plot([-1, xrange], [p_connection, p_connection], '--', color='gray')

    ax.set_xlabel('PN Identity')
    ax.set_ylabel('KC Connection Probability')

    xticks = [0, 9, 19, 29, 39, 49]
    yticks = np.linspace(0, .2, 3)
    ax.set_xticks(xticks)
    ax.set_xticklabels([x + 1 for x in xticks])
    ax.set_yticks(yticks)
    plt.ylim([0, yrange])
    plt.xlim([-1, xrange + 1])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    save_fig(dir, '_' + str(dir_ix) + '_pn_distribution')

# average correlation of weights between KCs decrease as a function of training
# and is similar to shuffled weights with the same connection probability
def plot_cosine_similarity(dir, dir_ix, shuffle_arg, log= True):
    def _get_similarity(mat):
        similarity_matrix = cosine_similarity(mat)
        diag_mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        corrs = similarity_matrix[diag_mask]
        average_correlation = np.mean(corrs)
        return average_correlation, similarity_matrix

    wglo_binaries, wglos = _get_claws(dir, ix = dir_ix)
    y = []
    for wglo in wglo_binaries:
        corr, similarity_matrix = _get_similarity(np.transpose(wglo))
        y.append(corr)

    n_shuffle = 3
    y_shuffled = []
    for j in range(len(wglo_binaries)):
        shuffled_similarities = []
        for i in range(n_shuffle):
            if j == 0:
                thres = 0
            else:
                thres = analysis_weight.infer_threshold(wglos[j])
            shuffled = _shuffle(wglo_binaries[j]>thres, arg=shuffle_arg)
            shuffled_similarity, _ = _get_similarity(shuffled)
            shuffled_similarities.append(shuffled_similarity)
        temp = np.mean(shuffled_similarities)
        y_shuffled.append(temp)

    legends = ['Trained', 'Shuffled']
    figsize = (2, 1.5)
    rect = [0.3, 0.3, 0.65, 0.5]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)

    if log == True:
        y = -np.log(y)
        y_shuffled = -np.log(y_shuffled)
        yticks = [0, 1, 2, 3]
        ylim = [0, 3]
    else:
        yticks = [0, .5, 1]
        ylim = [0, 1]
    xlim = len(y)
    ax.plot(y)
    ax.plot(range(xlim), y_shuffled, '--', color='gray')
    ax.legend(legends, fontsize=7, frameon=False)
    xticks =np.arange(0, xlim, 10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cosine Similarity')
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_ylim(ylim)
    ax.set_xlim([0, len(y)-1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    save_fig(dir, '_' + str(dir_ix) + '_cosine_similarity_' + shuffle_arg, dpi=500)

def display_matrix(wglo):

    wglo[np.isnan(wglo)] = 0
    thres = analysis_weight.infer_threshold(wglo)
    wglo_binary = wglo > thres
    trained_counts, trained_counts_matrix = _extract_paircounts(wglo_binary)
    lower = np.tril(trained_counts_matrix, k=-1)

    plt.imshow(lower)
    # plt.imshow(lower, cmap=plt.cm.get_cmap('jet',10))
    plt.colorbar()

    def rect(pos):
        r = plt.Rectangle(pos, 1, 1, facecolor="none", edgecolor="w", linewidth=2)
        plt.gca().add_patch(r)
    for i in range(50):
        rect([i-.5,i-.5])

    cmap = plt.get_cmap()
    colors = cmap.colors
    colors[0] = [0, 0, 0]
    cmap.colors = colors
    plt.set_cmap(cmap)
    plt.show()

