import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils
from sklearn.metrics.pairwise import cosine_similarity

def get_similarity(mat):
    similarity_matrix = cosine_similarity(mat)
    diag_mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)

    #method 1
    corrs = similarity_matrix[diag_mask]

    average_correlation = np.mean(corrs)
    return average_correlation, similarity_matrix

condition = "random"
thres = .05
mpl.rcParams['font.size'] = 7
fig_dir = os.path.join(os.getcwd(), condition, 'figures')
dir = os.path.join(os.getcwd(), condition, 'files')
dirs = [os.path.join(dir, n) for n in os.listdir(dir)]
wglos = utils.load_pickle(os.path.join(dirs[1],'epoch'), 'w_glo')

#frequency of identical pairs vs shuffled
def pair_distribution(wglo):
    bin_range = 70
    n_shuffle = 100

    def extract_paircounts(mat):
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

    wglo[np.isnan(wglo)] = 0
    wglo_binary = wglo > thres
    trained_counts, trained_counts_matrix = extract_paircounts(wglo_binary)

    probability_connection = np.mean(wglo_binary.flatten())
    shuffled_counts_matrix = np.zeros((n_shuffle, bin_range))
    for i in range(n_shuffle):
        shuffled_wglo_binary = np.random.uniform(size=[wglo.shape[0], wglo.shape[1]]) < probability_connection
        shuffled_counts, _ = extract_paircounts(shuffled_wglo_binary)

        y, _ = np.histogram(shuffled_counts, bins=bin_range, range=[0,bin_range], density=True)
        shuffled_counts_matrix[i,:] = y

    shuffled_mean = np.mean(shuffled_counts_matrix, axis=0)
    shuffled_std = np.std(shuffled_counts_matrix, axis=0)

    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    xrange = bin_range
    yrange = 0.1
    xticks = np.linspace(0, bin_range, bin_range//10 + 1)
    yticks = np.linspace(0, yrange, 3)
    legends = ['Trained','Shuffled']

    save_name = os.path.join(fig_dir, 'Pair Distribution')
    plt.hist(trained_counts, bins=bin_range, range=[0,bin_range], density=True, alpha = 0.5)
    plt.errorbar(range(bin_range), shuffled_mean, shuffled_std)
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
    plt.savefig(save_name + '.png', dpi=300)

    # lower = np.tril(shuffled_counts_matrix, k=-1)
    # plt.imshow(lower)
    # plt.colorbar()
    # plt.show()
pair_distribution(wglos[-1])


# distribution of connections is not a bernoulli distribution, but is more compact
def claw_distribution(wglo):
    wglo[np.isnan(wglo)] = 0
    wglo_binary = wglo > thres
    sparsity = np.count_nonzero(wglo_binary > 0, axis= 0)

    shuffle_factor = 100
    probability_connection = np.mean(wglo_binary.flatten())
    shuffled_wglo_binary = np.random.uniform(size=[wglo.shape[0], wglo.shape[1] * shuffle_factor]) < probability_connection
    shuffled_sparsity = np.count_nonzero(shuffled_wglo_binary > 0, axis= 0)

    save_name = os.path.join(fig_dir, 'Claw Distribution')
    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    xrange = 20
    yrange = .4
    xticks = [1, 5, 7, 10, 15, 20]
    yticks = np.linspace(0, yrange, 3)
    legends = ['Trained','Shuffled']

    plt.hist(sparsity, bins=xrange, range= (0, xrange), alpha= .5, density=True, align='left')
    plt.hist(shuffled_sparsity,bins=20, range= (0, xrange), alpha=.5, density=True, align='left')
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
    plt.savefig(save_name + '.png', dpi=300)
claw_distribution(wglos[-1])


#all PNs make the same number of connections onto KCs
def plot_distribution(wglo):
    wglo[np.isnan(wglo)] = 0
    wglo_binary = wglo > thres
    weights_per_pn = np.mean(wglo_binary, axis=1)
    savename = os.path.join(fig_dir, 'PN distribution')
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

    plt.savefig(savename + '.png', dpi=300)
plot_distribution(wglos[-1])

# average correlation of weights between KCs decrease as a function of training
# and is similar to shuffled weights with the same connection probability
def plot_cosine_similarity(wglos):
    y = []
    for wglo in wglos:
        n_nans = np.sum(np.isnan(wglo))
        wglo[np.isnan(wglo)] = 0
        mask = wglo > thres
        wglo *= mask
        print('There are %d NaNs in wglo' % n_nans)
        corr, similarity_matrix = get_similarity(np.transpose(wglo))
        y.append(corr)

    n_shuffle = 10
    shuffled_similarities = []
    for i in range(n_shuffle):
        wglo_binary = wglo[-1].flatten() > thres
        probability_connection = np.mean(wglo_binary)
        shuffled = np.random.uniform(size=wglo.shape) < probability_connection
        shuffled_similarity, _ = get_similarity(shuffled)
        shuffled_similarities.append(shuffled_similarity)
    y_shuffled = np.mean(shuffled_similarities)
    save_name = os.path.join(fig_dir, 'Cosine Similarity')
    legends = ['Trained', 'Shuffled']

    figsize = (2, 2)
    rect = [0.3, 0.3, 0.65, 0.5]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)

    ax.plot(y)
    ax.plot([0, 20], [y_shuffled, y_shuffled], '--', color='gray')
    ax.legend(legends, loc=1, bbox_to_anchor=(1.05, 0.4), fontsize=5)
    xticks = [0, 5, 10, 15]
    yticks = np.linspace(0, 1, 5)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cosine Similarity')
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, len(y)-1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.savefig(save_name + '.png', dpi=300)
plot_cosine_similarity(wglos)



#
#
#
# nr, nc = 3, 2
# fig, ax = plt.subplots(nrows=nr, ncols = nc)
# for i, d in enumerate(dirs):
#     wglo = utils.load_pickle(os.path.join(d,'epoch'), 'w_glo')
#     mask = wglo[0] > 0
#     data = wglo[0][mask].flatten()
#     ax[i,0].hist(data, bins=100, range=(0, .5))
#     changes = (wglo[-1][mask] - wglo[0][mask]).flatten()
#     ax[i,1].hist(changes, bins=100, range=(-1.5, 1.5))
# plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, 'kc_weight_changes.png'))

