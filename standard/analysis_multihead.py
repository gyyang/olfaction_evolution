import sys
import os
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors.kde import KernelDensity

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)  # This is hacky, should be fixed

import tools
import standard.analysis_pn2kc_training as analysis_pn2kc_training
from analysis import _easy_save

mpl.rcParams['font.size'] = 7

path = os.path.join(rootpath, 'files', 'multi_head')
figpath = os.path.join(rootpath, 'figures', 'multi_head')

analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True)

d = os.path.join(path, '000000', 'epoch')
# Load results from last epoch
wout1 = tools.load_pickle(d, 'model/layer3/kernel:0')[-1]
wout2 = tools.load_pickle(d, 'model/layer3_2/kernel:0')[-1]
wglo = tools.load_pickle(d, 'w_glo')[-1]

# Compute sparsity
thres = analysis_pn2kc_training.infer_threshold(wglo)
sparsity = np.count_nonzero(wglo > thres, axis=0)

 
# =============================================================================
# ind_sort = np.argsort(sparsity)
# sparsity = sparsity[ind_sort]
# wout1 = wout1[ind_sort, :]
# wout2 = wout2[ind_sort, :]
# plt.figure()
# plt.imshow(wout1[:500], aspect='auto')
# plt.figure()
# plt.imshow(wout2[:500], aspect='auto')
# =============================================================================

v1 = sparsity
# strength_wout1 = np.sum(abs(wout1), axis=1)
# strength_wout2 = np.sum(abs(wout2), axis=1)
strength_wout2 = np.linalg.norm(wout2, axis=1)

# v2 = strength_wout1
v2 = strength_wout2
# v2 = strength_wout2/(strength_wout1+strength_wout2)

xlabel = 'PN Input degree'
ylabel = 'Conn. strength to valence'

fig = plt.figure(figsize=(1.5, 1.5))
plt.scatter(v1, v2, alpha=0.3)

data = np.stack([v1, v2]).T
norm_factor = data.mean(axis=0)
data_norm = data / norm_factor


def _compute_silouette_score(data):
    n_clusters = np.arange(2, 10)
    scores = list()
    for n in n_clusters:
        labels = KMeans(n_clusters=n, random_state=0).fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append(score)
        
    fig = plt.figure(figsize=(1.5, 1.5))
    plt.plot(n_clusters, scores, 'o-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silouette score')
    
_compute_silouette_score(data_norm)

labels = KMeans(n_clusters=2, random_state=0).fit_predict(data_norm)
    
fig = plt.figure(figsize=(1.5, 1.5))
plt.scatter(v1[labels == 0], v2[labels == 0], alpha=0.02)
plt.scatter(v1[labels == 1], v2[labels == 1], alpha=0.3)
plt.xlabel(xlabel)
plt.ylabel(ylabel)


xmin, xmax, ymin, ymax = 0, 15, 0, 3
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.stack([X.ravel(), Y.ravel()]).T

# positions /= norm_factor
# data /= norm_factor

def _get_density(data, method='scipy'):
    """Get density of data.
    
    Args:
        data: array (n_samples, n_features)
    """
    if method == 'scipy':
        kernel = stats.gaussian_kde(data.T)
        Z = np.reshape(kernel(positions.T), X.shape)
    elif method == 'sklearn':
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)
        Z = np.reshape(np.exp(kde.score_samples(positions)), X.shape)
    else:
        raise ValueError('Unknown method')
    return Z

Z = _get_density(data)
Z1 = _get_density(data[labels==0])
Z2 = _get_density(data[labels==1])

def _plot_density(Z, savename):
    fig = plt.figure(figsize=(1.5, 1.5))
    ax = fig.add_axes([0.25, 0.25, 0.7, 0.7])
    ax.plot([7, 7], [ymin, ymax], '--', color='gray', linewidth=1)
    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
              extent=[xmin, xmax, ymin, ymax], aspect='auto')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xticks([0, 7, 15])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.show()
    _easy_save(figpath, savename)
    
    
_plot_density(Z, 'density')
_plot_density(Z1, 'density_group1')
_plot_density(Z2, 'density_group2')
_plot_density(Z1+Z2, 'density_group12')