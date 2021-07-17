import os
import sys
from scipy.spatial import Voronoi
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

from schematics.plotVoronoi import voronoi_plot_2d
from tools import save_fig
from task import _sample_input

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

def _spread_orn_activity(prototypes, spread=0.):
    assert spread >= 0 and spread < 1, 'spread is not within range of [0, 1)'
    if spread == 0:
        return prototypes
    spread_low = 1 - spread
    spread_high = 1 + spread
    n_samples = prototypes.shape[0]
    scale_factors = np.random.choice([spread_low, spread_high, spread_high],
                                     size=n_samples)
    return prototypes * scale_factors.reshape(-1, 1)

def get_labels(prototypes, odors):
    dist = euclidean_distances(prototypes, odors)
    label = np.argmin(dist, axis=0)
    return label

def _normalize(x):
    norm = np.linalg.norm(x, axis=1)
    x = (x.T / norm).T
    x[np.isnan(x)] = 0
    return x


def plot_task(mode='standard', include_prototypes=False, include_data = True, 
              prototype_marker = '^', meta_ix=0, spread=0):
    """Plot task schematic.
    
    Args:
        mode: str, standard, innate, innate2, concentration, metalearn, relabel
    
    """
    np.random.seed(2)
    colors = ['c','y','m','g']
    
# =============================================================================
#     colors = [np.array([81, 81, 96])/255.,
#               np.array([104, 130, 158])/255.,
#               np.array([174, 189, 56	])/255.,
#               np.array([89, 130, 52])/255.]
# =============================================================================
    colors = [np.array([55, 94, 151])/255.,  # blue
              np.array([251, 101, 66])/255.,  # orange
              np.array([255, 187, 0])/255.,  # red
              np.array([63, 104, 28])/255.,]  # green

    # From colorbrewer qualitative map
    colors = np.array([[102,194,165],[252,141,98],[141,160,203],
                       [231,138,195],[166,216,84], [255, 217, 47]])/255.

    size = 80
    figsize = (1.8, 1.8)
    ax_dim = [.2, .2, .7, .7]
    if mode == 'standard':
        proto_points = np.array([[2, 4], [4, 3], [3, 2],[1, 1]])
        texts = ['Class ' + i for i in ['A','B','C','D']]
        lim = 5
    elif mode == 'concentration':
        proto_points_ = np.array([[1, 4], [5, 3], [3, 3], [3, .5], [1.5, 2.5]])
        proto_points = _normalize(proto_points_)
        ind = [0, 0, 2, 1, 1]
        labels = ['A', 'B', 'C']
        colors = [colors[i] for i in ind]
        texts = ['Class ' + labels[i] for i in ind]
        lim = 5
        size = 150
    elif mode == 'relabel':
        proto_points = np.array([[1, 1], [1.5, 3], [2.5, 4], [2.5, 2.5], [4, 1], [4, 3]])
        ind = [0, 1, 2, 3, 2, 1]
        labels = ['A','B','C','D']
        colors = [colors[i] for i in ind]
        texts = ['Class ' + labels[i] for i in ind]
        lim = 5
    elif mode == 'metalearn':
        proto_points = [[[2, 4], [4, 3], [10, 10]], 
                        [[2, 2], [3, 4], [8, 6]], 
                        [[4, 1], [3, 2], [8, 8]]]
        proto_points = proto_points[meta_ix]
        texts = ['Class A', 'Class B', 'Class C']
        lim = 5
        size = 100
        figsize = (1.3, 1.3)
        ax_dim = [.2, .2, .6, .6]
    elif 'innate' in mode:
        innate_point = np.array([[1, 0]])
        innate_point2 = np.array([[0, 1]])
        neutral_proto_points = np.array([[1, 1], [1.5, 3], [2.5, 4], [2.5, 2.5], 
                                         [4, 1], [4, 3]])/5.
        proto_points = np.concatenate((neutral_proto_points, 
                                       innate_point, innate_point2))
        ind = [0, 1, 2, 3, 2, 1, 4, 5]
        labels = ['A','B','C','D', 'E', 'F']
        colors = [colors[i] for i in ind]
        texts = ['Class ' + labels[i] for i in ind]
        lim = 2
        if mode == 'innate2':
            colors = ([np.array([178]*3)/255.] * 6 + 
            [np.array([228, 26, 28])/255.] + [np.array([55,126,184])/255.])
    elif mode == 'correlate':
        orn_corr = 0.8
        rng = np.random.RandomState(seed=1)
        proto_points = _sample_input(3, 2, rng=rng, corr=orn_corr) * 5
        texts = ['Class ' + i for i in ['A','B','C','D']]
        lim = 5
    else:
        raise ValueError('Unknown mode: ', mode)

    if 'innate' in mode:
        rand_neutral_points = np.random.uniform(low=0, high=1, size=[size, 2])
        rand_innate_points = innate_point+np.random.uniform(low=0, high=1, size=[20, 2])
        rand_innate_points2 = innate_point2+np.random.uniform(low=0, high=1, size=[20, 2])
        rand_points = np.concatenate((rand_neutral_points, 
                                      rand_innate_points, rand_innate_points2), axis=0)
    elif mode == 'correlate':
        rand_points = _sample_input(size, 2, rng=rng, corr=orn_corr) * 5
    elif mode == 'concentration':
        rand_points = np.random.uniform(low=0, high=lim, size=[size, 2])
        rand_points = _spread_orn_activity(rand_points, spread=spread)
    else:
        rand_points = np.random.uniform(low=0, high=lim, size=[size, 2])
    
    if mode == 'concentration':
        rand_labels = get_labels(proto_points, _normalize(rand_points))
    elif 'innate' in mode:
        rand_neutral_labels = get_labels(neutral_proto_points, rand_neutral_points)
        rand_labels = np.concatenate((rand_neutral_labels,
                                     [6]*len(rand_innate_points), 
                                     [7]*len(rand_innate_points2)))
        print(rand_labels)
    else:
        rand_labels = get_labels(proto_points, rand_points)
        
    if mode == 'metalearn':
        # Choose only 5 per class
        inds = np.concatenate((np.where(rand_labels==0)[0][:5], 
                              np.where(rand_labels==1)[0][:5]))
        rand_points = rand_points[inds]
        rand_labels = rand_labels[inds]

    rand_colors = [colors[i] for i in rand_labels]
    
    #plotting
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(ax_dim)
    plt.sca(ax)

    if mode not in ['innate', 'innate2', 'correlate']:
        vor = Voronoi(proto_points)
        voronoi_plot_2d(vor,
                        ax=ax,
                        show_vertices = False,
                        show_points= False,
                        line_colors='k',
                        line_width=1)

    if mode == 'concentration':
        proto_points *= lim * 0.9
        

    if include_data:
        s = 1. if mode == 'concentration' else 2
        for c,p in zip(rand_colors, rand_points):
            ax.scatter(p[0], p[1], color=c, s=s)

    if include_prototypes:
        for c,p in zip(colors, proto_points):
            ax.scatter(p[0], p[1], color=c, s=15, marker=prototype_marker,
                       edgecolor='black', linewidth=0.5)

    for i, (txt,p) in enumerate(zip(texts, proto_points)):
        if mode in ['innate', 'concentration']:
            pass
# =============================================================================
#             if i < 4:
#                 ax.annotate(txt, (p[0]-.3, p[1]-.35))
#             else:
#                 ax.annotate(txt, (p[0]+0.3, p[1]+0.3))
# =============================================================================
        elif mode == 'innate2':
            pass
        elif mode in ['metalearn', 'correlate']:
            ax.annotate(txt, (p[0], p[1]+.35), ha='center')
        else:
            ax.annotate(txt, (p[0]-.3, p[1]-.35))

    plt.axis('square')
    if mode == 'concentration':
        lim *= 2
        plt.xlim([0, lim])
        plt.ylim([0, lim])
        plt.xticks([0, lim], ['0', '2'])
        plt.yticks([0, lim], ['0', '2'])
    elif 'innate' in mode:
        plt.xlim([0, lim])
        plt.ylim([0, lim])
        plt.xticks([0, lim])
        plt.yticks([0, lim])
    else:
        plt.xlim([0, lim])
        plt.ylim([0, lim])
        plt.xticks([0, lim], ['0', '1'])
        plt.yticks([0, lim], ['0', '1'])
    plt.xlabel('OR 1 Activity', labelpad=-5)
    
    if mode == 'metalearn':
        plt.title('Training set {:d}'.format(meta_ix+1), fontsize=7)
    if mode != 'metalearn' or meta_ix == 0:
        plt.ylabel('OR 2 Activity', labelpad=-5)
    if mode == 'correlate':
        plt.title('Correlation {:0.1f}'.format(orn_corr), fontsize=7)
    if mode == 'metalearn':
        name_str = '_' + str(meta_ix)
    if mode == 'concentration':
        name_str = '_spread {:0.1f}'.format(spread)
        plt.title(name_str[1:].capitalize(), fontsize=7)

    else:
        name_str = ''

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    save_fig('schematics', '_' + mode + '_task' + name_str)
    # plt.savefig('task.pdf',transparent=True)
    

if __name__ == '__main__':
    plot_task('standard', include_prototypes=True)
    # plot_task('innate', include_prototypes=True)
    # plot_task('innate2', include_prototypes=True)
    # plot_task('concentration', include_prototypes=True, include_data=True, spread=0)
    # plot_task('concentration', include_prototypes=True, include_data=True, spread=0.6)
    # plot_task('relabel', include_prototypes=True)
    # [plot_task('metalearn', include_prototypes=True, meta_ix=i) for i in range(3)]
    # plot_task('correlate', include_prototypes=True)