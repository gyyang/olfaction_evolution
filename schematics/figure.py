import os
import sys
for p in sys.path:
    print(p)
from scipy.spatial import Voronoi
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

from schematics.plotVoronoi import voronoi_plot_2d
from standard.analysis import _easy_save

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

def get_labels(prototypes, odors):
    dist = euclidean_distances(prototypes, odors)
    label = np.argmin(dist, axis=0)
    return label

def _normalize(x):
    norm = np.linalg.norm(x, axis=1)
    x = (x.T / norm).T
    x[np.isnan(x)] = 0
    return x


def plot_task(mode='standard', include_prototypes=False, include_data = True, prototype_marker = '^', meta_ix = 0):
    colors = ['c','y','m','g']
    
    # =============================================================================
    # colors = [np.array([81, 81, 96])/255.,
    #           np.array([104, 130, 158])/255.,
    #           np.array([174, 189, 56	])/255.,
    #           np.array([89, 130, 52])/255.]
    # =============================================================================
    colors = [np.array([55, 94, 151])/255.,  # blue
              np.array([251, 101, 66])/255.,  # orange
              np.array([255, 187, 0])/255.,  # red
              np.array([63, 104, 28])/255.,]  # green

    size = 80
    if mode == 'standard':
        proto_points = np.array([[2, 4], [4, 3], [3, 2],[1, 1]])
        texts = ['Class ' + i for i in ['A','B','C','D']]
        lim = 5
    elif mode == 'concentration':
        proto_points = np.array([[2, 4], [4, 3], [3, .2],[1, 1]])
        proto_points = _normalize(proto_points)
        texts = ['Class ' + i for i in ['A','B','C','D']]
        lim = 2
    elif mode == 'relabel':
        proto_points = np.array([[1, 1], [1.5, 3], [2.5, 4], [2.5, 2.5], [4, 1], [4, 3]])
        ind = [0, 1, 2, 3, 2, 1]
        labels = ['A','B','C','D']
        colors = [colors[i] for i in ind]
        texts = ['Class ' + labels[i] for i in ind]
        lim = 5
    elif mode == 'metalearn':
        proto_points = [[[2, 4], [4, 3], [10, 10]], [[2, 2], [3, 4], [8, 6]], [[4, 1], [3, 2], [8, 8]]]
        proto_points = proto_points[meta_ix]
        texts = ['Class A', 'Class B', 'Class C']
        lim = 5
        size = 10
    else:
        raise ValueError('Unknown mode: ', mode)

    rand_points = np.random.uniform(low=0, high=lim, size=[size, 2])
    if mode == 'concentration':
        rand_labels = get_labels(proto_points, _normalize(rand_points))
    else:
        rand_labels = get_labels(proto_points, rand_points)

    rand_colors = [colors[i] for i in rand_labels]
    
    #plotting
    fig = plt.figure(figsize=(2.5, 1.8))
    ax = fig.add_axes([.2, .2, .7, .7])
    plt.sca(ax)

    if include_prototypes:
        for c,p in zip(colors, proto_points):
            ax.scatter(p[0], p[1], color=c, s=15, marker= prototype_marker)

    if include_data:
        for c,p in zip(rand_colors, rand_points):
            ax.scatter(p[0], p[1], color=c, s=2)
    
    for i, (txt,p) in enumerate(zip(texts, proto_points)):
        ax.annotate(txt, (p[0]-.3, p[1]-.35))

    vor = Voronoi(proto_points)
    voronoi_plot_2d(vor,
                    ax=ax,
                    show_vertices = False,
                    show_points= False,
                    line_colors='k',
                    line_width=1)

    plt.axis('square')
    plt.xlim([0, lim])
    plt.ylim([0, lim])
    plt.xticks([0, lim])
    plt.yticks([0, lim])
    plt.xlabel('ORN 1 Activity', labelpad=-5)
    plt.ylabel('ORN 2 Activity', labelpad=-5)

    if mode == 'metalearn':
        name_str = '_' + str(meta_ix)
    else:
        name_str = ''

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    _easy_save('schematics', '_' + mode + '_task' + name_str)
    # plt.savefig('task.pdf',transparent=True)
    

if __name__ == '__main__':
    # plot_task('standard', include_prototypes=False)
    # plot_task('concentration', include_prototypes=False, include_data=True)
    # plot_task('relabel')
    plot_task('metalearn', include_data=True, include_prototypes=True, meta_ix=2)