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
from standard.analysis import _easy_save


def get_labels(prototypes, odors):
    dist = euclidean_distances(prototypes, odors)
    label = np.argmin(dist, axis=0)
    return label


def _plot_task(episode, update):
    colors = ['c','y','m','g']
    
    # =============================================================================
    # colors = [np.array([81, 81, 96])/255.,
    #           np.array([104, 130, 158])/255.,
    #           np.array([174, 189, 56	])/255.,
    #           np.array([89, 130, 52])/255.]
    # =============================================================================
    c_dict = {'blue': np.array([55, 94, 151])/255.,  # blue
              'red': np.array([251, 101, 66])/255.,  # orange
              'orange': np.array([255, 187, 0])/255.,  # red
              'green': np.array([63, 104, 28])/255.,}  # green
    
    proto_points = np.array([[2, 4], [4, 3], [3, 1]])
    texts = ['Odor ' + i for i in ['A','B','C']]
    
    if episode == 0:
        ind = [0, 1]  # the classes to plot
        colors = [c_dict[c_name] for c_name in ['blue', 'red', 'blue']]
        if update == 'pre':
            line_y = [2, 4]
        else:
            line_y = [0, 6]
    else:
        ind = [1, 2]
        colors = [c_dict[c_name] for c_name in ['blue', 'blue', 'red']]
        if update == 'pre':
            line_y = [1.5, 3]
        else:
            line_y = [2, 1.5]
        
    seed = episode
    rng = np.random.RandomState(seed)
    
    n_points = 10
    rand_points = rng.uniform(low=0,high=5,size=[n_points*3,2])
    rand_labels = get_labels(proto_points, rand_points)
    rand_colors = [colors[i] for i in rand_labels]
    
    #plotting
    mpl.rcParams['font.size'] = 7
    fig = plt.figure(figsize=(1., 1.))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])

    for i, (c,p) in enumerate(zip(colors, proto_points)):
        if i in ind:
            ax.scatter(p[0], p[1], color=c, s=15, marker='^')
    
    for l, c, p in zip(rand_labels, rand_colors, rand_points):
        if l in ind:
            ax.scatter(p[0], p[1], color=c, s=1.0)
    
    for i, (txt,p) in enumerate(zip(texts, proto_points)):
        if i in ind:
            ax.annotate(txt, (p[0]-1.5, p[1]-0.8), fontsize=7)
    
    ax.plot([0, 5], line_y, '--', color='gray', linewidth=1)
    
    plt.axis('square')
    plt.xlim([0, 5])
    plt.ylim([0, 5])
    
    if episode == 0 and update == 'pre':
        plt.xlabel('ORN 1 Activity', labelpad=-5)
        plt.ylabel('ORN 2 Activity', labelpad=-5)
    plt.xticks([0, 5], [])
    plt.yticks([0, 5], [])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    _easy_save('maml', 'ep{:d}_{:s}update_task'.format(episode, update))
    # plt.savefig('task.pdf',transparent=True)
    
def plot_task():
    for episode in [0, 1]:
        for update in ['pre', 'post']:
            _plot_task(episode, update)

if __name__ == '__main__':
    plot_task()
    # plot_task('relabel')