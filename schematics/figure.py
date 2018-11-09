from scipy.spatial import Voronoi
from schematics.plotVoronoi import voronoi_plot_2d
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_labels(prototypes, odors):
    dist = euclidean_distances(prototypes, odors)
    label = np.argmin(dist, axis=0)
    return label


proto_points = np.array([[2, 4], [4, 3], [3, 2],[1, 1]])
colors = ['c','y','m','g']
texts = ['Odor ' + i for i in ['A','B','C','D']]
rand_points = np.random.uniform(low=0,high=5,size=[80,2])
rand_labels = get_labels(proto_points, rand_points)
rand_colors = [colors[i] for i in rand_labels]

#plotting
mpl.rcParams['font.size'] = 10
fig = plt.figure()
ax = plt.gca()
plt.sca(ax)

vor = Voronoi(proto_points)
voronoi_plot_2d(vor,
                ax= ax,
                show_vertices = False,
                show_points= False,
                line_colors='k',
                line_width=3)

for c,p in zip(colors, proto_points):
    ax.plot(p[0], p[1], '.'+c, markersize=10)

for c,p in zip(rand_colors, rand_points):
    ax.plot(p[0], p[1], 'o'+c, markersize=2, fillstyle='none')

for i, (txt,p) in enumerate(zip(texts, proto_points)):
    ax.annotate(txt, (p[0]-.3, p[1]-.25))

plt.axis('square')
plt.xlim([0, 5])
plt.ylim([0, 5])
plt.xticks([0, 5])
plt.yticks([0, 5])
plt.xlabel('ORN 1 Activity')
plt.ylabel('ORN 2 Activity')
plt.savefig('task.pdf',transparent=True)