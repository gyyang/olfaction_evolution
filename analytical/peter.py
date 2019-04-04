import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances


def get_labels(prototypes, odors):
    dist = euclidean_distances(prototypes, odors)
    label = np.argmin(dist, axis=0)
    return label

n_samples = 200
n_proto = 4

n_pn = 5
n_kc = (n_pn-1) * (n_pn-2)
k = 3

orn = np.random.uniform(low=0, high=1, size=[n_samples, n_pn])

proto_points = np.random.uniform(low=0, high=1, size=[n_proto, n_pn])
rand_labels = get_labels(proto_points, orn)
colors = [np.array([55, 94, 151]) / 255.,  # blue
          np.array([251, 101, 66]) / 255.,  # orange
          np.array([255, 187, 0]) / 255.,  # red
          np.array([63, 104, 28]) / 255., ]  # green
rand_colors = [colors[i] for i in rand_labels]


combs = list(combinations(range(n_pn), k))
matrix = np.zeros((n_pn, len(combs)))
for i, comb in enumerate(combs):
    matrix[comb,i] = 1
kc = np.matmul(orn, matrix)

ixs = [0, 1, 2]

fig = plt.figure(figsize=[10,10])
axs = fig.add_subplot(211, projection='3d')
mat = orn

# for point, color in zip(mat, rand_colors):
axs.scatter(mat[:,ixs[0]], mat[:,ixs[1]], zs= mat[:,ixs[2]], zdir='z', s=20, c=rand_colors, depthshade=True)

axs = fig.add_subplot(212, projection='3d')
mat = kc
axs.scatter(mat[:,ixs[0]], mat[:,ixs[1]], zs= mat[:,ixs[2]], zdir='z', s=20, c=rand_colors, depthshade=True)

plt.show()