import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

rootpath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(rootpath)  # This is hacky, should be fixed

import standard.analysis as sa

path = '../files/multi_head'

# sa.plot_weights(path)
# sa.plot_progress(path)
import standard.analysis_pn2kc_training as analysis_pn2kc_training
# analysis_pn2kc_training.plot_distribution(path)
analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True)

import tools

d = os.path.join(path, '000000', 'epoch')
wout1 = tools.load_pickle(d, 'model/layer3/kernel:0')[-1]
wout2 = tools.load_pickle(d, 'model/layer3_2/kernel:0')[-1]
wglo = tools.load_pickle(d, 'w_glo')[-1]

thres = analysis_pn2kc_training.infer_threshold(wglo)
sparsity = np.count_nonzero(wglo > thres, axis=0)

ind_sort = np.argsort(sparsity)

sparsity = sparsity[ind_sort]
wout1 = wout1[ind_sort, :]
wout2 = wout2[ind_sort, :]

plt.figure()
plt.imshow(wout1[:500], aspect='auto')

plt.figure()
plt.imshow(wout2[:500], aspect='auto')

v1 = sparsity

strength_wout1 = np.sum(abs(wout1), axis=1)
strength_wout2 = np.sum(abs(wout2), axis=1)

v2 = strength_wout2
# v2 = strength_wout2/(strength_wout1+strength_wout2)


plt.figure()
plt.scatter(v1, v2, alpha=0.3)

X = np.stack([v1, v2]).T
X = X / np.linalg.norm(X, axis=0)
from sklearn.cluster import KMeans

labels = KMeans(n_clusters=2, random_state=0).fit_predict(X)

plt.figure()
plt.scatter(v1[labels == 0], v2[labels == 0], alpha=0.02)
plt.scatter(v1[labels == 1], v2[labels == 1], alpha=0.3)
plt.xlabel('PN Input degree')
plt.ylabel('Overall strength to valence output')