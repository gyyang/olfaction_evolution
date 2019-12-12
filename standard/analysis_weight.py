"""Analysis tools related to connection weights"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

THRES = 0.1


def infer_threshold(x, use_logx=True, visualize=False, force_thres=None,
                    downsample=True):

    """Infers the threshold of a bi-modal distribution.

    The log-input will be fit as a mixture of 2 gaussians.

    Args:
        x: an array containing the values to be fitted
        use_logx: bool, if True, fit log-input

    Returns:
        thres: a scalar threshold that separates the two gaussians
    """
    # Select neurons that receive both strong and weak connections
    # weak connections should be around median, where strong should be around max
    x = np.array(x)
    ratio = np.max(x, axis=0) / np.median(x, axis=0)
    # heuristic that works well for N=50-500, can plot hist of ratio
    ind = ratio > 15
    if np.sum(ind) > 0:
        x = x[:, ind]  # select expansion layer neurons

    x = x.flatten()

    if downsample:
        if len(x) > 1e5:
            x = np.random.choice(x, size=(int(1e5),))

    if use_logx:
        x = np.log( x +1e-10)
    x = x[:, np.newaxis]

    if force_thres is not None:
        thres_ = np.log(force_thres) if use_logx else force_thres
    else:
        clf = GaussianMixture(n_components=2, means_init=[[-5], [0.]], n_init=1)
        clf.fit(x)
        x_tmp = np.linspace(x.min(), x.max(), 1000)

        pdf1 = multivariate_normal.pdf(x_tmp, clf.means_[0],
                                       clf.covariances_[0]) * clf.weights_[0]
        pdf2 = multivariate_normal.pdf(x_tmp, clf.means_[1],
                                       clf.covariances_[1]) * clf.weights_[1]

        if clf.means_[0, 0] < clf.means_[1, 0]:
            diff = pdf1 < pdf2
        else:
            diff = pdf1 > pdf2

        try:
            thres_ = x_tmp[np.where(diff)[0][0]]
        except IndexError:
            print('Unable to find proper threshold, revert to default')
            thres_ = np.log(THRES) if use_logx else THRES

    thres = np.exp(thres_) if use_logx else thres_

    if visualize:
        bins = np.linspace(x.min(), x.max(), 100)
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        ax.hist(x[:, 0], bins=bins, density=True)
        if force_thres is None:
            pdf = pdf1 + pdf2
            ax.plot(x_tmp, pdf)
        ax.plot([thres_, thres_], [0, 1])

        if use_logx:
            x = np.exp(x)
            thres_ = np.exp(thres_)
            bins = np.linspace(x.min(), x.max(), 100)
            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
            ax.hist(x[:, 0], bins=bins, density=True)
            ax.plot([thres_, thres_], [0, 1])
            # ax.set_ylim([0, 1])
    return thres