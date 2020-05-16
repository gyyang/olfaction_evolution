"""Analysis tools related to connection weights"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

THRES = 0.1


def fit_bimodal(x):
    """Fit bimodal distribution to data.

    Args:
        x: np array of data

    Return:
        x_plot: np array
        pdf1: np array, pdf of modal 1
        pdf2: np array, pdf of modal 2
        clf: classifier object
    """
    x = x[:, np.newaxis]
    clf = GaussianMixture(n_components=2, means_init=[[-5], [0.]], n_init=10)
    clf.fit(x)
    x_plot = np.linspace(x.min(), x.max(), 1000)

    pdf1 = multivariate_normal.pdf(x_plot, clf.means_[0],
                                   clf.covariances_[0]) * clf.weights_[0]
    pdf2 = multivariate_normal.pdf(x_plot, clf.means_[1],
                                   clf.covariances_[1]) * clf.weights_[1]
    return x_plot, pdf1, pdf2, clf


def fit_multimodal(x, max_n_modal=2, verbose=False):
    """Fit multimodal distribution to data.

    Args:
        x: np array of data
        max_n_modal: int, maximum number of modal
        verbose: bool

    Return:
        x_plot: np array
        pdf1: np array, pdf of modal 1
        pdf2: np array, pdf of modal 2
        clf: classifier object
    """
    x = x[:, np.newaxis]
    n_modals = range(1, max_n_modal+1)
    clfs = []
    bics = []
    for n_modal in n_modals:
        if n_modal == 1:
            means_init = [[-2]]
        elif n_modal == 2:
            means_init = [[-5], [0.]]
        elif n_modal == 3:
            means_init = [[-5], [0.], [1.5]]
        else:
            means_init = None
        clf = GaussianMixture(n_components=n_modal, means_init=means_init, n_init=1)
        clf.fit(x)
        clfs.append(clf)
        bic = clf.bic(x)
        bics.append(bic)

        if verbose:
            print('Modal {}, means {}, BIC {:0.2f}'.format(
                n_modal, np.array(clf.means_).flatten(), bic))

    i_modal = np.argmin(bics)
    n_modal = n_modals[i_modal]
    clf = clfs[i_modal]

    if verbose:
        print('Choosing modals:', n_modal)

    x_plot = np.linspace(x.min(), x.max(), 1000)

    ind_sort = np.argsort(np.array(clf.means_).flatten())

    pdfs = list()
    for i in ind_sort:
        pdf = multivariate_normal.pdf(x_plot, clf.means_[i],
                                      clf.covariances_[i]) * clf.weights_[i]
        pdfs.append(pdf)

    return x_plot, np.array(pdfs), clf, n_modal


def infer_threshold(x, use_logx=True, visualize=False, force_thres=None,
                    downsample=True):

    """Infers the threshold of a bi-modal distribution.

    The log-input will be fit as a mixture of 2 gaussians.

    Args:
        x: an array containing the values to be fitted
        use_logx: bool, if True, fit log-input

    Returns:
        thres: a scalar threshold that separates the two gaussians
        res_fit: dictionary containing parameters related to fitting process
    """
    # Select neurons that receive both strong and weak connections
    # weak connections should be around median, where strong should be around max
    res_fit = {}

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
        x = np.log(x +1e-10)

    if force_thres is not None:
        thres_ = np.log(force_thres) if use_logx else force_thres
    else:
        thres_not_found = False
        # x_plot, pdf1, pdf2, clf = fit_bimodal(x)
        x_plot, pdfs, clf, n_modal = fit_multimodal(x)
        res_fit['x_plot'] = x_plot
        res_fit['pdfs'] = pdfs
        res_fit['n_modal'] = n_modal

        if n_modal < 2:
            print('Only 1 modal found')
            thres_not_found = True
        else:
            diff = pdfs[0] < np.sum(pdfs[1:], axis=0)
            try:
                thres_ = x_plot[np.where(diff)[0][0]]
            except IndexError:
                print('Unable to find proper threshold, revert to default')
                thres_not_found = True

        if thres_not_found:
            thres_ = np.log(THRES) if use_logx else THRES

    thres = np.exp(thres_) if use_logx else thres_

    if visualize:
        bins = np.linspace(x.min(), x.max(), 100)
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        ax.hist(x, bins=bins, density=True)
        if force_thres is None:
            pdf = np.sum(pdfs, axis=0)
            ax.plot(x_plot, pdf)
        ax.plot([thres_, thres_], [0, 1])

        if use_logx:
            x = np.exp(x)
            thres_ = np.exp(thres_)
            bins = np.linspace(x.min(), x.max(), 100)
            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
            ax.hist(x, bins=bins, density=True)
            ax.plot([thres_, thres_], [0, 1])
            # ax.set_ylim([0, 1])

    res_fit['thres'] = thres
    return thres, res_fit