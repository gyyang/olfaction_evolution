import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from configs import input_ProtoConfig

arg_positive = True
arg_expand = False

def f(x, argInverse):
    if argInverse == False:
        return np.log(x + 1)
    else:
        return np.exp(x) - 1

def _make_hallem_dataset(config=None, arg_positive=True, arg_expand= True):
    '''
    :param config:
    :return: hallem carlson dataset in matrix format

    110 odors * 24 ORNs, with spontaneous activity subtracted
    '''
    if config is None:
        config = input_ProtoConfig()
    N_ODORS = 110
    N_ORNS = 24
    N_ORNS_TOTAL = config.N_ORN
    N_ORNS_FILL = N_ORNS_TOTAL - N_ORNS

    file = config.hallem_path
    with open(file) as f:
        vec = f.readlines()
    vec = [int(x.strip()) for x in vec]
    mat = np.reshape(vec, (N_ODORS+1, N_ORNS),'F')
    spontaneous_activity = mat[-1,:]
    odor_activation = mat[:-1,:] + spontaneous_activity

    if arg_expand:
        out = np.zeros((N_ODORS, N_ORNS_TOTAL))
        for i in range(N_ODORS):
            sampled = np.random.choice(odor_activation[i,:], size=N_ORNS_FILL, replace=True)
            out[i,:N_ORNS] = odor_activation[i,:]
            out[i,N_ORNS:] = sampled
    else:
        out = odor_activation

    if arg_positive:
        out[out < 0] = 0
    return out

def _simple_distribution_subplot(data, r, c, max, savename):
    fig, ax = plt.subplots(r, c)
    for i in range(data.shape[1]):
        ix = np.unravel_index(i, (r,c))
        ax[ix].hist(data[:,i], range=(0,max), bins=20)
    plt.savefig(savename)

def _covariance_image(data, savename):
    plt.figure()
    plt.imshow(data, cmap='RdBu_r', interpolation='none')
    plt.colorbar()
    plt.savefig(savename)



def _generate_from_hallem(config=None):
    if config is None:
        config = input_ProtoConfig()
    odor_activation = _make_hallem_dataset(config, arg_positive=arg_positive, arg_expand=arg_expand)

    log_odor_activation = f(odor_activation, argInverse=False)
    means = np.mean(log_odor_activation, axis=0)
    covs = np.cov(log_odor_activation, rowvar=False)
    sampled = np.random.multivariate_normal(means, covs, size= 110)
    fsampled = f(sampled, argInverse=True)

    sampled_cc = np.cov(fsampled, rowvar=False)
    hallem_cc = np.cov(odor_activation, rowvar=False)
    #distribution of odor-evoked activity for each of the 23 odor receptors
    plt.figure()
    plt.hist(np.sum(odor_activation, axis=1))
    plt.show()

    # _simple_distribution_subplot(odor_activation, 7, 8, 100, 'HALLEM')
    # _simple_distribution_subplot(fsampled, 7, 8, 100, 'SAMPLED')
    # _covariance_image(hallem_cc, 'HALLEM_COV')
    # _covariance_image(sampled_cc, 'SAMPLED_COV')


_make_hallem_dataset()
_generate_from_hallem()
#
a, m = 10., 2.
s = (np.random.pareto(a, 1000))
count, bins, _ = plt.hist(s, 100, density=True)
plt.show()