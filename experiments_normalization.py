from collections import OrderedDict
from standard.hyper_parameter_train import local_train
import task
import configs
import tools
import standard.analysis as sa
import os
import matplotlib.pyplot as plt
import numpy as np
import utils

def temp():
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/mask'
    config.max_epoch = 3
    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['pn_norm_post'] = ['biology']
    return config, hp_ranges

path = './files/pn_normalization'

# local_train(temp(), path)

try:
    rmax = tools.load_pickle(path, 'model/layer1/r_max:0')
    print('rmax: {}'.format(rmax))
    rho = tools.load_pickle(path, 'model/layer1/rho:0')
    print('rho: {}'.format(rho))
    m = tools.load_pickle(path, 'model/layer1/m:0')
    print('m: {}'.format(m))
except:
    pass

try:
    gamma = tools.load_pickle(path, 'model/layer1/LayerNorm/gamma:0')
    print('gamma params: {}'.format(gamma))
except:
    pass



#
d = dirs[2]
glo_act, glo_in, glo_in_pre, kc = utils.get_model_vars(d)
fig, ax = plt.subplots(nrows=3, ncols=2)
i=0
var = glo_in
ax[i,0].hist(var.flatten(), bins=100, range =(0, 15))
ax[i,0].set_title('Activity')
sparsity = np.count_nonzero(var > 0, axis= 1) / var.shape[1]
ax[i,1].hist(sparsity, bins=50, range=(0,1))
ax[i,1].set_title('Sparseness')

i=1
var = glo_act
ax[i,0].hist(var.flatten(), bins=100, range =(0, 15))
ax[i,0].set_title('Activity')
sparsity = np.count_nonzero(var > 0, axis= 1) / var.shape[1]
ax[i,1].hist(sparsity, bins=50, range=(0,1))
ax[i,1].set_title('Sparseness')

i=2
var = kc
ax[i,0].hist(var.flatten(), bins=100, range =(0, 15))
ax[i,0].set_title('Activity')
sparsity = np.count_nonzero(var > 0.05, axis= 1) / var.shape[1]
ax[i,1].hist(sparsity, bins=50, range=(0,1))
ax[i,1].set_title('Sparseness')
plt.show()