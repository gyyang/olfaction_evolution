#changed
from collections import OrderedDict
from collections.__init__ import OrderedDict

import argparse
import configs
import standard.experiment as se
from standard.hyper_parameter_train import local_train, cluster_train

parser = argparse.ArgumentParser()
parser.add_argument('-p','--pn', nargs='+', help='N_PN', default=[50, 75, 100, 125, 150])
args = parser.parse_args()

def temp(n_pn=50):
    """Standard training setting"""
    config = configs.FullConfig()
    config.max_epoch = 100

    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn'+str(n_pn)

    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0.
    config.skip_orn2pn = True
    config.sparse_pn2kc = False
    config.train_pn2kc = True
    config.initial_pn2kc = 10/n_pn
    config.train_kc_bias = False
    config.kc_loss = True #wtf

    # config.pn_norm_pre = 'batch_norm'
    # config.save_every_epoch = True

    hp_ranges = OrderedDict()
    hp_ranges['lr'] = [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    hp_ranges['N_KC'] = [2500, 5000, 10000, 20000]
    return config, hp_ranges

def tempK(n_pn=50):
    """Standard training setting"""
    config = configs.FullConfig()
    config.max_epoch = 100

    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn'+str(n_pn)
    config.model = 'K'
    config.train_pn2kc = True
    config.save_log_only = True
    config.kc_dropout_rate = .2
    config.kc_dropout = True

    hp_ranges = OrderedDict()
    hp_ranges['lr'] = [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    hp_ranges['N_KC'] = [2500, 5000, 10000]
    hp_ranges['initial_K'] = [n_pn, n_pn/2, n_pn/4]
    return config, hp_ranges


train = cluster_train
cluster_path = '/axsys/scratch/ctn/users/yw2500/olfaction_evolution'
n_pns = [int(x) for x in args.pn]
print(n_pns)
for n_pn in n_pns:
    path = './files/cluster_parameterize_K_vary_initial_K_fixed_bias' + str(n_pn)
    cluster_train(tempK(n_pn), path, path= cluster_path)

## local_train
# n_pns = [500]
# for n_pn in n_pns:
#     path = './files/test' + str(n_pn)
#
#     try:
#         import shutil
#         shutil.rmtree(path)
#     except:
#         pass
#     local_train(temp(is_test, n_pn), path)
