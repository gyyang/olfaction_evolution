#changed
from collections import OrderedDict
from collections.__init__ import OrderedDict

import argparse
import configs
import standard.experiment as se
from standard.hyper_parameter_train import local_train, cluster_train

parser = argparse.ArgumentParser()
parser.add_argument('-p','--pn', nargs='+', help='N_PN', default=[50, 100])
args = parser.parse_args()

testing_epochs = 50
def temp(argTest=False, n_pn=50):
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
    config.initial_pn2kc = n_pn / 1000 # heuristic
    # config.pn_norm_pre = 'batch_norm'

    config.save_every_epoch = False

    hp_ranges = OrderedDict()
    hp_ranges['lr'] = [5e-3, 2e-3, 1e-3, 5*1e-4, 2*1e-4, 1e-4]
    hp_ranges['N_KC'] = [2500, 5000, 10000, 20000]
    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges

is_test = True
train = cluster_train
cluster_path = '/axsys/scratch/ctn/users/yw2500/olfaction_evolution'
n_pns = [int(x) for x in args.pn]
print(n_pns)
for n_pn in n_pns:
    path = './files/cluster_initial_pn2kc' + str(n_pn)
    cluster_train(temp(is_test, n_pn), path, path= cluster_path)

#local_train
# n_pns = [500]
# for n_pn in n_pns:
#     path = './files/test' + str(n_pn)
#     local_train(temp(is_test, n_pn), path)