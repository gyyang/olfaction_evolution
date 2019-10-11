#changed
from collections import OrderedDict
from collections.__init__ import OrderedDict

import argparse
import configs
import standard.experiment as se
from standard.hyper_parameter_train import local_train, cluster_train

parser = argparse.ArgumentParser()
parser.add_argument('-p','--pn', nargs='+', help='N_PN', default=[50])
args = parser.parse_args()

def temp(n_pn=50):
    config = configs.FullConfig()
    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn' + str(n_pn)

    config.max_epoch = 100
    config.direct_glo = True

    config.kc_dropout = True
    config.kc_dropout_rate = 0.5

    config.train_pn2kc = True
    config.sparse_pn2kc = False
    config.coding_level = None

    config.separate_optimizer = False
    config.separate_lr = 1e-3
    config.save_log_only = True

    config.kc_prune_weak_weights = True
    config.initial_pn2kc = 6 / n_pn

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['lr'] = [3e-3, 1e-3, 3e-4, 1e-4]
    hp_ranges['kc_prune_threshold'] = [1/n_pn, 4/n_pn]
    hp_ranges['N_KC'] = [2500, 5000, 10000]
    return config, hp_ranges

def temp_vary_K(n_pn=50):
    """Standard training setting"""
    config = configs.FullConfig()
    config.max_epoch = 10

    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn'+str(n_pn)

    config.max_epoch = 100
    config.N_ORN_DUPLICATION = 1
    config.kc_dropout = True
    config.kc_dropout_rate = .2
    config.direct_glo = True
    config.save_log_only = True
    config.model = 'K'

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_inputs'] = [1, 3, 5, 7, 9, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
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
    path = './files/cluster_simple' + str(n_pn)
    cluster_train(temp(n_pn), path, path= cluster_path)

# local_train
n_pns = [50]
for n_pn in n_pns:
    path = './files/test' + str(n_pn)

    try:
        import shutil
        shutil.rmtree(path)
    except:
        pass
    local_train(temp(n_pn), path)
