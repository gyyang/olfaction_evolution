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

    config.max_epoch = 50
    config.direct_glo = True

    config.lr = 3e-3

    config.kc_dropout = True
    config.kc_dropout_rate = 0.5

    config.train_pn2kc = True
    config.sparse_pn2kc = False
    config.coding_level = None

    config.save_log_only = True

    config.initial_pn2kc = 8 / n_pn
    config.kc_prune_threshold = 5 / n_pn

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_prune_weak_weights'] = [True, False]
    return config, hp_ranges

def temp_(n_pn=50):
    config = configs.FullConfig()
    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn' + str(n_pn)

    config.max_epoch = 50
    config.direct_glo = True

    config.lr = 3e-3

    config.kc_dropout = True
    config.kc_dropout_rate = 0.5

    config.train_pn2kc = True
    config.sparse_pn2kc = False
    config.coding_level = None

    config.save_log_only = True

    config.initial_pn2kc = 6 / n_pn
    config.kc_prune_threshold = 5 / n_pn
    config.kc_prune_weak_weights = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['lr'] = [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    hp_ranges['N_KC'] = [2500, 5000, 10000]
    hp_ranges['kc_prune_threshold'] = [1/n_pn, 2/n_pn, 5/n_pn]
    hp_ranges['kc_dropout_rate'] = [0, .3, .6]
    # hp_ranges['initial_pn2kc'] = [2/n_pn, 5/n_pn, 10/n_pn]
    return config, hp_ranges

def temp_glomeruli(n_pn=50):
    config = configs.FullConfig()
    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn' + str(n_pn)

    config.max_epoch = 100
    config.pn_norm_pre = 'batch_norm'

    config.train_pn2kc = False
    config.sparse_pn2kc = True

    config.save_log_only = False

    config.initializer_orn2pn = 'constant'
    config.initial_orn2pn = .1
    config.pn_prune_threshold = .05

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['pn_prune_weak_weights'] = [True, False]
    return config, hp_ranges

train = cluster_train
cluster_path = '/axsys/scratch/ctn/users/yw2500/olfaction_evolution'
n_pns = [int(x) for x in args.pn]
print(n_pns)
for n_pn in n_pns:
    path = './files/cluster_big' + str(n_pn)
    cluster_train(temp_(n_pn), path, path= cluster_path)

## local_train
#n_pns = [50]
#for n_pn in n_pns:
#    path = './files/test' + str(n_pn)

#    try:
#        import shutil
#        shutil.rmtree(path)
#    except:
#        pass
#    local_train(temp(n_pn), path)
