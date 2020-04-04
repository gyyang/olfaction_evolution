import os
import shutil

import task
import torchtrain as train
import configs

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# config = configs.input_ProtoConfig()
# config.label_type = 'multi_head_sparse'
# task.save_proto(config, folder_name='multi_head')

config = configs.FullConfig()
config.N_PN = 50
# config.N_KC = 40000
config.save_path = './files/tmp_train'
config.max_epoch = 100
# config.batch_size = 256
config.batch_size = 8192
config.N_ORN_DUPLICATION = 1
config.ORN_NOISE_STD = 0.3*0
config.train_pn2kc = True
config.sparse_pn2kc = False
config.lr = 2*1e-3
# config.lr = 1*1e-3
config.pn_norm_pre = 'batch_norm'

config.kc_prune_weak_weights = True
config.kc_prune_threshold = 1./config.N_PN

config.initial_pn2kc = 5./config.N_PN
config.initializer_pn2kc = 'uniform'
# config.initializer_pn2kc = 'constant'

config.skip_orn2pn = True
config.kc_ffinh = False
config.kc_ffinh_coeff = 0

config.kc_recinh = True
config.kc_recinh_coeff = 0.5
config.kc_recinh_step = 10
# config.kc_bias = -2.5 * (1 - config.kc_ffinh_coeff)
# config.kc_bias = -1 * (1 - config.kc_ffinh_coeff)
# config.save_every_epoch = False
# config.initial_pn2kc = .1
# config.train_kc_bias = False
# config.kc_loss = False

config.data_dir = './datasets/proto/standard'
# config.data_dir = './datasets/proto/orn' + str(config.N_PN)
# config.data_dir = './datasets/proto/n_or_per_orn' + str(50)
# config.data_dir = './datasets/proto/orn200'

try:
    shutil.rmtree(config.save_path)
except FileNotFoundError:
    pass

train.train(config, reload=True)
