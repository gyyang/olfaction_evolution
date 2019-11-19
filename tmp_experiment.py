import os
import shutil

import task
import train
import configs

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# config = configs.input_ProtoConfig()
# config.label_type = 'multi_head_sparse'
# task.save_proto(config, folder_name='multi_head')

config = configs.FullConfig()
config.save_path = './files/tmp_train'
config.max_epoch = 15
config.batch_size = 256
config.N_ORN_DUPLICATION = 1
config.ORN_NOISE_STD = 0
config.train_pn2kc = True
config.sparse_pn2kc = False
config.lr = 1e-3
config.pn_norm_pre = 'batch_norm'

config.save_every_epoch = False
# config.initial_pn2kc = .1
# config.train_kc_bias = False
# config.kc_loss = False

config.data_dir = './datasets/proto/standard'

try:
    shutil.rmtree(config.save_path)
except FileNotFoundError:
    pass

train.train(config, reload=True)
