"""Vary the basic arthictecture"""

import train
import configs


skip_orn2pns = [True, True, False]
skip_pn2kcs = [True, False, False]

for i in range(3):
    save_name = 'skip_orn2pn' + str(skip_orn2pns[i]) + 'skip_pn2kc' + str(skip_pn2kcs[i])
    config = configs.FullConfig()
    config.max_epoch = 30
    config.train_orn2pn = False
    config.train_pn2kc = False
    config.kc_norm_pre = 'batch_norm'
    config.skip_orn2pn = skip_orn2pns[i]
    config.skip_pn2kc = skip_pn2kcs[i]
    config.data_dir = './datasets/proto/_100_generalization_onehot_s0'
    config.save_path = './files/' + save_name

    train.train(config, reload=False)
