import os
import task
import configs
from collections import OrderedDict
import numpy as np
import tools
import train
import standard.analysis as sa
import pickle
import model as network_models
import standard.analysis_activity as analysis_activity
import standard.analysis_pn2kc_training as analysis_training
import standard.analysis_metalearn as analysis_metalearn
import shutil
import matplotlib.pyplot as plt
import standard.analysis_multihead as analysis_multihead
import mamlmetatrain
import matplotlib as mpl
import oracle.evaluatewithnoise

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

def t(experiment, save_path,s=0,e=1000):
    """Train all models locally."""
    for i in range(s, e):
        config = tools.varying_config(experiment, i)
        if config:
            print('[***] Hyper-parameter: %2d' % i)
            config.save_path = os.path.join(save_path, str(i).zfill(6))
            train.train(config)


def st(experiment, save_path, s=0,e=1000):
    """Train all models locally."""
    for i in range(s, e):
        config = tools.varying_config_sequential(experiment, i)
        if config:
            print('[***] Hyper-parameter: %2d' % i)
            config.save_path = os.path.join(save_path, str(i).zfill(6))
            train.train(config)

def temp_meta():
    '''
    '''
    config = configs.MetaConfig()
    config.meta_lr = .001
    config.N_CLASS = 5
    config.save_every_epoch = True
    config.meta_output_dimension = 5
    config.meta_batch_size = 32
    config.meta_num_samples_per_class = 32
    config.meta_print_interval = 250

    config.replicate_orn_with_tiling = True
    config.N_ORN_DUPLICATION = 10
    config.train_kc_bias = True

    config.metatrain_iterations = 20000
    config.pn_norm_pre = 'batch_norm'
    config.kc_norm_pre = 'batch_norm'
    config.sparse_pn2kc = False
    config.train_pn2kc = True

    config.sign_constraint_pn2kc = False
    config.sign_constraint_orn2pn = False

    config.data_dir = './datasets/proto/test'
    config.save_path = './files/test/0'

    hp_ranges = OrderedDict()
    hp_ranges['dummy'] = [True]
    return config, hp_ranges

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

    config.separate_optimizer = True
    config.separate_lr = 1e-3
    config.save_log_only = True

    config.kc_prune_weak_weights = True
    config.initial_pn2kc = 8 / n_pn
    config.kc_prune_threshold = 5/n_pn

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_prune_weak_weights'] = [True, False]
    return config, hp_ranges

def temp_glomeruli(n_pn=50):
    config = configs.FullConfig()
    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn' + str(n_pn)

    config.max_epoch = 15
    config.pn_norm_pre = 'batch_norm'

    config.initializer_orn2pn = 'constant'
    config.initial_orn2pn = .1
    config.pn_prune_threshold = .05
    config.pn_prune_weak_weights = True

    config.train_pn2kc = False
    config.sparse_pn2kc = True

    config.save_log_only = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['lr'] = [1e-3]
    return config, hp_ranges

# mamlmetatrain.train(temp_meta()[0])
#
path = './files_temp/cluster_pn2kc_prune_or_not_prune_lr_1e-3_50'
# try:
#     shutil.rmtree(path)
# except:
#     pass
# t(temp_glomeruli(), path, s=0)

sa.plot_progress(path, legends=['Prune','None'], plot_vars= ['K','K_inferred'], ylim=[0, 20], epoch_range=[0,100])
sa.plot_progress(path, legends=['Prune','None'], plot_vars= ['val_acc'], ylim=[.5, 1.05], epoch_range=[0,100])
sa.plot_progress(path, legends=['Prune','None'], plot_vars= ['val_logloss', 'train_logloss'], epoch_range=[0,100])

# sa.plot_progress(path, legends=['Prune','None'], plot_vars= ['K','K_inferred'], ylim=[0, 20])
# sa.plot_progress(path, legends=['Prune','None'], plot_vars= ['val_acc'], ylim=[.5, 1.05])
# sa.plot_progress(path, legends=['Prune','None'], plot_vars= ['val_logloss', 'train_logloss'])

# sa.plot_progress(path, plot_vars= ['val_logloss', 'train_logloss', 'glo_score'], legends=[3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
# sa.plot_progress(path, legends=['prune','none'], plot_vars= ['val_logloss', 'train_logloss', 'glo_score'])


# analysis_training.plot_distribution(path, xrange=.5, log=False)
# analysis_training.plot_distribution(path, xrange=.5, log=True)
# analysis_training.plot_sparsity(path, dynamic_thres=True, thres=0.1, visualize=True, epochs = [-1], xrange=50)



# path = r'files/cluster_fast_convergence/cluster_fast_convergence_no_bn50'
# sa.plot_results(path, x_key='kc_inputs', y_key='val_logloss', select_dict={'ORN_NOISE_STD': 0},
#                         figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
#                         ax_args={'ylim':[-1, 4], 'yticks':[-1,0,1,2]})




#
# t = [1, 2, 10, 30, 60, 99]
# for i in t:
#     res = tools.load_all_results(path, argLast=False, ix=i)
#     sa.plot_results(path, x_key='kc_inputs', y_key='val_loss',
#                     select_dict={'ORN_NOISE_STD': 0}, res=res, string=str(i), yticks='', ax_args={'ylim':[-1, 4]})

# np = os.path.join(path,'000000','epoch')
# wglos = tools.load_pickle(np, 'w_glo')
# ks = tools.load_pickle(np, 'K')

#
# sa.plot_progress(path)

# oracle.evaluatewithnoise.evaluate_across_epochs(path=path, values=[0, .01, .03, .1],n_rep=1)
# oracle.evaluatewithnoise.plot_acrossmodels(path=path, model_var='epoch')

# analysis_training.plot_distribution(path, xrange=.5, log=False)
# analysis_training.plot_distribution(path, xrange=.5, log=True)
# analysis_training.plot_sparsity(path, dynamic_thres=False, thres=.05, visualize=True, epochs = [-1], xrange=50)
#
# analysis_training.plot_sparsity_acrossepochs(os.path.join(path,'000000'))
# analysis_training.plot_sparsity_acrossepochs(os.path.join(path,'000001'))
# analysis_training.plot_sparsity_acrossepochs(os.path.join(path,'000002'))

# glo_in, glo_out, kc_in, kc_out, results = sa.load_activity(path)
# print(kc_in)
# b_orns = tools.load_pickle(path, 'b_glo')
# plt.hist(glo_in.flatten(), bins=20)
# plt.show()

# config = configs.input_ProtoConfig()
# config.N_CLASS = 1000
# config.N_ORN = 200
# task.save_proto(config, seed=0, folder_name='test_norn_200')
#
# path = './files/metalearn'
# folder = '0'
# ix = '7000'
# sa.plot_weights(os.path.join(path, folder,'epoch', ix), var_name='w_orn', sort_axis=1, dir_ix=-0, average=False)
# sa.plot_weights(os.path.join(path, folder,'epoch', ix), var_name='w_glo', sort_axis=-1, dir_ix=0)
# import standard.analysis_pn2kc_training
# standard.analysis_pn2kc_training.plot_distribution(path, xrange=1, log=True)
# standard.analysis_pn2kc_training.plot_distribution(path, xrange=1)
# standard.analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True, thres=.01, epochs=[-1])


# sa.plot_weights(path, var_name = 'w_or', sort_axis=0, dir_ix=0)
# sa.plot_weights(path, var_name = 'w_combined', dir_ix=0)
# sa.plot_weights(path, var_name = 'w_orn', sort_axis=1, dir_ix=0)
# sa.plot_weights(path, var_name = 'w_glo', dir_ix=0)
#

# analysis_multihead.main1(arg='multi_head')
#
# path = './files/metalearn'

# analysis_training.plot_distribution(path, xrange=.5, log=False)
# analysis_training.plot_distribution(path, xrange=.5, log=True)
# analysis_training.plot_sparsity(path, dynamic_thres=False, thres=.05, visualize=True)
#
# analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'w_orn', dir_ix=0, xlim=25)
# analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'w_glo', dir_ix=0, xlim=25)
# plt.show()

# analysis_training.plot_distribution(path)
# analysis_training.plot_pn2kc_claw_stats(path, x_key = 'n_trueclass', dynamic_thres=False)
# analysis_activity.sparseness_activity(path, 'glo_out')
# analysis_activity.sparseness_activity(path, 'kc_out', activity_threshold=.01)
# analysis_activity.plot_mean_activity_sparseness(path, 'kc_out', x_key='n_trueclass', loop_key='kc_dropout_rate')

# sa.plot_weights(path, var_name='model/layer3/kernel:0', dir_ix=0)