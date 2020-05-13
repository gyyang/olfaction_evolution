"""File that summarizes all key results.

To train and analyze all models quickly, run in command line
python paper.py -d=0 --train --analyze --testing

To reproduce the results from paper, run
python paper.py -d=0 --train --analyze

To analyze pretrained networks, run
python paper.py -d=0 --analyze

To run specific experiments (e.g. orn2pn, vary_pn), run
python paper.py -d=0 --train --analyze --experiment orn2pn vary_pn
"""

import os
import argparse

from standard.hyper_parameter_train import train_experiment
import matplotlib as mpl

SCRATCHPATH = '/share/ctn/projects/olfaction_evolution'
ROBERT_SCRATCHPATH = '/share/ctn/users/gy2259/olfaction_evolution'
PETER_SCRATCHPATH = '/share/ctn/users/yw2500/olfaction_evolution'

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', help='CUDA device number', default=0, type=int)
parser.add_argument('-t', '--train', help='Training', action='store_true')
parser.add_argument('-a', '--analyze', help='Analyzing', action='store_true')
parser.add_argument('-test', '--testing', help='For debugging', action='store_true')
parser.add_argument('-e','--experiment', nargs='+', help='Experiments', default='core')
parser.add_argument('-cp', '--clusterpath', help='cluster path', default=SCRATCHPATH)
parser.add_argument('-c','--cluster', help='Use cluster?', action='store_true')
parser.add_argument('--torch', help='Use torch', action='store_true')
args = parser.parse_args()

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

for item in args.__dict__.items():
    print(item)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
TRAIN, ANALYZE, is_test, use_cluster, cluster_path = args.train, args.analyze, args.testing, args.cluster, args.clusterpath

if use_cluster:
    if cluster_path == 'peter' or cluster_path == 'pw':
        cluster_path = PETER_SCRATCHPATH
    elif cluster_path == 'robert' or cluster_path == 'gry':
        cluster_path = ROBERT_SCRATCHPATH
    else:
        cluster_path = SCRATCHPATH

    save_path = cluster_path
else:
    save_path = './'

# TRAIN = True
# is_test = True
# ANALYZE = True
# args.experiment = ['metalearn']
#
if ANALYZE:
    import standard.analysis as sa
    import standard.analysis_pn2kc_training as analysis_pn2kc_training
    import standard.analysis_pn2kc_random as analysis_pn2kc_random
    import standard.analysis_orn2pn as analysis_orn2pn
    import standard.analysis_rnn as analysis_rnn


if args.experiment == 'core':
    experiments = ['standard',
                   'receptor',
                   'vary_pn',
                   'vary_kc',
                   'metalearn',
                   'pn_normalization',
                   'vary_kc_activity_fixed', 'vary_kc_activity_trainable',
                   'vary_kc_claws', 'vary_kc_claws_new','train_kc_claws', 'random_kc_claws', 'train_orn2pn2kc',
                   'kcrole', 'kc_generalization',
                   'multi_head']
else:
    experiments = args.experiment

if TRAIN:
    for experiment in experiments:
        train_experiment(experiment, use_cluster=use_cluster, path=save_path,
                         use_torch=args.torch, testing=is_test)


if 'standard' in experiments:
    path = './files/standard'
    if ANALYZE:
        # accuracy
        sa.plot_progress(path, ykeys=['val_acc', 'glo_score', 'K_inferred'])

        # orn-pn
        sa.plot_weights(os.path.join(path,'000000'), var_name='w_orn', sort_axis=1)
        try:
            analysis_orn2pn.correlation_across_epochs(path, arg='weight')
            analysis_orn2pn.correlation_across_epochs(path, arg='activity')
        except ModuleNotFoundError:
            pass

        # pn-kc
        sa.plot_weights(os.path.join(path,'000000'), var_name='w_glo')

        # pn-kc K
        analysis_pn2kc_training.plot_distribution(path, xrange=1.5, log=False)
        analysis_pn2kc_training.plot_distribution(path, xrange=1.5, log=True)
        analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True, epoch=-1)

        # pn-kc random
        analysis_pn2kc_random.plot_cosine_similarity(path, dir_ix= 0, shuffle_arg='preserve', log=False)
        analysis_pn2kc_random.plot_distribution(path, dir_ix= 0)
        analysis_pn2kc_random.claw_distribution(path, dir_ix= 0, shuffle_arg='random')
        analysis_pn2kc_random.pair_distribution(path, dir_ix= 0, shuffle_arg='preserve')

if 'receptor' in experiments:
    path = './files/receptor'
    if ANALYZE:
        sa.plot_progress(path, ykeys=['val_acc', 'glo_score', 'K_inferred'])

        for var_name in ['w_or', 'w_orn', 'w_combined', 'w_glo']:
            sort_axis = 0 if var_name == 'w_or' else 1
            sa.plot_weights(os.path.join(path, '000000'),
                            var_name='w_or', sort_axis=sort_axis)

        # pn-kc K
        analysis_pn2kc_training.plot_distribution(path, xrange=1.5, log=False)
        analysis_pn2kc_training.plot_distribution(path, xrange=1.5, log=True)
        analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True, epoch=-1)

if 'vary_pn' in experiments:
    # Vary nPN
    path = './files/vary_pn'
    if ANALYZE:
        xticks = [20, 50, 100, 200, 1000]
        ylim, yticks = [0, 1.05], [0, .25, .5, .75, 1]
        ykeys = ['val_acc', 'glo_score']
        for ykey in ykeys:
            sa.plot_results(path, xkey='N_PN', ykey=ykey, figsize=(1.75, 1.75), ax_box=(0.25, 0.25, 0.65, 0.65),
                            loop_key='kc_dropout_rate', logx=True, ax_args={'ylim': ylim, 'yticks': yticks, 'xticks':xticks})

if 'vary_kc' in experiments:
    # Vary nKC
    path = './files/vary_kc'
    if ANALYZE:
        xticks = [50, 200, 1000, 2500, 10000]
        ylim, yticks = [0, 1.05], [0, .25, .5, .75, 1]
        ykeys = ['val_acc', 'glo_score']
        for ykey in ykeys:
            sa.plot_results(path, xkey='N_KC', ykey=ykey, figsize=(1.75, 1.75), ax_box=(0.25, 0.25, 0.65, 0.65),
                            loop_key='kc_dropout_rate', logx=True, ax_args={'ylim': ylim, 'yticks': yticks, 'xticks':xticks})

if 'rnn' in experiments:
    path = './files/rnn'
    if ANALYZE:
        sa.plot_progress(path, ykeys=['val_acc'], legend_key='TIME_STEPS')
        # analysis_rnn.analyze_t0(path, dir_ix=0)
        analysis_rnn.analyze_t_greater(path, dir_ix=1)
        analysis_rnn.analyze_t_greater(path, dir_ix=2)

if 'metalearn' in experiments:
    path = './files/metalearn'
    if ANALYZE:
        # sa.plot_weights(path, var_name='w_orn', sort_axis=1, dir_ix=-0, average=False)
        # sa.plot_weights(os.path.join(path, '0','epoch','2000'), var_name='w_glo', sort_axis=-1)
        analysis_pn2kc_training.plot_distribution(path, xrange=1)
        analysis_pn2kc_training.plot_sparsity(path, dir_ix=0, dynamic_thres=True, thres=.05)
        # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'w_orn', dir_ix = 0)
        # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'w_glo', dir_ix= 1)
        # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'model/layer3/kernel:0', dir_ix = 0)
        # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'model/layer3/kernel:0', dir_ix = 1)

if 'pn_normalization' in experiments:
    path = './files/pn_normalization'
    if ANALYZE:
        sa.plot_results(path, xkey='data_dir', ykey='val_acc', loop_key='pn_norm_pre',
                        select_dict={
                            'pn_norm_pre': ['None', 'fixed_activity', 'biology'],
                            'data_dir': ['./datasets/proto/standard',
                                         './datasets/proto/concentration_mask_row_0.6'
                                         ]},
                        figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65), sort=False)

        sa.plot_results(path, xkey='data_dir', ykey='val_acc', loop_key='pn_norm_pre',
                        select_dict={
                            'pn_norm_pre': ['None', 'fixed_activity', 'biology'],
                            'data_dir': ['./datasets/proto/concentration',
                                         './datasets/proto/concentration_mask_row_0',
                                         './datasets/proto/concentration_mask_row_0.6',
                                         ]},
                        figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65), sort=False)
        # import tools
        # rmax = tools.load_pickle(path, 'model/layer1/r_max:0')
        # rho = tools.load_pickle(path, 'model/layer1/rho:0')
        # m = tools.load_pickle(path, 'model/layer1/m:0')
        # print(rmax)
        # print(rho)
        # print(m)
        #
        # analysis_activity.image_activity(path, 'glo_out')
        # analysis_activity.image_activity(path, 'kc_out')
        # analysis_activity.distribution_activity(path, 'glo_out')
        # analysis_activity.distribution_activity(path, 'kc_out')
        # analysis_activity.sparseness_activity(path, 'kc_out')
        
if 'multi_head' in experiments:
    path = './files/multi_head'
    if ANALYZE:
        # analysis_multihead.main1('multi_head')
        sa.plot_weights(os.path.join(path, '000000'), var_name='w_orn', sort_axis=1)
