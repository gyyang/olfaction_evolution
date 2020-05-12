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
import copy

import standard.analysis_orn2pn
import standard.experiment as se
import standard.experiment_controls
import standard.experiment_controls as experiment_controls
from standard.hyper_parameter_train import local_train, cluster_train
import matplotlib as mpl
import numpy as np

SCRATCHPATH = '/share/ctn/projects/olfaction_evolution'
ROBERT_SCRATCHPATH = '/share/ctn/users/gy2259/olfaction_evolution'
PETER_SCRATCHPATH = '/share/ctn/users/yw2500/olfaction_evolution'

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', help='CUDA device number', default=0, type=int)
parser.add_argument('-t', '--train', help='Training', action='store_true')
parser.add_argument('-a', '--analyze', help='Analyzing', action='store_true')
parser.add_argument('-test', '--testing', help='For debugging', action='store_true')
parser.add_argument('-e', '--experiment', nargs='+', help='Experiments', default='core')
parser.add_argument('-cp', '--clusterpath', help='cluster path', default=SCRATCHPATH)
parser.add_argument('-c', '--cluster', help='Use cluster?', action='store_true')
parser.add_argument('-p','--pn', nargs='+', help='N_PN', default=[50])
parser.add_argument('--torch', help='Use torch', action='store_true')
args = parser.parse_args()

for item in args.__dict__.items():
    print(item)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
TRAIN, ANALYZE, is_test, use_cluster, cluster_path = args.train, args.analyze, args.testing, args.cluster, args.clusterpath

# TRAIN = True
# use_cluster = True
# args.pn = [50]
# ANALYZE = True
# args.experiment =['control_vary_pn']


if use_cluster:
    if cluster_path == 'peter' or cluster_path == 'pw':
        cluster_path = PETER_SCRATCHPATH
    elif cluster_path == 'robert' or cluster_path == 'gry':
        cluster_path = ROBERT_SCRATCHPATH
    else:
        cluster_path = SCRATCHPATH

    def train(experiment, save_path, **kwargs):
        cluster_train(experiment, save_path, path=cluster_path,
                      use_torch=args.torch, **kwargs)
else:
    train = local_train

if ANALYZE:
    import standard.analysis as sa
    try:
        import standard.analysis_pn2kc_peter
    except:
        pass
    import standard.analysis_pn2kc_training as analysis_pn2kc_training
    import standard.analysis_pn2kc_random as analysis_pn2kc_random
    import standard.analysis_orn2pn as analysis_orn2pn
    import standard.analysis_activity as analysis_activity
    import standard.analysis_multihead as analysis_multihead
    import standard.analysis_metalearn as analysis_metalearn
    import analytical.numerical_test as numerical_test
    import analytical.analyze_simulation_results as analyze_simulation_results
    import standard.analysis_nonnegative as analysis_nonnegative

# experiments
if args.experiment == 'core':
    experiments = ['']
else:
    experiments = args.experiment

if 'control_nonnegative' in experiments:
    path = './files/control_nonnegative'
    if TRAIN:
        train(experiment_controls.control_nonnegative(), save_path=path)
    if ANALYZE:
        sa.plot_weights(os.path.join(path, '000000'), sort_axis=1, average=False)
        sa.plot_weights(os.path.join(path, '000001'), sort_axis=1, average=False, positive_cmap=False, vlim=[-1, 1])
        for ix in range(0,2):
            standard.analysis_orn2pn.correlation_matrix(path, ix=ix, arg='ortho')
            standard.analysis_orn2pn.correlation_matrix(path, ix=ix, arg='corr')

        # # #sign constraint
        sa.plot_progress(path, ykeys=['glo_score','val_acc'], legend_key='sign_constraint_orn2pn')
        sa.plot_results(path, xkey='sign_constraint_orn2pn', ykey='glo_score')
        sa.plot_results(path, xkey='sign_constraint_orn2pn', ykey='val_acc')


if 'control_orn2pn' in experiments:
    # Vary ORN n duplication under different nKC
    path = './files/control_orn2pn'
    if TRAIN:
        train(experiment_controls.control_orn2pn(), save_path=path, control=True)
    if ANALYZE:
        default = {'ORN_NOISE_STD': 0, 'pn_norm_pre': 'batch_norm', 'kc_dropout_rate': 0.5, 'N_ORN_DUPLICATION':10, 'lr':1e-3}
        ykeys = ['glo_score', 'val_acc']

        for yk in ykeys:
            for xk, v in default.items():
                temp = copy.deepcopy(default)
                temp.pop(xk)
                if xk == 'lr':
                    logx= True
                else:
                    logx = False
                sa.plot_results(path, xkey=xk, ykey=yk,
                                select_dict=temp, logx=logx)

                sa.plot_progress(path, select_dict=temp, ykeys=[yk], legend_key=xk)

if 'control_pn2kc' in experiments:
    path = './files/control_pn2kc'
    if TRAIN:
        train(experiment_controls.control_pn2kc(), save_path=path, control=True)
    if ANALYZE:
        default = {'pn_norm_pre': 'batch_norm', 'kc_dropout_rate': 0.5, 'lr': 1e-3, 'initial_pn2kc':0, 'train_kc_bias':True}
        ykeys = ['val_acc', 'K_inferred']

        for yk in ykeys:
            exclude_dict = None
            if yk in ['K_inferred', 'sparsity_inferred', 'K','sparsity']:
                exclude_dict = {'lr': [3e-3, 1e-2, 3e-2]}

            for xk, v in default.items():
                temp = copy.deepcopy(default)
                temp.pop(xk)
                if xk == 'lr':
                    logx = True
                else:
                    logx = False
                sa.plot_results(path, xkey=xk, ykey=yk,
                                select_dict=temp, logx=logx)

                sa.plot_progress(path, select_dict=temp, ykeys=[yk],
                                 legend_key=xk, exclude_dict=exclude_dict)
        #
        res = standard.analysis_pn2kc_peter.do_everything(path, filter_peaks=False, redo=True)
        for xk, v in default.items():
            temp = copy.deepcopy(default)
            temp.pop(xk)
            sa.plot_xy(path, select_dict=temp, xkey='lin_bins_', ykey='lin_hist_', legend_key=xk, log=res,
                       ax_args={'ylim':[0, 500]})

if 'control_pn2kc_inhibition' in experiments:
    path = './files/control_pn2kc_inhibition'
    if TRAIN:
        train(experiment_controls.control_pn2kc_inhibition(), save_path=path, sequential=True)
    if ANALYZE:
        xkey = 'kc_recinh_coeff'
        ykeys = ['val_acc', 'K_inferred']
        # loop_key = 'kc_recinh_step'
        loop_key = None
        select_dict = {'kc_prune_weak_weights': False, 'kc_recinh_step': 9}
        for yk in ykeys:
            sa.plot_results(path, xkey=xkey, ykey=yk, loop_key=loop_key,
                            select_dict=select_dict)

            sa.plot_progress(path, ykeys=[yk], legend_key=xkey,
                             select_dict=select_dict)

        res = standard.analysis_pn2kc_peter.do_everything(path, filter_peaks=False, redo=True)
        sa.plot_xy(path, xkey='lin_bins_', ykey='lin_hist_', legend_key=xkey, log=res,
                   ax_args={'ylim':[0, 500]})

if 'control_pn2kc_prune_boolean' in experiments:
    n_pns = [int(x) for x in args.pn]
    path = './files/control_pn2kc_prune_boolean'
    if TRAIN:
        for n_pn in n_pns:
            cur_path = path + '_' + str(n_pn)
            train(experiment_controls.control_pn2kc_prune_boolean(n_pn),
                  save_path=cur_path)
    if ANALYZE:
        xkey = 'kc_prune_weak_weights'
        ykeys = ['val_acc', 'K_inferred','K']
        for n_pn in n_pns:
            cur_path = path + '_' + str(n_pn)
            for yk in ykeys:
                sa.plot_progress(cur_path, ykeys=[yk], legend_key=xkey)

            res = standard.analysis_pn2kc_peter.do_everything(cur_path, filter_peaks=False, redo=True, range=1)
            sa.plot_xy(cur_path, xkey='lin_bins_', ykey='lin_hist_', legend_key=xkey, log=res,
                       ax_args={'ylim': [0, 500]})

if 'control_pn2kc_prune_hyper' in experiments:
    n_pns = [int(x) for x in args.pn]
    path = './files/control_pn2kc_prune_hyper'
    if TRAIN:
        for n_pn in n_pns:
            cur_path = path + '_' + str(n_pn)
            train(experiment_controls.control_pn2kc_prune_hyper(n_pn), control=True,
                  save_path=cur_path)
    if ANALYZE:
        for n_pn in n_pns:
            cur_path = path + '_' + str(n_pn)
            default = {'N_KC': 2500, 'lr': 1e-3, 'initial_pn2kc':4./n_pn, 'kc_prune_threshold': 1./n_pn}
            ykeys = ['val_acc', 'K']
            for yk in ykeys:
                exclude_dict = None
                if yk in ['K_inferred', 'sparsity_inferred', 'K', 'sparsity']:
                    # exclude_dict = {'lr': [3e-3, 1e-2, 3e-2]}
                    pass

                for xk, v in default.items():
                    temp = copy.deepcopy(default)
                    temp.pop(xk)
                    logx = True
                    # sa.plot_results(cur_path, xkey=k, ykey=yk, figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
                    #                 select_dict=temp,
                    #                 logx=logx)
                    #
                    # sa.plot_progress(cur_path, select_dict=temp, ykeys=[yk], legend_key=k, exclude_dict=exclude_dict)
            #
            res = standard.analysis_pn2kc_peter.do_everything(cur_path, filter_peaks=True, redo=True, range=.75)
            for xk, v in default.items():
                temp = copy.deepcopy(default)
                temp.pop(xk)
                sa.plot_xy(cur_path, select_dict=temp, xkey='lin_bins_', ykey='lin_hist_', legend_key=xk, log=res,
                           ax_args={'ylim': [0, 500]})

if 'control_vary_pn' in experiments:
    path = './files/control_vary_pn'
    if TRAIN:
        train(experiment_controls.control_vary_pn(), save_path=path)
    if ANALYZE:
        sa.plot_weights(os.path.join(path,'000004'), sort_axis=1, average=False)
        sa.plot_weights(os.path.join(path,'000010'), sort_axis=1, average=False, vlim=[0, 5])
        sa.plot_weights(os.path.join(path,'000022'), sort_axis=1, average=False, vlim=[0, 5])

        ix = 22
        ix_good, ix_bad = analysis_orn2pn.multiglo_gloscores(path, ix, cutoff=.9, shuffle=False)
        analysis_orn2pn.multiglo_pn2kc_distribution(path, ix, ix_good, ix_bad)
        analysis_orn2pn.multiglo_lesion(path, ix, ix_good, ix_bad)

        default = {'kc_dropout_rate': 0.5, 'N_PN':50}
        ykeys = ['val_acc', 'glo_score']
        xticks = [20, 50, 100, 200, 1000]
        for ykey in ykeys:
            sa.plot_results(path, xkey='N_PN', ykey=ykey, figsize=(1.75, 1.75), ax_box=(0.3, 0.3, 0.65, 0.65),
                            loop_key='kc_dropout_rate',
                            logx=True, ax_args={'xticks': xticks}, plot_args={'alpha':0.7})
            sa.plot_results(path, xkey='N_PN', ykey=ykey, figsize=(1.75, 1.75), ax_box=(0.25, 0.25, 0.65, 0.65),
                            loop_key='kc_dropout_rate', select_dict={'kc_dropout_rate':0.5},
                            logx=True, ax_args={'xticks':xticks})
            sa.plot_progress(path, ykeys=[ykey], legend_key='N_PN', select_dict={'kc_dropout_rate':0.5})

if 'control_vary_kc' in experiments:
    path = './files/control_vary_kc'
    if TRAIN:
        train(experiment_controls.control_vary_kc(), save_path=path)
    if ANALYZE:
        sa.plot_weights(os.path.join(path, '000000'), sort_axis=1, average=False)
        sa.plot_weights(os.path.join(path, '000021'), sort_axis=1, average=False)
        # default = {'kc_dropout_rate': 0.5, 'N_KC':2500}
        # ykeys = ['val_acc', 'glo_score']
        # ylim, yticks = [0, 1.1], [0, .25, .5, .75, 1]
        # xticks = [50, 200, 1000, 2500, 10000]
        # for ykey in ykeys:
        #     sa.plot_results(path, xkey='N_KC', ykey=ykey, figsize=(1.75, 1.75), ax_box=(0.3, 0.3, 0.65, 0.65),
        #                     loop_key='kc_dropout_rate',
        #                     logx=True, ax_args={'ylim': ylim, 'yticks': yticks, 'xticks': xticks}, plot_args={'alpha':0.7})
        #     sa.plot_results(path, xkey='N_KC', ykey=ykey, figsize=(1.75, 1.75), ax_box=(0.25, 0.25, 0.65, 0.65),
        #                     loop_key='kc_dropout_rate', select_dict={'kc_dropout_rate':0.5},
        #                     logx=True, ax_args={'ylim': ylim, 'yticks': yticks, 'xticks':xticks})

#TODO
if 'multi_head_prune' in experiments:
    path = './files/multi_head_prune'
    if TRAIN:
        train(se.train_multihead_pruning(is_test), path)

if 'train_kc_claws' in experiments:
    path = './files/train_kc_claws'
    if TRAIN:
        train(standard.experiment_controls.train_claw_configs(is_test), path, sequential=True)
    if ANALYZE:
        sa.plot_progress(
            path, alpha=.75, linestyles=[':', '-'],
            legends=['Trained', 'Fixed']),
        sa.plot_weights(path, var_name='w_glo', sort_axis=-1, dir_ix=1)
        analysis_pn2kc_training.plot_distribution(path)
        analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=False)

if 'controls_receptor' in experiments:
    path = './files/controls_receptor'
    if TRAIN:
        local_train(experiment_controls.controls_receptor(is_test), path, control=True)
    if ANALYZE:
        default = {'N_ORN_DUPLICATION': 10, 'or2orn_normalization': True, 'pn_norm_pre':'batch_norm'}
        sa.plot_results(path, xkey='N_ORN_DUPLICATION', ykey='or_glo_score',
                        select_dict={'or2orn_normalization': True, 'pn_norm_pre':'batch_norm'}),
        sa.plot_results(path, xkey='or2orn_normalization', ykey='or_glo_score',
                        select_dict={'N_ORN_DUPLICATION': 10, 'pn_norm_pre':'batch_norm'})
        sa.plot_results(path, xkey='pn_norm_pre', ykey='or_glo_score',
                        select_dict={'N_ORN_DUPLICATION': 10, 'or2orn_normalization': True})

        sa.plot_results(path, xkey='N_ORN_DUPLICATION', ykey='combined_glo_score',
                        select_dict={'or2orn_normalization': True, 'pn_norm_pre':'batch_norm'}),
        sa.plot_results(path, xkey='or2orn_normalization', ykey='combined_glo_score',
                        select_dict={'N_ORN_DUPLICATION': 10, 'pn_norm_pre':'batch_norm'})
        sa.plot_results(path, xkey='pn_norm_pre', ykey='combined_glo_score',
                        select_dict={'N_ORN_DUPLICATION': 10, 'or2orn_normalization': True})

        sa.plot_results(path, xkey='N_ORN_DUPLICATION', ykey='val_acc',
                        select_dict={'or2orn_normalization': True, 'pn_norm_pre':'batch_norm'}),
        sa.plot_results(path, xkey='or2orn_normalization', ykey='val_acc',
                        select_dict={'N_ORN_DUPLICATION': 10, 'pn_norm_pre':'batch_norm'})
        sa.plot_results(path, xkey='pn_norm_pre', ykey='val_acc',
                        select_dict={'N_ORN_DUPLICATION': 10, 'or2orn_normalization': True})

if 'vary_kc_claws' in experiments:
    path = './files/vary_kc_claws'
    if TRAIN:
        train(standard.experiment_controls.vary_claw_configs(is_test), path)
    if ANALYZE:
        import tools
        t = [1, 2, 9, 19, 29, 39, 49, 59, 69]
        for i in t:
            res = tools.load_all_results(path, argLast=False, ix=i)
            sa.plot_results(path, xkey='kc_inputs', ykey='val_logloss',
                            select_dict={'ORN_NOISE_STD':0}, res=res, string = str(i), figsize=(2, 2))

        sa.plot_progress(path, select_dict = {'kc_inputs':[7,15,30], 'ORN_NOISE_STD':0}, legends=['7', '15', '30'])
        # analysis_activity.sparseness_activity(path, 'kc_out')
        # import tools
        # for i in range(8):
        #     res = tools.load_all_results(path, argLast=False, ix=i)
        #     sa.plot_results(path, xkey='kc_inputs', ykey='train_loss',
        #                     select_dict={'ORN_NOISE_STD':0}, res=res, string = str(i))

        # sa.plot_results(path, xkey='kc_inputs', ykey='val_acc', loop_key='ORN_NOISE_STD',
        #                 figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),)
        sa.plot_results(path, xkey='kc_inputs', ykey='val_acc', select_dict={'ORN_NOISE_STD':0},
                        figsize=(2, 2))
        # sa.plot_results(path, xkey='kc_inputs', ykey='val_logloss', loop_key='ORN_NOISE_STD',
        #                 figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
        #                 ax_args={'ylim':[-1, 2], 'yticks':[-1,0,1,2]})
        sa.plot_results(path, xkey='kc_inputs', ykey='val_logloss', select_dict={'ORN_NOISE_STD': 0},
                        figsize=(2, 2),
                        ax_args={'ylim':[-1, 2], 'yticks':[-1,0,1,2]})

if 'vary_kc_activity_fixed' in experiments:
    # Vary KC activity under different number of relabels
    path = './files/vary_kc_activity_fixed'
    if TRAIN:
        train(se.vary_kc_activity_fixed(is_test), path)
    if ANALYZE:
        # sa.plot_results(path, xkey='n_trueclass', ykey='val_acc', loop_key='kc_dropout_rate')
        analysis_activity.sparseness_activity(path, 'kc_out')
        analysis_activity.plot_mean_activity_sparseness(path, 'kc_out', xkey='n_trueclass', loop_key='kc_dropout_rate')

if 'vary_kc_activity_trainable' in experiments:
    # Vary KC activity under different number of relabels
    path = './files/vary_kc_activity_trainable'
    if TRAIN:
        train(se.vary_kc_activity_trainable(is_test), path)
    if ANALYZE:
        analysis_pn2kc_training.plot_distribution(path)
        analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True)
        # sa.plot_results(path, xkey='n_trueclass', ykey='val_acc', loop_key='kc_dropout_rate')
        # analysis_activity.sparseness_activity(path, 'kc_out')
        # analysis_activity.plot_mean_activity_sparseness(path, 'kc_out', xkey='n_trueclass', loop_key='kc_dropout_rate')

if 'apl' in experiments:
    # Adding inhibitory APL unit.
    path = './files/apl'
    if TRAIN:
        train(standard.experiment_controls.vary_apl(is_test), path)
    if ANALYZE:
        analysis_activity.sparseness_activity(
            path, 'kc_out', activity_threshold=0., lesion_kwargs=None)
        lk = {'name': 'model/apl2kc/kernel:0',
              'units': 0, 'arg': 'outbound'}
        analysis_activity.sparseness_activity(
            path, 'kc_out', activity_threshold=0., lesion_kwargs=lk,
            figname='lesion_apl_')

if 'meansub' in experiments:
    # Subtracting mean from activity
    path = './files/meansub'
    if TRAIN:
        train(standard.experiment_controls.vary_w_glo_meansub_coeff(is_test), path, sequential=True)
    if ANALYZE:
        pass

if 'vary_init_sparse' in experiments:
    # Vary PN2KC initialization to be sparse or dense
    path = './files/vary_init_sparse'
    if TRAIN:
        train(standard.experiment_controls.vary_init_sparse(is_test), path)
    if ANALYZE:
        pass

if 'vary_orn_corr' in experiments:
    # Vary ORN correlation
    path = './files/vary_orn_corr'
    if TRAIN:
        train(standard.experiment_controls.vary_orn_corr(), sequential=True,
              save_path=path)
    if ANALYZE:
        xkey = 'orn_corr'
        ykeys = ['val_acc', 'K_inferred', 'glo_score']
        progress_keys = ['val_logloss', 'train_logloss', 'val_loss',
                         'train_loss', 'val_acc', 'glo_score', 'K_inferred']
        for yk in ykeys:
            sa.plot_results(path, xkey=xkey, ykey=yk, figsize=(3.0, 1.5))
            sa.plot_progress(path, legend_key=xkey, ykeys=progress_keys)

if 'analytical' in experiments:
    if TRAIN:
        numerical_test.get_optimal_K_simulation()
    if ANALYZE:
        numerical_test.main_compare()
        numerical_test.main_plot()
        analyze_simulation_results.main()