"""File that summarizes all key results."""

import standard.experiment as standard_experiment
from standard.hyper_parameter_train import basic_train, local_train, local_sequential_train
from train import train
import standard.analysis as standard_analysis
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TRAIN = False
ANALYZE = True
TESTING_EXPERIMENTS = True
run_ix = [5]


save_paths = ['./files/orn2pn',
              './files/vary_ORN_duplication',
              './files/vary_PN',
              './files/vary_KC',
              './files/vary_KC_claws',
              './files/train_KC_claws'
              ]

experiments = [standard_experiment.train_orn2pn, # Reproducing glomeruli-like activity
               standard_experiment.vary_orn_duplication_configs, # Vary ORN n duplication under different nKC
               standard_experiment.vary_pn_configs, # Vary nPN under different noise levels
               standard_experiment.vary_kc_configs, # Vary nKC under different noise levels
               standard_experiment.vary_claw_configs, # Vary nClaws under different noise levels
               standard_experiment.train_claw_configs, # Compare training vs fixing weights for PN2KC
               ]

run_methods = [local_train,
               local_train,
               local_train,
               local_train,
               local_train,
               local_sequential_train
               ]

analysis_methods_per_experiment = [
    [standard_analysis.plot_progress,
     standard_analysis.plot_weights],
    [lambda x: standard_analysis.plot_results(x, x_key='N_ORN_DUPLICATION', y_key='glo_score',
                                              loop_key='N_KC'),
     lambda x: standard_analysis.plot_results(x, x_key='N_ORN_DUPLICATION', y_key='val_acc',
                                              loop_key='N_KC')],
    [lambda x: standard_analysis.plot_results(x, x_key='N_PN', y_key='glo_score',
                                              loop_key='ORN_NOISE_STD'),
     lambda x: standard_analysis.plot_results(x, x_key='N_PN', y_key='val_acc',
                                              loop_key='ORN_NOISE_STD')],
    [lambda x: standard_analysis.plot_results(x, x_key='N_KC', y_key='glo_score',
                                              loop_key='ORN_NOISE_STD'),
     lambda x: standard_analysis.plot_results(x, x_key='N_KC', y_key='val_acc',
                                              loop_key='ORN_NOISE_STD')],
    [lambda x: standard_analysis.plot_results(x, x_key='kc_inputs', y_key='val_acc',
                                              loop_key='ORN_NOISE_STD')],
    [standard_analysis.plot_progress]
]

def wrapper(experiment):
    return lambda: experiment(TESTING_EXPERIMENTS)

for i in run_ix:
    save_path = save_paths[i]
    experiment = wrapper(experiments[i])
    run_method = run_methods[i]
    analysis_methods = analysis_methods_per_experiment[i]
    if TRAIN:
        run_method(experiment, save_path)

    for analysis_method in analysis_methods:
        analysis_method(save_path)




# # Reproducing glomeruli-like activity
# save_path = './files/standard/orn2pn'
# if MODE == 'train':
#     standard_experiment.train_orn2pn(save_path)
# else:
#     standard_analysis.plot_progress(save_path)
#     standard_analysis.plot_weights(save_path)
#     # TODO: Add activity distribution
#
# # Vary nPN under different noise levels
# save_path = './files/vary_PN'
# if MODE == 'train':
#     local_train(wrapper(standard_experiment.vary_pn_configs), save_path)
# else:
#     standard_analysis.plot_results(save_path, x_key='N_PN', y_key='glo_score',
#                                    loop_key='ORN_NOISE_STD')
#     standard_analysis.plot_results(save_path, x_key='N_PN', y_key='val_acc',
#                                    loop_key='ORN_NOISE_STD')
#
# # Vary nKC under different noise levels
# save_path = './files/vary_KC'
# if MODE == 'train':
#     local_train(standard_experiment.vary_kc_configs, save_path)
# else:
#     standard_analysis.plot_results(save_path, x_key='N_KC', y_key='glo_score',
#                                    loop_key='ORN_NOISE_STD')
#     standard_analysis.plot_results(save_path, x_key='N_KC', y_key='val_acc',
#                                    loop_key='ORN_NOISE_STD')
#
# # Vary ORN n duplication under different nKC
# save_path = './files/vary_n_orn_duplication'
# if MODE == 'train':
#     local_train(standard_experiment.vary_orn_duplication_configs, save_path)
# else:
#     standard_analysis.plot_results(
#         save_path, x_key='N_ORN_DUPLICATION', y_key='glo_score')
#     standard_analysis.plot_results(
#         save_path, x_key='N_ORN_DUPLICATION', y_key='val_acc')

# Reproducing sparse PN-KC connectivity
if TRAIN == 'train':
    pass
else:
    pass

# Varying PN-KC connectivity sparseness
if TRAIN == 'train':
    pass
else:
    pass

# Reproducing random connectivity
if TRAIN == 'train':
    pass
else:
    pass

# The impact of various normalization
if TRAIN == 'train':
    pass
else:
    pass

# The impact of various task variants
if TRAIN == 'train':
    pass
else:
    pass

# A search of various hyperparameters
if TRAIN == 'train':
    pass
else:
    pass