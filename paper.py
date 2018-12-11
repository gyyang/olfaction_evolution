"""File that summarizes all key results."""

import standard.experiment as standard_experiment
from standard.experiment import local_train
import standard.analysis as standard_analysis

# MODE = 'train'
MODE = 'analysis'

# Reproducing glomeruli-like activity
# save_path = './files/standard/orn2pn'
# if MODE == 'train':
#     standard_experiment.train_orn2pn(save_path)
# else:
#     standard_analysis.plot_progress(save_path)
#     standard_analysis.plot_weights(save_path)
#     # TODO: Add activity distribution

# Varying #PN and #KC
if MODE == 'train':
    pass
else:
    pass

# Varying the noise level while varying #KC
# save_path = './files/vary_noise3'
# if MODE == 'train':
#     local_train(vary_noise_experiment.vary_kc_configs, save_path)
# else:
#     standard_analysis.plot_results(save_path, x_key='N_KC', y_key='glo_score',
#                                    loop_key='ORN_NOISE_STD')
#     standard_analysis.plot_results(save_path, x_key='N_KC', y_key='val_acc',
#                                    loop_key='ORN_NOISE_STD')

# Varying ORN duplication numbers
save_path = './files/vary_n_orn_duplication'
if MODE == 'train':
    local_train(standard_experiment.vary_n_orn_duplication, save_path)
else:
    standard_analysis.plot_results(
        save_path, x_key='N_ORN_DUPLICATION', y_key='glo_score')
    standard_analysis.plot_results(
        save_path, x_key='N_ORN_DUPLICATION', y_key='val_acc')

# Reproducing sparse PN-KC connectivity
if MODE == 'train':
    pass
else:
    pass

# Varying PN-KC connectivity sparseness
if MODE == 'train':
    pass
else:
    pass

# Reproducing random connectivity
if MODE == 'train':
    pass
else:
    pass

# The impact of various normalization
if MODE == 'train':
    pass
else:
    pass

# The impact of various task variants
if MODE == 'train':
    pass
else:
    pass