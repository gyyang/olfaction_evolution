import os
import standard.analysis as sa
import standard.analysis_pn2kc_training as analysis_pn2kc_training

path = r'..\files\torch_metalearn\000000'
# sa.plot_weights(path, var_name='w_orn', sort_axis=0, average=True)
# sa.plot_weights(path, var_name='w_glo', sort_axis=None)
sa.plot_weights(path)
analysis_pn2kc_training.plot_distribution(path, xrange=1)
analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True, thres=.05)