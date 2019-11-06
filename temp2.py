import os
import standard.analysis as sa

path = './files_temp/meta'
folder = '0'
ix = '8000'
sa.plot_weights(os.path.join(path, folder,'epoch', ix), var_name='w_orn', sort_axis=1, dir_ix=-0, average=False)
sa.plot_weights(os.path.join(path, folder,'epoch', ix), var_name='w_glo', sort_axis=-1, dir_ix=0)
import standard.analysis_pn2kc_training
standard.analysis_pn2kc_training.plot_distribution(path, xrange=1, log=True)
standard.analysis_pn2kc_training.plot_distribution(path, xrange=1)
standard.analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True, thres=.01, epochs=[-1])