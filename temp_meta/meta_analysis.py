import os
import sys
from pathlib import Path

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import standard.analysis as sa
import standard.analysis_pn2kc_training as analysis_pn2kc_training
import tools

path = Path('..') / 'files' / 'meta_num_updates'

modeldir = tools.get_modeldirs(path)[0]

# sa.plot_weights(path, var_name='w_orn', sort_axis=0, average=True)
# sa.plot_weights(path, var_name='w_glo', sort_axis=None)
sa.plot_progress(modeldir, ykeys=['val_acc', 'K_smart', 'val_loss', 
                                  'train_pre_acc', 'train_pre_loss',
                                  'train_post_acc', 'train_post_loss'])
sa.plot_weights(modeldir)
sa.plot_xy(modeldir, xkey='lin_bins', ykey='lin_hist')
analysis_pn2kc_training.plot_sparsity(modeldir)
