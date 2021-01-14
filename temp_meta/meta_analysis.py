import os
import sys
from pathlib import Path

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import standard.analysis as sa
import standard.analysis_pn2kc_training as analysis_pn2kc_training
import tools

path = Path('..') / 'files' / 'torch_metalearn'

modeldir = tools.get_modeldirs(path)[0]

# sa.plot_weights(path, var_name='w_orn', sort_axis=0, average=True)
# sa.plot_weights(path, var_name='w_glo', sort_axis=None)
sa.plot_weights(modeldir)
analysis_pn2kc_training.plot_distribution(modeldir, xrange=1)
analysis_pn2kc_training.plot_sparsity(modeldir, dynamic_thres=True, thres=.05)