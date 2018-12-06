import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools
import utils

param = "kc_inputs"
condition = "test"

mpl.rcParams['font.size'] = 7
fig_dir = os.path.join(os.getcwd(), condition, 'figures')
dir = os.path.join(os.getcwd(), condition, 'files')
dirs = [os.path.join(dir, n) for n in os.listdir(dir)]

kcs = []
for d in dirs:
    wglo_names = [os.path.join(d, n) for n in os.listdir(d) if n[:5] == 'w_glo']
    for wglo_names




    glo_act, glo_in, glo_in_pre, kc = utils.get_model_vars(d)
    kcs.append(kc)




configs, glo_score, val_acc, val_loss, train_loss = utils.load_results(dir)
parameters = [getattr(config, param) for config in configs]
list_of_legends = [param +': ' + str(n) for n in parameters]
data = [glo_score, val_acc, val_loss, train_loss]
titles = ['glo score', 'val acc', 'val loss', 'train loss']
