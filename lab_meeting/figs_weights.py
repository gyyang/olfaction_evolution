import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools
from matplotlib import pylab
import utils
from utils import _load_results, adjust
from utils import plot_weights, plot_small_weights




mpl.rcParams['font.size'] = 12

plt.style.use('dark_background')
fig_dir = 'C:/Users/Peter/Dropbox/TALKS/LAB MEETINGS/2018.11.29'
fig_dir = os.path.join(fig_dir)
root_dir = 'C:/Users/Peter/PycharmProjects/olfaction_evolution/vary_size_experiment/duplication'
save_name = ['.25_noise','one_epoch','no_noise', 'no_training']
for s in save_name:
    dir = os.path.join(root_dir,s, 'files')
    file_folders = [os.path.join(dir, n) for n in os.listdir(dir)]
    w = []


    for i, d in enumerate(file_folders):
        model_dir = os.path.join(d, 'model.pkl')
        with open(model_dir, 'rb') as f:
            var_dict = pickle.load(f)
            w_orn = var_dict['w_orn']
            w.append(w_orn)
            name = os.path.join(fig_dir, s)
            # plot_weights(w_orn, name, 1)

file_path = 'C:/Users/Peter/PycharmProjects/olfaction_evolution/vary_size_experiment/vary_PN/files/09'
model_dir = os.path.join(file_path, 'model.pkl')
with open(model_dir, 'rb') as f:
    var_dict = pickle.load(f)
    w_orn = var_dict['w_orn']
    w.append(w_orn)
    name = os.path.join(fig_dir, 'vary_pn_weight')
    plot_weights(w_orn, name, 1)
