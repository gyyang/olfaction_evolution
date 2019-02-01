import os
import task
import configs
from collections import OrderedDict
import numpy as np
import tools
import train
import standard.analysis as sa
import pickle
import model as network_models
import standard.analysis_activity as analysis_activity

import matplotlib.pyplot as plt

def t(experiment, save_path,s=0,e=1000):
    """Train all models locally."""
    for i in range(s, e):
        config = tools.varying_config(experiment, i)
        if config:
            print('[***] Hyper-parameter: %2d' % i)
            config.save_path = os.path.join(save_path, str(i).zfill(6))
            train.train(config)

def st(experiment, save_path, s=0,e=1000):
    """Train all models locally."""
    for i in range(s, e):
        config = tools.varying_config_sequential(experiment, i)
        if config:
            print('[***] Hyper-parameter: %2d' % i)
            config.save_path = os.path.join(save_path, str(i).zfill(6))
            train.train(config)

def temp_rnn():
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 16
    config.model = 'rnn'

    config.replicate_orn_with_tiling = True
    config.N_ORN_DUPLICATION = 10
    config.ORN_NOISE_STD = 0
    config.NEURONS = 2000
    config.TIME_STEPS = 2
    config.WEIGHT_LOSS = False
    config.WEIGHT_ALPHA = 0
    config.BATCH_NORM = False
    config.DIAGONAL_INIT = True

    config.dropout = True
    config.dropout_rate = .5

    hp_ranges = OrderedDict()
    hp_ranges['TIME_STEPS'] = [1, 2, 3]
    hp_ranges['replicate_orn_with_tiling'] = [False, True, True]
    hp_ranges['N_ORN_DUPLICATION'] = [1, 10, 10]
    return config, hp_ranges

def temp_rnn_relabel():
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/200_20'
    config.max_epoch = 10
    config.model = 'rnn'

    config.replicate_orn_with_tiling = True
    config.N_ORN_DUPLICATION = 10
    config.ORN_NOISE_STD = 0.2
    config.NEURONS = 2000
    config.TIME_STEPS = 3
    config.WEIGHT_LOSS = False
    config.WEIGHT_ALPHA = 0
    config.BATCH_NORM = False
    config.DIAGONAL_INIT = True

    config.dropout = True
    config.dropout_rate = .5

    hp_ranges = OrderedDict()
    hp_ranges['TIME_STEPS'] = [1, 2, 3]
    return config, hp_ranges

def temp_normal():
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 8
    config.model = 'full'

    config.replicate_orn_with_tiling = False
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.skip_orn2pn = True
    config.kc_dropout = True
    hp_ranges = OrderedDict()
    hp_ranges['dummy'] = [True]
    return config, hp_ranges

def _easy_weights(w_plot, x_label, y_label, dir_ix, save_path):
    rect = [0.15, 0.15, 0.65, 0.65]
    rect_cb = [0.82, 0.15, 0.02, 0.65]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes(rect)
    vlim = np.round(np.max(abs(w_plot)), decimals=1)
    im = ax.imshow(w_plot, cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                   interpolation='none')
    plt.axis('tight')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks([0, w_plot.shape[1]])
    ax.set_yticks([0, w_plot.shape[0]])
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[-vlim, vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Weight', fontsize=7, labelpad=-10)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.axis('tight')
    sa._easy_save(save_path, '__' + str(dir_ix) + '_' + y_label + '_' + x_label, dpi=400)

def load_activity(save_path):
    import tensorflow as tf
    # # Reload the network and analyze activity
    config = tools.load_config(save_path)
    train_x, train_y, val_x, val_y = task.load_data(config.dataset, config.data_dir)

    tf.reset_default_graph()
    CurrentModel = network_models.RNN
    val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)
    model = CurrentModel(val_x_ph, val_y_ph, config=config, training=False)
    model.save_path = save_path

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.load()
        rnn_outputs = sess.run(model.rnn_outputs, {val_x_ph: val_x, val_y_ph: val_y})
    return rnn_outputs

path = './files_temp/RNN_relabel'
# st(temp_rnn(), path, s=0, e=100)
t(temp_rnn_relabel(), path, s=1, e=2)

var_name = 'w_rnn'
dirs = [os.path.join(path, n) for n in os.listdir(path)]
dir_ix = 0
save_path = dirs[dir_ix]
config = tools.load_config(save_path)
rnn_outputs = load_activity(save_path)
w_rnns = tools.load_pickle(path, var_name)
w_rnn = w_rnns[dir_ix]

N_OR = config.N_ORN
N_ORN = config.N_ORN * config.N_ORN_DUPLICATION

if config.TIME_STEPS == 1:
    ixs = [np.arange(w_rnn.shape[0])]
    pn_ixs = [np.arange(N_OR)]
else:
    ixs = []
    pn_ixs = []
    for i in range(1, config.TIME_STEPS):
        pn = np.mean(rnn_outputs[i], axis=0)
        ix = np.argsort(pn)[::-1]
        pn_cutoff= np.argmax(pn[ix] < .2)
        pn_ix = ix[:pn_cutoff]
        ixs.append(ix)
        pn_ixs.append(pn_ix)

    # sorted to first layer
    for i in range(config.TIME_STEPS):
        _easy_weights(rnn_outputs[i][:, ixs[0]], dir_ix= dir_ix, y_label='odors',
                      x_label='Sorted to Layer 1, Layer' + '_' + str(i), save_path=path)

_easy_weights(rnn_outputs[0], dir_ix= dir_ix, y_label='odors', x_label='Sorted, Layer_0', save_path=path)
# sorted to each
for i, ix in enumerate(ixs):
    _easy_weights(rnn_outputs[i + 1][:, ix], dir_ix= dir_ix, y_label='odors',
                  x_label='Sorted, Layer' + '_' + str(i + 1), save_path=path)


w_orn = w_rnn[:N_ORN, pn_ixs[0]]
w_orn_reshaped = tools._reshape_worn(w_orn, N_OR, mode='tile')
w_orn_reshaped = w_orn_reshaped.mean(axis=0)
ind_max = np.argmax(w_orn_reshaped, axis=0)
ind_sort = np.argsort(ind_max)
w_orn_reshaped = w_orn_reshaped[:, ind_sort]

w_glo = w_rnn[pn_ixs[-1], :]
w_glo_sorted = np.sort(w_glo, axis=0)[::-1, :]

if len(pn_ixs) == 2:
    pn_to_pn1 = w_rnn[pn_ixs[1][:,None], pn_ixs[0]]
    ind_max = np.argmax(pn_to_pn1, axis=1)
    ind_sort = np.argsort(ind_max)
    pn_to_pn1_reshaped = pn_to_pn1[ind_sort,:]
    _easy_weights(pn_to_pn1_reshaped, dir_ix= dir_ix, y_label='Layer_1', x_label='Layer_2', save_path=path)

_easy_weights(w_rnn, y_label='Input', x_label='Output', dir_ix= dir_ix, save_path = path)
_easy_weights(w_rnn[:N_ORN, ixs[0]], y_label='ORN', x_label='All', dir_ix= dir_ix, save_path = path)
_easy_weights(w_orn, y_label='ORN', x_label='PN', dir_ix= dir_ix, save_path = path)
_easy_weights(w_orn_reshaped, y_label='ORN', x_label='sorted PN', dir_ix= dir_ix, save_path = path)
_easy_weights(w_glo, y_label='PN', x_label='KC', dir_ix= dir_ix, save_path = path)
_easy_weights(w_glo_sorted, y_label='PN', x_label='KC_sorted', dir_ix= dir_ix, save_path = path)

threshold = 0.08
claw_count = np.sum(w_glo>threshold,axis=0)
plt.hist(w_glo.flatten(), bins= 100, range=[0.05, 1], density=False)
sa._easy_save(path, str= '__' + str(dir_ix) + '_KC_histogram')

plt.hist(claw_count, bins=N_OR, range=[0, N_OR], density=False)
sa._easy_save(path, str = '__' + str(dir_ix) + '_claw_distribution')