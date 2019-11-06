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
import standard.analysis_pn2kc_training
import standard.analysis_activity as analysis_activity

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

#TODO: make code neater

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

def rnn():
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 50
    config.model = 'rnn'

    config.NEURONS = 2500
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

def _easy_weights(w_plot, x_label, y_label, dir_ix, save_path, xticks=None, extra_str ='', vlim = None):
    rect = [0.2, 0.15, 0.6, 0.6]
    rect_cb = [0.82, 0.15, 0.02, 0.6]
    fig = plt.figure(figsize=(2.6, 2.6))
    ax = fig.add_axes(rect)
    if vlim == None:
        vlim = np.round(np.max(abs(w_plot)), decimals=1)

    cmap = tools.get_colormap()
    im = ax.imshow(w_plot, cmap=cmap, vmin=0, vmax=vlim,
                   interpolation='none')
    plt.axis('tight')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if xticks:
        ax.set_xticks(xticks)
    else:
        ax.set_xticks([0, w_plot.shape[1]])

    ax.set_yticks([0, w_plot.shape[0]])
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[0, vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Weight', fontsize=7, labelpad=-10)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.axis('tight')
    tools.save_fig(save_path, '__' + str(dir_ix) + '_' + y_label + '_' + x_label + '_' + extra_str, dpi=400)

def load_activity(save_path):
    import tensorflow as tf
    # # Reload the network and analyze activity
    config = tools.load_config(save_path)
    config.data_dir = r'C:\Users\Peter\PycharmProjects\olfaction_evolution\datasets\proto\orn50'
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

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
figpath = os.path.join(rootpath, 'figures')
figpath = r'C:\Users\Peter\PycharmProjects\olfaction_evolution\figures'
def rnn_distribution(w_glo, dir_ix, path):
    titles = ['Before Training', 'After Training']
    training_ix = 1
    save_path = os.path.join(figpath, path.split('/')[-1])
    fig_name = os.path.join(save_path, 'distribution_' + str(dir_ix) + '_' + str(training_ix))
    standard.analysis_pn2kc_training._plot_distribution(w_glo.flatten(), savename= fig_name, title= titles[training_ix],
                                                        xrange=1.0, yrange=5000)
def rnn_sparsity(w_glo, dir_ix, path):
    titles = ['Before Training', 'After Training']
    training_ix = 1
    yrange = [0.5, 0.5]
    force_thres = 0.05
    thres = standard.analysis_pn2kc_training.infer_threshold(w_glo, visualize=False)
    print('thres=', str(thres))
    claw_count = np.count_nonzero(w_glo>thres,axis=0)

    save_path = os.path.join(figpath, path.split('/')[-1])
    fig_name = os.path.join(save_path, 'sparsity_' + str(dir_ix) + '_' + str(training_ix))
    standard.analysis_pn2kc_training._plot_sparsity(claw_count, savename=fig_name, title= titles[training_ix],
                                                    yrange = yrange[training_ix])

def plot_activity(rnn_outputs, dir_ix, path):
    ## plot summary
    mean_activities = [np.mean(x, axis=0) for x in rnn_outputs]
    neurons_active = [np.sum(x > 0.05) for x in mean_activities]
    log_neurons_active = np.log(neurons_active)
    xticks = [50, 500, 3000]


    fig = plt.figure(figsize=(1.5, 1.2 ))
    ax = fig.add_axes([0.35, 0.3, .15 * len(rnn_outputs), 0.6])
    ax.plot(log_neurons_active, 'o-', markersize=3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Active Neurons')
    ax.set_yticks(np.log(xticks))
    ax.set_yticklabels([str(x) for x in xticks])
    ax.set_xticks(np.arange(len(rnn_outputs)))

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    fig_name = '_activity_' + str(dir_ix)
    tools.save_fig(path, fig_name, pdf=True)


path = './files/test50'
# st(rnn(), path, s=0, e=100)

sa.plot_progress(path, plot_vars=['val_acc', 'val_logloss'], legends=['0','1','2'])

var_name = 'w_rnn'
dirs = [os.path.join(path, n) for n in os.listdir(path)]
dir_ix = 0
save_path = dirs[dir_ix]
config = tools.load_config(save_path)
rnn_outputs = load_activity(save_path)
w_rnns = tools.load_pickle(path, var_name)
w_rnn = w_rnns[dir_ix]

def analyze_t0(w_rnn):
    N_ORN = config.N_ORN * config.N_ORN_DUPLICATION
    w_glo = w_rnn[:N_ORN, N_ORN:]
    w_glo_subsample = w_glo[:, :20]
    _easy_weights(w_glo_subsample, y_label='T=0', x_label='T=1', dir_ix=dir_ix, save_path = path)
    rnn_distribution(w_glo, dir_ix, path)
    rnn_sparsity(w_glo, dir_ix, path)

def analyze_t_greater(w_rnn, time_steps):
    N_OR = config.N_ORN
    N_ORN = config.N_ORN * config.N_ORN_DUPLICATION
    ixs = []
    pn_ixs = []
    for i in range(1, config.TIME_STEPS):
        pn = np.mean(rnn_outputs[i], axis=0)
        ix = np.argsort(pn)[::-1]
        pn_cutoff= np.argmax(pn[ix] < .2)
        pn_ix = ix[:pn_cutoff]
        ixs.append(ix)
        pn_ixs.append(pn_ix)

    plot_activity(rnn_outputs, dir_ix=dir_ix, path=path)

    ## plot activity
    #ORN activation
    x_range = 1000
    _easy_weights(rnn_outputs[0][:100,:x_range], dir_ix=dir_ix, y_label='Odors', x_label='T=0',
                  xticks = [0, 500, x_range],
                  extra_str='focused', save_path=path)
    _easy_weights(rnn_outputs[0][:100,:], dir_ix=dir_ix, y_label='Odors', x_label='T=0',
                  xticks = [0, 500, w_rnn.shape[0]],
                  save_path=path)
    #PN activation, sorted to T=1
    i = 1
    x_range = 100
    _easy_weights(rnn_outputs[i][:100, ixs[0][:x_range]], dir_ix= dir_ix, y_label='Odors',
                  x_label='T=' + str(i),
                  xticks=[0, 50, x_range],
                  extra_str='focused', save_path=path)
    _easy_weights(rnn_outputs[i][:100, ixs[0]], dir_ix= dir_ix, y_label='Odors',
                  x_label='T=' + str(i),
                  save_path=path)

    if time_steps == 3:
        i = 2
        _easy_weights(rnn_outputs[i][:100, ixs[1][:x_range]], dir_ix=dir_ix, y_label='Odors',
                      x_label='T=' + str(i),
                      xticks=[0, 50, x_range],
                      extra_str='focused', save_path=path)
        _easy_weights(rnn_outputs[i][:100, ixs[1]], dir_ix=dir_ix, y_label='Odors',
                      x_label='T=' + str(i),
                      save_path=path)

    #KC activation, sorted to T=1
    _easy_weights(rnn_outputs[time_steps][:100, ixs[0]], dir_ix= dir_ix, y_label='Odors',
                  x_label='T=' + str(time_steps),
                  save_path=path)


    ## plot weight
    w_orn = w_rnn[:N_ORN, pn_ixs[0]]
    ind_max = np.argmax(w_orn, axis=1)
    ind_sort = np.argsort(ind_max)
    w_orn_reshaped = w_orn[ind_sort,:]
    _easy_weights(w_orn_reshaped, y_label='T=0', x_label='T=1', extra_str= 'reshaped', vlim=.4,
                  dir_ix= dir_ix, save_path = path)

    w_orn_mean = tools._reshape_worn(w_orn, N_OR, mode='tile')
    w_orn_mean = w_orn_mean.mean(axis=0)
    ind_max = np.argmax(w_orn_mean, axis=0)
    ind_sort = np.argsort(ind_max)
    w_orn_mean = w_orn_mean[:, ind_sort]
    _easy_weights(w_orn_mean, y_label='T=0', x_label='T=1', extra_str='mean',
                  dir_ix= dir_ix, save_path = path)

    w_glo = w_rnn[pn_ixs[-1], :]
    rnn_distribution(w_glo, dir_ix, path)
    rnn_sparsity(w_glo, dir_ix, path)

    w_glo_sorted = np.sort(w_glo, axis=0)[::-1, :]

    if time_steps == 3:
        pn_to_pn1 = w_rnn[pn_ixs[0][:,None], pn_ixs[1]]
        ind_max = np.argmax(pn_to_pn1, axis=1)
        ind_sort = np.argsort(ind_max)
        pn_to_pn1_reshaped = pn_to_pn1[ind_sort,:]
        _easy_weights(pn_to_pn1_reshaped, dir_ix= dir_ix, y_label='T=1', x_label='T=2', extra_str='sorted',
                      save_path=path)

        _easy_weights(w_glo_sorted, y_label='T=2', x_label='T=3', extra_str='sorted',
                      dir_ix= dir_ix, save_path = path)

        w_glo_subsample = w_glo[:, 1000:1020]
        _easy_weights(w_glo_subsample, y_label='T=2', x_label='T=3', dir_ix=dir_ix, save_path = path)
    else:
        _easy_weights(w_glo_sorted, y_label='T=1', x_label='T=2', extra_str='sorted',
                      dir_ix=dir_ix, save_path=path)

        w_glo_subsample = w_glo[:, 1000:1020]
        _easy_weights(w_glo_subsample, y_label='T=1', x_label='T=2', dir_ix=dir_ix, save_path=path)

N_OR = 50
N_ORN = 500

if dir_ix == 0:
    # analyze_t0(w_rnn)
    analyze_t_greater(w_rnn, config.TIME_STEPS)
else:
    analyze_t_greater(w_rnn, config.TIME_STEPS)


# if config.TIME_STEPS == 1:
#     ixs = [np.arange(w_rnn.shape[0])]
#     pn_ixs = [np.arange(N_OR)]
# else:
#     ixs = []
#     pn_ixs = []
#     for i in range(1, config.TIME_STEPS):
#         pn = np.mean(rnn_outputs[i], axis=0)
#         ix = np.argsort(pn)[::-1]
#         pn_cutoff= np.argmax(pn[ix] < .2)
#         pn_ix = ix[:pn_cutoff]
#         ixs.append(ix)
#         pn_ixs.append(pn_ix)
#
#     # sorted to first layer
#     for i in range(config.TIME_STEPS):
#         _easy_weights(rnn_outputs[i][:, ixs[0]], dir_ix= dir_ix, y_label='odors',
#                       x_label='Sorted to Layer 1, Layer' + '_' + str(i), save_path=path)
#
# _easy_weights(rnn_outputs[0], dir_ix= dir_ix, y_label='odors', x_label='Sorted, Layer_0', save_path=path)
# # sorted to each
# for i, ix in enumerate(ixs):
#     _easy_weights(rnn_outputs[i + 1][:, ix], dir_ix= dir_ix, y_label='odors',
#                   x_label='Sorted, Layer' + '_' + str(i + 1), save_path=path)
#
# w_orn = w_rnn[:N_ORN, pn_ixs[0]]
# w_orn_reshaped = tools._reshape_worn(w_orn, N_OR, mode='tile')
# w_orn_reshaped = w_orn_reshaped.mean(axis=0)
# ind_max = np.argmax(w_orn_reshaped, axis=0)
# ind_sort = np.argsort(ind_max)
# w_orn_reshaped = w_orn_reshaped[:, ind_sort]
#
# w_glo = w_rnn[pn_ixs[-1], :]
# w_glo_sorted = np.sort(w_glo, axis=0)[::-1, :]
#
# if len(pn_ixs) == 2:
#     pn_to_pn1 = w_rnn[pn_ixs[1][:,None], pn_ixs[0]]
#     ind_max = np.argmax(pn_to_pn1, axis=1)
#     ind_sort = np.argsort(ind_max)
#     pn_to_pn1_reshaped = pn_to_pn1[ind_sort,:]
#     _easy_weights(pn_to_pn1_reshaped, dir_ix= dir_ix, y_label='Layer_1', x_label='Layer_2', save_path=path)
#
# _easy_weights(w_rnn, y_label='Input', x_label='Output', dir_ix= dir_ix, save_path = path)
# _easy_weights(w_rnn[:50, ixs[0]], y_label='ORN', x_label='All', dir_ix= dir_ix, save_path = path)
# _easy_weights(w_orn, y_label='ORN', x_label='PN', dir_ix= dir_ix, save_path = path)
# _easy_weights(w_orn_reshaped, y_label='ORN', x_label='sorted PN', dir_ix= dir_ix, save_path = path)
# _easy_weights(w_glo, y_label='PN', x_label='KC', dir_ix= dir_ix, save_path = path)
# _easy_weights(w_glo_sorted, y_label='PN', x_label='KC_sorted', dir_ix= dir_ix, save_path = path)