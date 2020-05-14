"""Evaluate a trained network with different amount of noise."""

import sys
import os
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import configs
import task
from model import FullModel
import tools
from tools import save_fig
from tools import nicename

mpl.rcParams['font.size'] = 7


def _evaluate(name, value, model, model_dir, n_rep=1):
    assert name is not 'weight_perturb'
    if model == 'oracle':
        path = os.path.join(rootpath, 'files', oracle_dir, '000000')
    else:
        path = model_dir

    config = tools.load_config(path)

    # TODO: clean up these paths
    config.data_dir = rootpath + config.data_dir[1:]
    config.save_path = rootpath + config.save_path[1:]

    if name == 'orn_noise_std':
        config.ORN_NOISE_STD = value
    elif name == 'orn_dropout_rate':
        config.orn_dropout = True
        config.orn_dropout_rate = value
    elif name == 'alpha':
        config.oracle_scale = value
    else:
        raise ValueError('Unknown name', str(name))

    tf.reset_default_graph()
    # Build validation model
    val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)

    if model == 'oracle':
        # Over-write config
        config.skip_orn2pn = True
        config.skip_pn2kc = True
        config.kc_dropout = False
        config.set_oracle = True
        # Helper model for oracle
        oracle_x_ph = tf.placeholder(val_x.dtype, [config.N_CLASS, val_x.shape[1]])
        oracle_y_ph = tf.placeholder(val_y.dtype, [config.N_CLASS])
        oracle = FullModel(oracle_x_ph, oracle_y_ph, config=config, training=False)
    
    val_model = FullModel(val_x_ph, val_y_ph, config=config, training=False)
    
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        if model == 'oracle':
            oracle.set_oracle_weights()
        else:
            val_model.load()

        # Validation
        val_loss, val_acc = sess.run(
            [val_model.loss, val_model.acc],
            {val_x_ph: val_x, val_y_ph: val_y})

    return val_loss, val_acc


def _select_random_directions(weight):
    """Select normalized random direction given a weight matrix.

    Args:
        weight: numpy array (n_pre, n_post). It is important that the matrix
            is oriented such that the n_pre is the first dimension

    Return:
        d: a direction in weight space in the same shape as weight
    """
    d = np.random.randn(*weight.shape)
    d /= np.linalg.norm(d, axis=0)
    d *= np.linalg.norm(weight, axis=0)
    return d


def select_random_directions(weights):
    """Select normalized random directions in the weight space."""
    return [_select_random_directions(w) for w in weights]


def evaluate_weight_perturb(values, modelname, model_dir, n_rep=1, dataset='val',
                            perturb_mode='multiplicative', epoch=None,
                            multidirection=False, perturb_output=True):
    """Evaluate the performance under weight perturbation.

    Args:
        values: a list of floats about the strength of perturbations
        modelname: str, the model name
        model_dir: str, the model directory
        n_rep: int, the number of repetition
        dataset: 'train' or 'val', the dataset for computing loss
        perturb_mode: 'feature_norm' or 'multiplicative'.
            If 'feature_norm', uses feature-normalized perturbation
            If 'multiplicative', uses independent multiplicative perturbation
        epoch: int or None. If int, analyze the results at specific training epoch
        multidirection: int or False. if not False, then the perturbation
            will be along multiple directions, values must be list of (multidirection)-tuple

    Return:
        losses: a np array of losses, the same size as values
        accs: np array of accuracies, the same size as values
    """
    print('Perturbation mode: ' + perturb_mode)

    if modelname == 'oracle':
        path = os.path.join(rootpath, 'files', oracle_dir, '000000')
    else:
        path = model_dir

    config = tools.load_config(path)

    # TODO: clean up these paths
    config.data_dir = rootpath + config.data_dir[1:]
    config.save_path = rootpath + config.save_path[1:]

    if config.data_dir != data_dir:
        train_x, train_y, val_x, val_y = task.load_data(config.data_dir)
    else:
        train_x, train_y, val_x, val_y = TRAIN_X, TRAIN_Y, VAL_X, VAL_Y

    tf.reset_default_graph()
    # Build validation model
    val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)

    if modelname == 'oracle':
        # Over-write config
        config.skip_orn2pn = True
        config.skip_pn2kc = True
        config.kc_dropout = False
        config.set_oracle = True
        config.oracle_scale = 8
        # Helper model for oracle
        oracle_x_ph = tf.placeholder(val_x.dtype,
                                     [config.N_CLASS, val_x.shape[1]])
        oracle_y_ph = tf.placeholder(val_y.dtype, [config.N_CLASS])
        oracle = FullModel(oracle_x_ph, oracle_y_ph, config=config,
                           training=False)

    val_model = FullModel(val_x_ph, val_y_ph, config=config, training=False)
    
    if epoch is not None:
        val_model.save_path = os.path.join(
                val_model.save_path, 'epoch', str(epoch).zfill(4))

    # Variables to perturb
    perturb_var = None
    if perturb_output:
        perturb_var = ['model/layer3/kernel:0']
    else:
        perturb_var = ['model/layer2/kernel:0']

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        if modelname == 'oracle':
            oracle.set_oracle_weights()
        else:
            val_model.load()

        if dataset == 'val':
            data_x, data_y = val_x, val_y
        elif dataset == 'train':
            rnd_ind = np.random.choice(
                train_x.shape[0], size=(val_x.shape[0],), replace=False)
            data_x, data_y = train_x[rnd_ind], train_y[rnd_ind]
        else:
            raise ValueError('Wrong dataset type')

        print('Perturbing weights:')
        for v in perturb_var:
            print(v)

        if perturb_var is None:
            perturb_var = tf.trainable_variables()
        else:
            perturb_var = [v for v in tf.trainable_variables() if
                           v.name in perturb_var]

        origin_weights = [sess.run(v) for v in perturb_var]
        
        w0 = origin_weights[0]
        plt.figure()
        plt.imshow(w0, aspect='auto')
        plt.figure()
        plt.hist(w0.flatten())

        val_loss = np.zeros((n_rep, len(values)))
        val_acc = np.zeros((n_rep, len(values)))
        angle = np.zeros((n_rep, len(values)))

        for i_rep, rep in enumerate(range(n_rep)):
            if perturb_mode == 'feature_norm':
                if multidirection:
                    directions_list = [select_random_directions(origin_weights)
                                       for _ in range(multidirection)]
                else:
                    directions = select_random_directions(origin_weights)

            for i_value, value in enumerate(values):
                if perturb_mode == 'multiplicative':
                    new_var_val = [w*np.random.uniform(1-value, 1+value, size=w.shape)
                                   for w in origin_weights]
# =============================================================================
#                     new_var_val = [w.copy()
#                                    for w in origin_weights]
#                     new_var_val = list()
#                     for w in origin_weights:
#                         w0 = w.copy()
#                         np.random.shuffle(w0)
#                         new_var_val.append(w0)
#                         plt.figure()
#                         plt.imshow(w0, aspect='auto')
#                         plt.figure()
#                         plt.hist(w0.flatten())
#                         raise ValueError()
# =============================================================================
                        
                elif perturb_mode == 'feature_norm':
                    if multidirection:
                        new_var_val = list()
                        for i_w, w in enumerate(origin_weights):
                            new_w = 0
                            for v, d_list in zip(value, directions_list):
                                new_w += d_list[i_w] * v
                            new_w += w
                            new_var_val.append(new_w)
                    else:
                        new_var_val = list()
                        for w, d in zip(origin_weights, directions):
                            new_var_val.append(w+d*value)
                else:
                    raise ValueError()

                for j in range(len(perturb_var)):
                    sess.run(perturb_var[j].assign(new_var_val[j]))

                # Validation
                val_loss_tmp, val_acc_tmp, kc_tmp = sess.run(
                    [val_model.loss, val_model.acc, val_model.kc],
                    {val_x_ph: data_x, val_y_ph: data_y})

                print('Perturbation value: ', str(value))
                print('KC coding level: ', str(np.mean(kc_tmp > 0.)))
                # Compute KC angle
                if i_value == 0:
                    if value != 0:
                        raise ValueError(
                            'First perturbation value should be 0')
                    kc_original = kc_tmp
                else:
                    angle_tmp = _angle(kc_tmp, kc_original)
                    # plt.figure()
                    # plt.hist(angle_tmp, bins=50)
                    angle[i_rep, i_value] = np.mean(angle_tmp)

                val_loss[i_rep, i_value] = val_loss_tmp
                val_acc[i_rep, i_value] = val_acc_tmp

    results = {'loss': val_loss, 'acc': val_acc, 'angle': angle}

    return results


def _angle(Y, Y2):
    """Compute angle between two sets of vectors.

    Args:
        Y, Y2: (n_vecs, dim)

    Returns:
        theta: the angle between the n_vecs pairs of dim-D vectors
    """
    norm_Y = np.linalg.norm(Y, axis=1)
    norm_Y2 = np.linalg.norm(Y2, axis=1)

    cos_theta = (np.sum(Y * Y2, axis=1) / (norm_Y * norm_Y2))
    cos_theta = cos_theta[(norm_Y * norm_Y2) > 0]
    theta = np.arccos(cos_theta) / np.pi * 180
    return theta

def evaluate(name, values, model, model_dir, n_rep=1):
    losses = list()
    accs = list()
    for value in values:
        val_loss, val_acc = _evaluate(name, value, model, model_dir,
                                      n_rep=n_rep)
        losses.append(val_loss)
        accs.append(val_acc)
    return losses, accs


def evaluate_kcrole(path, name):
    """Evaluate KC layer's role."""
    if name == 'orn_dropout_rate':
        values = np.linspace(0, 0.3, 10)
    elif name == 'orn_noise_std':
        values = np.linspace(0, 0.3, 10)
    elif name == 'alpha':
        values = np.linspace(0.2, 8, 10)
    elif name == 'weight_perturb':
        values = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    else:
        raise ValueError()

    models = ['oracle', 'sgd + no kc', 'sgd + trained kc', 'sgd + fixed kc']
    model_dirs = ['none', '000002', '000000', '000001']
    # models = ['oracle', 'sgd + no kc', 'sgd + trained kc']
    # model_dirs = ['none', '000002', '000000']
    loss_dict = {}
    acc_dict = {}
    for model, model_dir in zip(models, model_dirs):
        model_dir = os.path.join(path, model_dir)
        if name == 'weight_perturb':
            results = evaluate_weight_perturb(
                values, model, model_dir, n_rep=1)
            losses = results['loss']
            accs = results['acc']
        else:
            losses, accs = evaluate(name, values, model, model_dir)
        loss_dict[model] = losses
        acc_dict[model] = accs

    results = {'loss_dict': loss_dict,
               'acc_dict': acc_dict,
               'models': models,
               'values': values,
               'name': name}

    file = os.path.join(path, 'vary_' + name + '.pkl')
    with open(file, 'wb') as f:
        pickle.dump(results, f)


def plot_kcrole(path, name):
    file = os.path.join(path, 'vary_' + name + '.pkl')
    with open(file, 'rb') as f:
        results = pickle.load(f)

    values = results['values']
    loss_dict = results['loss_dict']
    acc_dict = results['acc_dict']
    # models = results['models']
    models = ['oracle', 'sgd + no kc', 'sgd + trained kc']
    for ylabel in ['val_acc', 'val_loss']:
        res_dict = acc_dict if ylabel == 'val_acc' else loss_dict
        fig = plt.figure(figsize=(2.75, 2))
        ax = fig.add_axes([0.2, 0.2, 0.4, 0.6])
        for model in models:
            res_plot = res_dict[model]
            if ylabel == 'val_loss':
                res_plot = np.log(res_plot)
            ax.plot(values, res_plot, 'o-', markersize=3, label=model)
        plt.xlabel(nicename(name))
        plt.ylabel(nicename(ylabel))
        plt.legend(loc=2, bbox_to_anchor=(1.0, 1.0), frameon=False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        if ylabel == 'val_acc':
            plt.ylim([0, 1.05])
            ax.set_yticks([0, 0.5, 1.0])
        else:
            ax.set_yticks([-2, -1, 0, 1, 2, 3, 4])
            plt.ylim([-2, 4])
        figname = ylabel + '_' + name
        save_fig('kc_role', figname)


def select_config(config, select_dict):
    if select_dict is not None:
        for key, val in select_dict.items():
            if getattr(config, key) != val:
                return False
    return True

def evaluate_across_epochs(path, values=None, select_dict=None, dataset='val', mode = 'angle',
                          file=None, n_rep=1, epoch=None, multidirection=False):
    """Evaluate models from the same root directory."""
    name = 'weight_perturb'
    model_dirs = tools.get_allmodeldirs(path)
    model_dir = model_dirs[0]

    loss_dict = {}
    acc_dict = {}

    model_var = 'epoch'
    models = list()
    epochs = len(tools.get_allmodeldirs(os.path.join(model_dir,'epoch')))

    if mode == 'angle':
        raise NotImplementedError('Not implemented yet')

    for model in range(epochs):
        results = evaluate_weight_perturb(
            values, model, model_dir, n_rep=n_rep, perturb_output=False, perturb_mode='multiplicative',
            dataset=dataset, epoch=model, multidirection=multidirection)

        loss_dict[model] = results['loss']
        acc_dict[model] = results['acc']
        models.append(model)

    results = {'loss_dict': loss_dict,
               'acc_dict': acc_dict,
               'models': models,
               'model_var': model_var,
               'values': values,
               'name': name}

    if file is None:
        file = os.path.join(path, name + '_' + model_var + '_' + dataset)
        if epoch is not None:
            file = file + 'ep' + str(epoch)
    else:
        file = os.path.join(path, file)

    with open(file + '.pkl', 'wb') as f:
        pickle.dump(results, f)

def evaluate_acrossmodels(path, values=None, select_dict=None, dataset='val',
                          file=None, n_rep=1, epoch=None, multidirection=False):
    """Evaluate models from the same root directory.

    Args:
        path: path of models
        values: list of weight perturbation values
        select_dict: dictionary of conditions to select models
        dataset: whether to evaluate on val or train
        file: file to store results, if None then use default value
        n_rep: number of repetition
        epoch: the training epoch to analyze
        multidirection: whether perturbation is applied to multiple directions
    """
    name = 'weight_perturb'

    model_dirs = tools.get_allmodeldirs(path)

    loss_dict = {}
    acc_dict = {}
    angle_dict = {}

    models = list()
    model_var = 'kc_inputs'

    for model_dir in model_dirs:
        config = tools.load_config(model_dir)
        if not select_config(config, select_dict):
            continue
        
        model = getattr(config, model_var)
        results = evaluate_weight_perturb(
            values, model, model_dir, n_rep=n_rep, perturb_output=False,
            dataset=dataset, epoch=epoch, multidirection=multidirection)

        loss_dict[model] = results['loss']
        acc_dict[model] = results['acc']
        angle_dict[model] = results['angle']
        models.append(model)

    results = {'loss_dict': loss_dict,
               'acc_dict': acc_dict,
               'angle_dict': angle_dict,
               'models': models,
               'model_var': model_var,
               'values': values,
               'name': name}

    if file is None:
        file = os.path.join(path, name + '_' + model_var+ '_' + dataset)
        if epoch is not None:
            file = file + 'ep' + str(epoch)
    else:
        file = os.path.join(path, file)

    with open(file+'.pkl', 'wb') as f:
        pickle.dump(results, f)


def plot_acrossmodels(path, model_var='kc_inputs', dataset='val', file=None, epoch=None):
    name = 'weight_perturb'

    if file is None:
        file = os.path.join(path, name + '_' + model_var+ '_' + dataset)
        if epoch is not None:
            file = file + 'ep' + str(epoch)
    else:
        file = os.path.join(path, file)
    
    with open(file + '.pkl', 'rb') as f:
        results = pickle.load(f)

    values = results['values']
    models = results['models']

    if len(values) > 1:
        colors = plt.cm.cool(np.linspace(0, 1, len(values)))
    else:
        colors = [tools.blue]

    diff_dict = dict()
    for ylabel in ['val_acc', 'val_loss', 'angle']:
        if ylabel == 'val_acc':
            res_dict = results['acc_dict']
        elif ylabel == 'val_loss':
            res_dict = results['loss_dict']
        elif ylabel == 'angle':
            res_dict = results['angle_dict']
        else:
            raise ValueError('Unknown ylabel', str(ylabel))

        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_axes((0.2, 0.2, 0.5, 0.5))

        for i in range(len(values)):
            try:
                res_plot = [res_dict[model][:,i].mean(axis=0) for model in models]
            except IndexError:
                res_plot = [res_dict[model][i] for model in models]
            res_plot = np.array(res_plot)
            if ylabel == 'val_logloss':
                res_plot = np.log(res_plot)  # TODO: this log?
            if i == 0:
                diff_dict[ylabel] = res_plot
            if i == len(values) - 1:
                diff_dict[ylabel] = res_plot - diff_dict[ylabel]  # compute diff

            ax.plot(models, res_plot, 'o-', markersize=3, label=values[i], color=colors[i])
        ax.set_xlabel(nicename(model_var))
        ax.set_ylabel(nicename(ylabel))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks([3, 7, 15, 30])
        ax.plot([7, 7], [ax.get_ylim()[0], ax.get_ylim()[-1]], '--', color = 'gray')
        
        plt.locator_params(axis='y', nbins=2)
# =============================================================================
#         if ylabel == 'val_acc':
#             ax.set_yticks([0.8, 0.9, 1.0])
#             yrange = [0.8, 1]
#         else:
#             ax.set_yticks([-2, -1, 0, 1])
#             yrange = [-2.5, -0.5]
#         ax.set_xticks([3, 7, 15, 30, 50])
#         plt.ylim(yrange)
# =============================================================================
        # ax.plot([7, 7], yrange, '--', color='gray')
        if len(values) > 1:
            l = ax.legend(loc=2, bbox_to_anchor=(1.0, 1.0), frameon = False)
        if dataset == 'train':
            title_txt = nicename(dataset) + ' '
        else:
            title_txt = ''
        figname = ylabel+model_var+name+dataset
        if epoch is not None:
            title_txt += 'Epoch ' + str(epoch)
            figname = figname + 'ep' + str(epoch)
            
        plt.title(title_txt, fontsize=7)
        save_fig(path.split('/')[-1], figname)


    for ylabel in ['val_acc', 'val_loss']:
        plt.figure()
        # plt.scatter(diff_dict['angle'], diff_dict['val_loss'])
        plt.scatter(diff_dict['angle'][1:], diff_dict[ylabel][1:])  # TODO: TEMP, ignoring first point
        plt.xlabel('Angle')
        plt.ylabel('Change in ' + ylabel + ' (perturb - original)')
        save_fig(path.split('/')[-1], 'angle_vs_delta'+ylabel)
        

def evaluate_onedim_perturb(path, dataset='val', epoch=None):
    filename = 'onedim_perturb'+dataset
    if epoch is not None:
        filename += 'ep'+str(epoch)
    evaluate_acrossmodels(path, select_dict={'ORN_NOISE_STD': 0},
                          values=np.linspace(-1, 1, 3),
                          n_rep=50, dataset=dataset, epoch=epoch,
                          file=filename)

def plot_onedim_perturb(path, dataset='val', epoch=None, minzero=False):
    filename = 'onedim_perturb'+dataset
    if epoch is not None:
        filename += 'ep'+str(epoch)
    file = os.path.join(path, filename+'.pkl')
    with open(file, 'rb') as f:
        results = pickle.load(f)
    
    import matplotlib as mpl
    colors = mpl.cm.viridis(np.linspace(0, 1, len(results['models'])))

    plt.figure()    
    for i, name in enumerate(results['models']):
        y_plot = np.median(results['loss_dict'][name], axis=0)
        if minzero:
            y_plot -= y_plot.min()
        plt.plot(results['values'], y_plot,
                 label=str(name), color=colors[i])
    plt.legend(title='K')
    plt.xlabel('alpha')
    if minzero:
        plt.ylabel('loss (min zero)')
    else:
        plt.ylabel('loss')
    title_txt = nicename(dataset)
    figname = 'onedim_perturb'+dataset
    if epoch is not None:
        title_txt += ' epoch ' + str(epoch)
        figname += 'ep'+str(epoch)
    plt.title(title_txt)
    save_fig(path.split('/')[-1], figname)
    
    
    for ylabel in ['val_acc', 'val_loss']:
        res_dict = results['acc_dict'] if ylabel == 'val_acc' else results['loss_dict']
        end_points = list()
        for i, name in enumerate(results['models']):
            y_plot = np.median(res_dict[name], axis=0)
            if minzero:
                if ylabel == 'val_loss':
                    y_plot -= y_plot.min()
                if ylabel == 'val_acc':
                    y_plot -= y_plot.max()
            end_points.append(y_plot[0])
                    
        fig = plt.figure(figsize=(1.5, 1.5))
        ax = fig.add_axes((0.3, 0.3, 0.6, 0.55))
        ax.plot(results['models'], end_points, 'o-', markersize=3, color=tools.blue)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks([3, 7, 15, 30])
        ax.plot([7, 7], [ax.get_ylim()[0], ax.get_ylim()[-1]], '--', color='gray')        
        plt.locator_params(axis='y', nbins=2)                
        plt.xlabel(nicename('kc_inputs'))
        plt.ylabel('Change in '+nicename(ylabel)+'\n(sharpness)')        
        if dataset == 'train':
            title_txt = nicename(dataset) + ' '
        else:
            title_txt = ''
        figname = 'onedim_'+ylabel+'change'+dataset
        if epoch is not None:
            title_txt += 'Epoch ' + str(epoch)
            figname += 'ep'+str(epoch)
        plt.title(title_txt, fontsize=7)        
        save_fig(path.split('/')[-1], figname)


def evaluate_twodim_perturb(path, dataset='val', epoch=None, K=None):
    filename = 'twodim_perturb'+dataset
    if epoch is not None:
        filename += 'ep'+str(epoch)
    if K is not None:
        filename += 'K'+str(K)
    X, Y = np.meshgrid(np.linspace(-1, 1, 15), np.linspace(-1, 1, 15))
    values = list(zip(X.flatten(), Y.flatten()))
    select_dict = {'ORN_NOISE_STD': 0}
    if K is not None:
        select_dict['kc_inputs'] = K
    evaluate_acrossmodels(path, select_dict=select_dict,
                          values=values,
                          n_rep=1, dataset=dataset, epoch=epoch,
                          file=filename, multidirection=2)
    

def plot_twodim_perturb(path, dataset='val', epoch=None, K=None):
    filename = 'twodim_perturb'+dataset
    if epoch is not None:
        filename += 'ep'+str(epoch)
    if K is not None:
        filename += 'K'+str(K)
    file = os.path.join(path, filename+'.pkl')
    with open(file, 'rb') as f:
        results = pickle.load(f)
    
    import matplotlib as mpl
    colors = mpl.cm.viridis(np.linspace(0, 1, len(results['models'])))

    plt.figure()
    X, Y = zip(*results['values'])
    name = results['models'][0]
    Z = results['loss_dict'][name]
    
    nx = int(np.sqrt(len(X)))
    X = np.reshape(X, (nx, nx))
    Y = np.reshape(Y, (nx, nx))
    Z = np.reshape(Z, (nx, nx))
    
    
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=mpl.cm.coolwarm,
                           linewidth=0, antialiased=False)
    

if __name__ == '__main__':
    oracle_dir = 'kcrole'
    # Load dataset
    # TODO: Make sure this works for dataset is different
    # data_dir = os.path.join(rootpath, 'datasets', 'proto', 'small')
    data_dir = os.path.join(rootpath, 'datasets', 'proto', 'standard')
    TRAIN_X, TRAIN_Y, VAL_X, VAL_Y = task.load_data(data_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    path = '../files/vary_kc_claws_new'
    # evaluate_acrossmodels(path, select_dict={'ORN_NOISE_STD': 0})
    # path = '../files/vary_kc_claws_epoch15'
    # path = '../files/vary_kc_claws_epoch2'
    # path = '../files/vary_kc_claws_epoch2_1000class'
    path = '../files/vary_kc_claws_fixedacc'
# =============================================================================
#     evaluate_acrossmodels(
#             path, select_dict={'ORN_NOISE_STD': 0},
#             values=[0, 0.5], n_rep=1, dataset='val')
#     plot_acrossmodels(path, dataset='val')
# =============================================================================


    
