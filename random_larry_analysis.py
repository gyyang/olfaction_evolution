import tools
import matplotlib.pyplot as plt
import task
import standard.analysis as sa
import numpy as np
import os

def _plot_gloscores(path, ix, cutoff, shuffle=False):
    def _helper(w_plot, str):
        ind_max = np.argmax(w_plot, axis=0)
        ind_sort = np.argsort(ind_max)
        w_plot = w_plot[:, ind_sort]
        plt.imshow(w_plot, cmap='RdBu_r', vmin=-1, vmax=1,
                           interpolation='none')
        tools.save_fig(path, 'cutoff_' + str)

    w_orns = tools.load_pickle(path, 'w_orn')
    w_orn = w_orns[ix]
    print(w_orn.shape)
    w_orn = tools._reshape_worn(w_orn, 50)
    w_orn = w_orn.mean(axis=0)
    if shuffle:
        np.random.shuffle(w_orn.flat)

    avg_gs, all_gs = tools.compute_glo_score(w_orn, 50, mode='tile', w_or=None)
    all_gs = np.array(all_gs)
    gs_ix = np.argsort(all_gs)[::-1]

    w_kcs = tools.load_pickle(path, 'w_glo')
    w_kc = w_kcs[ix]
    sum_kc_weights = np.sum(w_kc, axis=1)
    ix_good = all_gs > .9
    ix_bad = all_gs < .9
    weight_to_good = np.mean(sum_kc_weights[ix_good])
    weight_to_bad = np.mean(sum_kc_weights[ix_bad])
    print(weight_to_bad)
    print(weight_to_good)

    arg = 'shuffled' if shuffle else ''
    w_plot = w_orn[:, gs_ix[:cutoff]]
    _helper(w_plot, 'top_' + arg)
    w_plot = w_orn[:, gs_ix[cutoff:]]
    _helper(w_plot, 'bottom_' + arg)

    if shuffle:
        all_gs = []
        for i in range(10):
            np.random.shuffle(w_orn.flat)
            avg_gs, all_gs_ = tools.compute_glo_score(w_orn, 50, mode='tile', w_or=None)
            all_gs.append(all_gs_)
        all_gs = np.concatenate(all_gs, axis=0)

    arg = 'hist_'
    arg = arg + 'shuffled' if shuffle else arg
    plt.figure()
    weights = np.ones_like(all_gs) / float(len(all_gs))
    plt.hist(all_gs, bins=20, range=[0, 1], weights= weights)
    tools.save_fig(path, arg)
    return ix_good, ix_bad

def _lesion_multiglomerular_pn(path, ix, units):
    import tensorflow as tf
    from model import FullModel

    config = tools.load_config(path)
    tf.reset_default_graph()

    # Load dataset
    train_x, train_y, val_x, val_y = task.load_data(
        config.dataset, config.data_dir)

    # Build validation model
    val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)
    val_model = FullModel(val_x_ph, val_y_ph, config=config, training=False)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        val_model.load()

        if units is not None:
            val_model.lesion_units('model/layer2/kernel:0', units)

        # Validation
        val_loss, val_acc = sess.run(
            [val_model.loss, val_model.acc],
            {val_x_ph: val_x, val_y_ph: val_y})
        print(val_acc)
        return val_acc

path = r'C:\Users\Peter\PycharmProjects\olfaction_evolution\files_temp\temp'
ix = 0
ix_good, ix_bad = _plot_gloscores(path, ix, 100, shuffle=False)
# _plot_gloscores(path, ix, 100, shuffle=True)

print(np.sum(ix_good), np.sum(ix_bad))
_lesion_multiglomerular_pn(os.path.join(path,'000000'), ix, None)
_lesion_multiglomerular_pn(os.path.join(path,'000000'), ix, ix_good)
_lesion_multiglomerular_pn(os.path.join(path,'000000'), ix, ix_bad)

def _orthogonality(path, ix, arg):
    w_orns = tools.load_pickle(path, 'w_orn')
    w_orn = w_orns[ix]
    print(w_orn.shape)

    out = np.zeros((w_orn.shape[1], w_orn.shape[1]))
    for i in range(w_orn.shape[1]):
        for j in range(w_orn.shape[1]):
            x = np.dot(w_orn[:,i], w_orn[:,j])
            y = np.corrcoef(w_orn[:,i], w_orn[:,j])[0,1]
            if arg == 'ortho':
                out[i,j] = x
            else:
                out[i,j] = y
    plt.imshow(out, cmap='RdBu_r', interpolation='none')
    plt.colorbar()

    txt = '_orthogonality_' if arg == 'ortho' else '_correlation_'
    tools.save_fig(path, txt + str(ix))

# path = r'C:\Users\Peter\PycharmProjects\olfaction_evolution\files\standard_net'
# _orthogonality(path, 0, arg='ortho')

def _distance_preserve(x_arg, y_arg):
    save_path = r'C:\Users\Peter\PycharmProjects\olfaction_evolution\files\pn_normalization\000010'
    data_dir = r'C:\Users\Peter\PycharmProjects\olfaction_evolution\datasets\proto\standard'
    config = tools.load_config(save_path)
    train_x, train_y, val_x, val_y = task.load_data(config.dataset, data_dir)

    import tensorflow as tf
    from model import FullModel
    tf.reset_default_graph()
    CurrentModel = FullModel
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
        glo_out, glo_in, kc_in, kc_out, logits = sess.run(
            [model.glo, model.glo_in, model.kc_in, model.kc, model.logits],
            {val_x_ph: val_x, val_y_ph: val_y})

    n = 10000
    n_odors = val_x.shape[0]
    xs = []
    ys = []

    if x_arg == 'kc_in':
        x = kc_in
        xlim = [-1, 30]
    elif x_arg == 'glo_in':
        x = glo_in
        xlim = [0, 4]
    elif x_arg == 'input':
        x = val_x
        xlim = [0, 4]

    if y_arg == 'kc_out':
        y = kc_out
        ylim = [-1, 30]
    elif y_arg == 'kc_in':
        y = kc_in
        ylim = [0, 30]
    elif y_arg == 'glo_in':
        y = glo_in
        ylim = [0, 4]
    elif y_arg == 'glo_out':
        y = glo_out
        ylim = [0, 4]

    for i in range(n):
        pair = np.random.choice(n_odors, 2)
        x1, x2 = x[pair[0]], x[pair[1]]
        y1, y2 = y[pair[0]], y[pair[1]]
        x_diff = np.linalg.norm(x2-x1)
        y_diff = np.linalg.norm(y2-y1)
        xs.append(x_diff)
        ys.append(y_diff)
    plt.scatter(xs,ys, s= .5, alpha = .5)
    plt.xlim(xlim)
    plt.ylim(ylim)
    tools.save_fig(save_path, '_' +x_arg + '_' + y_arg)

# _distance_preserve('input','kc_out')