import tools
import matplotlib.pyplot as plt
import task
import numpy as np

def _distance_preserve(x_arg, y_arg):
    save_path = r'C:\Users\Peter\PycharmProjects\olfaction_evolution\files\pn_normalization\000010'
    data_dir = r'C:\Users\Peter\PycharmProjects\olfaction_evolution\datasets\proto\standard'
    config = tools.load_config(save_path)
    train_x, train_y, val_x, val_y = task.load_data(data_dir)

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