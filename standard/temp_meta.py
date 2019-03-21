import sys
import os
import pickle
import numpy as np
import tensorflow as tf

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)
import tools

def main():
    foldername = 'metatrain'
    # foldername = 'tmp_train'

    path = os.path.join(rootpath, 'files', foldername)
    figpath = os.path.join(rootpath, 'figures', foldername)

    # analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True)

    # TODO: clean up these paths
    path = os.path.join(path, '0')
    config = tools.load_config(path)
    config.data_dir = rootpath + config.data_dir[1:]
    config.save_path = rootpath + config.save_path[1:]

    def meta_lesion_analysis(units=None):
        def print_results(res):
            n_steps = len(res[1])
            print('Pre-update train loss >> {:0.4f}.  acc >> {:0.2f}, {:0.2f}'.format(res[0], res[3][0], res[3][1]))
            print('Post-update train loss >> {:0.4f}.  acc >> {:0.2f}, {:0.2f} '.format(res[2], res[5][0], res[5][1]))
            print('Post-update val loss step 1 >> {:0.4f}.  acc >> {:0.2f}, {:0.2f}'.format(res[1][0], res[4][0][0],
                                                                                            res[4][0][1]))
            print('Post-update val loss step {:d} >> {:0.4f}.  acc >> {:0.2f}, {:0.2f}'.format(n_steps, res[1][-1],
                                                                                               res[4][-1][0],
                                                                                               res[4][-1][1]))

        import mamldataset
        import mamlmodel
        num_samples_per_class = config.meta_num_samples_per_class
        num_class = config.N_CLASS
        dim_output = config.meta_output_dimension
        data_generator = mamldataset.DataGenerator(
            dataset= config.data_dir,
            batch_size=num_samples_per_class * num_class * 2,
            meta_batch_size=config.meta_batch_size,
            num_samples_per_class=num_samples_per_class,
            num_class=num_class,
            dim_output=dim_output)

        train_x_ph = tf.placeholder(tf.float32, (config.meta_batch_size,
                                                 data_generator.batch_size,
                                                 config.N_PN))

        train_y_ph = tf.placeholder(tf.float32, (config.meta_batch_size,
                                                 data_generator.batch_size,
                                                 dim_output + config.n_class_valence))

        val_model = mamlmodel.MAML(train_x_ph, train_y_ph, config, training=False)

        iterations = 10
        tf.reset_default_graph()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            val_model.load()

            if units is not None:
                val_model.lesion_units('model/layer3/kernel:0', units)
                val_model.lesion_units('model/layer3_2/kernel:0', units)

            for itr in range(iterations):
                lr = sess.run(val_model.update_lr)
                print('Learned update_lr is : ' +
                      np.array2string(lr, precision=2, separator=',', suppress_small=True))

                val_x, val_y = data_generator.generate('val')
                input_tensors = [
                    val_model.total_loss1, val_model.total_loss2, val_model.total_loss3,
                    val_model.total_acc1, val_model.total_acc2, val_model.total_acc3]
                res = sess.run(input_tensors,
                               {train_x_ph: val_x, train_y_ph: val_y})
                print('Meta-val')
                print_results(res)

    meta_lesion_analysis(None)

main()