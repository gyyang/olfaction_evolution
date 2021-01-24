"""Functions to generate dataset.

Convention is to name a function
def make_datasetname_dataset()
"""

import configs
import task

seed = 0


def make_standard_dataset():
    """Standard dataset."""
    task_config = task.input_ProtoConfig()
    task.save_proto(config=task_config, seed=0, folder_name='standard')


def make_primordial_dataset():
    config = configs.input_ProtoConfig()
    config.label_type = 'multi_head_sparse'
    config.n_proto_valence = 25
    config.has_special_odors = True
    config.n_class_valence = 3
    config.n_trueclass = 1000
    task.save_proto(config, seed=seed, folder_name='primordial')


def make_relabel_dataset(mode='small'):
    """Generate dataset with n_trueclass, then relabel to fewer classes."""
    config = configs.input_ProtoConfig()

    if mode == 'small':
        relabel_class = 20
        true_classes = [20, 40, 80, 200, 400, 800, 2000]
    elif mode == 'large':
        relabel_class = 100
        true_classes = [100, 200, 500, 1000, 2000, 5000]
    else:
        raise ValueError('Unknown mode', mode)

    config.N_CLASS = relabel_class
    for i in true_classes:
        config.n_trueclass = i
        config.relabel = True
        fn = 'relabel_' + str(config.n_trueclass) + '_' + str(config.N_CLASS)
        task.save_proto(config, seed=seed, folder_name=fn)
        print('Done Relabel Dataset: ' + str(i))


def make_relabel_vary_or_dataset():
    """Vary the number of olfactory receptors for relabel dataset."""
    task_config = task.input_ProtoConfig()
    for n_or in [25, 35, 50, 75, 100, 150, 200]:
        task_config.N_ORN = n_or
        task_config.relabel = True
        task_config.N_CLASS = 100
        task_config.n_trueclass = 200
        task.save_proto(config=task_config, seed=0,
                        folder_name='relabel_orn'+str(n_or))
        print('Done Relabel Vary OR Dataset: ' + str(n_or))


def make_relabel_corr_vary_or_dataset():
    """Vary the number of olfactory receptors for relabel dataset."""
    task_config = task.input_ProtoConfig()
    for n_or in [25, 35, 50, 75, 100, 150, 200]:
        task_config.N_ORN = n_or
        task_config.relabel = True
        task_config.N_CLASS = 100
        task_config.n_trueclass = 200
        task_config.orn_corr = 0.1
        task.save_proto(config=task_config, seed=0,
                        folder_name='relabel_corr_orn'+str(n_or))
        print('Done Relabel Corr Vary OR Dataset: ' + str(n_or))


def make_concentration_dataset():
    """Impose odor concentration invariance."""
    config = configs.input_ProtoConfig()
    config.N_CLASS = 100
    config.vary_concentration = True
    task.save_proto(config, seed=seed, folder_name='concentration')
    print('Done Concentration Dataset')


def make_mask_row_dataset():
    """Impose sparsity on ORN activation."""
    config = configs.input_ProtoConfig()
    config.N_CLASS = 100
    for i in [.2, .4, .6, .8, 1]:
        config.spread_orn_activity = (True, i)
        task.save_proto(config, seed=seed, folder_name='mask_row_' + str(i))
    print('Done Mask Dataset')


def make_concentration_mask_row_dataset():
    """Impose sparsity on ORN activation and concentration invariance."""
    config = configs.input_ProtoConfig()
    config.N_CLASS = 100
    config.vary_concentration = True
    config.is_spread_orn_activity = True
    for spread in [0, .3, .6, .9]:
        config.spread_orn_activity = spread
        fn = 'concentration_mask_row_{:0.1f}'.format(spread)
        task.save_proto(config, seed=seed, folder_name=fn)
    print('Done Concentration_Mask Dataset')


def make_concentration_relabel_mask_row_dataset():
    """Impose sparsity on ORN activation and concentration invariance."""
    config = configs.input_ProtoConfig()
    # relabel
    config.relabel = True
    config.N_CLASS = 100
    config.n_trueclass = 200

    config.vary_concentration = True
    config.is_spread_orn_activity = True

    for spread in [0, .3, .6, .9]:
        config.spread_orn_activity = spread
        fn = 'concentration_relabel_mask_row_{:0.1f}'.format(spread)
        task.save_proto(config, seed=seed, folder_name=fn)
    print('Done Concentration_Mask Dataset')
make_concentration_mask_row_dataset()
make_concentration_relabel_mask_row_dataset()

def make_combinatorial_dataset():
    """Map an odor to a combinatorial code, instead of a single class."""
    config = configs.input_ProtoConfig()
    config.N_CLASS = 20
    config.n_combinatorial_classes = 20
    config.combinatorial_density = .2
    config.label_type = 'combinatorial'
    fn = 'combinatorial_' + str(config.N_CLASS) + '_' + str(config.combinatorial_density)
    task.save_proto(config, seed=seed, folder_name=fn)


def make_small_training_set_dataset():
    """Dataset with fewer training samples."""
    config = configs.input_ProtoConfig()
    config.N_CLASS = 100
    for i in [100, 1000, 10000, 100000, 1000000]:
        config.n_train = i
        fn = 'small_training_set_' + str(i)
        task.save_proto(config=config, seed=0, folder_name=fn)
        print('Done small training dataset: ' + str(i))


def make_multihead_dataset():
    """Simultaneous classification of odor class and valence."""
    task_config = task.input_ProtoConfig()
    task_config.label_type = 'multi_head_sparse'
    task_config.has_special_odors = True
    task_config.n_proto_valence = 5
    task.save_proto(config=task_config, seed=0, folder_name='multihead')


def make_multihead_relabel_dataset():
    """Simultaneous classification of odor class and valence."""
    task_config = task.input_ProtoConfig()

    # Multihead
    task_config.label_type = 'multi_head_sparse'
    task_config.has_special_odors = True
    task_config.n_proto_valence = 5

    # Relabel
    task_config.relabel = True
    task_config.N_CLASS = 100
    task_config.n_trueclass = 200

    task.save_proto(config=task_config, seed=0,
                    folder_name='multihead_relabel')


def make_vary_or_dataset():
    """Vary the number of olfactory receptors."""
    task_config = task.input_ProtoConfig()
    for n_or in [25, 35, 50, 75, 100, 150, 200]:
        task_config.N_ORN = n_or
        task.save_proto(config=task_config, seed=0, folder_name='orn'+str(n_or))


def make_orncorr_dataset():
    """Vary the correlation of olfactory receptor neuron activity."""
    task_config = task.input_ProtoConfig()
    for orn_corr in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        task_config.orn_corr = orn_corr
        fn = 'orn_corr_{:0.2f}'.format(orn_corr)
        task.save_proto(config=task_config, seed=0, folder_name=fn)
        print('Done orn_corr training dataset: ', orn_corr)


def make_multi_or_dataset():
    """Vary the number of receptors expressed in each receptor neuron."""
    task_config = task.input_ProtoConfig()
    # for n_or_per_orn in range(1, 10):
    for n_or_per_orn in [0, 50]:
        task_config.n_or_per_orn = n_or_per_orn
        fn = 'n_or_per_orn'+str(n_or_per_orn)
        task.save_proto(config=task_config, seed=0, folder_name=fn)


def make_dataset(dataset_name):
    func_name = 'make_' + dataset_name + '_dataset'
    globals()[func_name]()  # call function by the name func_name


