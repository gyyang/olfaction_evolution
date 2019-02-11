import configs
import task

seed = 0

def make_standard_dataset():
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

def make_relabel_datasets_large():
    config = configs.input_ProtoConfig()
    config.N_CLASS = 100
    for i in [100, 200, 500, 1000, 2000, 5000]:
        config.n_trueclass = i
        config.relabel = True
        task.save_proto(config, seed=seed, folder_name= str(config.n_trueclass) + '_' + str(config.N_CLASS))
        print('Done Relabel Dataset: ' + str(i))

def make_relabel_datasets_small():
    config = configs.input_ProtoConfig()
    config.N_CLASS = 20
    for i in [20, 40, 80, 200, 400, 800, 2000]:
        config.n_trueclass = i
        config.relabel = True
        task.save_proto(config, seed=seed, folder_name= str(config.n_trueclass) + '_' + str(config.N_CLASS))
        print('Done Relabel Dataset: ' + str(i))

def make_concentration_dataset():
    config = configs.input_ProtoConfig()
    config.N_CLASS = 100
    config.vary_concentration = True
    task.save_proto(config, seed=seed, folder_name='concentration')
    print('Done Concentration Dataset')

def make_mask_dataset():
    config = configs.input_ProtoConfig()
    config.N_CLASS = 100
    config.realistic_orn_mask = True
    task.save_proto(config, seed=seed, folder_name='mask')
    print('Done Mask Dataset')

def make_concentration_with_masking_dataset():
    config = configs.input_ProtoConfig()
    config.N_CLASS = 100
    config.vary_concentration = True
    config.realistic_orn_mask = True
    task.save_proto(config, seed=seed, folder_name='concentration_mask')
    print('Done Concentration_Mask Dataset')

def make_combinatorial_dataset():
    config = configs.input_ProtoConfig()
    config.N_CLASS = 20
    config.n_combinatorial_classes = 20
    config.combinatorial_density = .2
    config.label_type = 'combinatorial'
    task.save_proto(config, seed=seed, folder_name='combinatorial')

def temp():
    config = configs.input_ProtoConfig()
    task.save_proto(config, seed=seed, folder_name='test')
    print('Done test dataset')

if __name__ == '__main__':
    make_standard_dataset()
    # make_relabel_datasets_small()
    # make_relabel_datasets_large()
    # make_concentration_dataset()
    # make_concentration_with_masking_dataset()
    # make_primordial_dataset()
    # make_mask_dataset()
    # make_combinatorial_dataset()

