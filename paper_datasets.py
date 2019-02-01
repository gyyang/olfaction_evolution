import configs
import task

seed = 0

def make_standard_dataset():
    task_config = task.input_ProtoConfig()
    task.save_proto(config=task_config, seed=0, folder_name='standard')

def make_relabel_datasets():
    config = configs.input_ProtoConfig()
    for i in [20, 40, 60, 80, 100, 120, 140, 160, 200, 500, 1000]:
        config.n_trueclass = i
        config.N_CLASS = 20
        config.relabel = True
        task.save_proto(config, seed=seed, folder_name= str(config.n_trueclass) + '_' + str(config.N_CLASS))
        print('Done Relabel Dataset: ' + str(i))

def make_concentration_dataset():
    config = configs.input_ProtoConfig()
    config.N_CLASS = 100
    config.vary_concentration = True
    task.save_proto(config, seed=seed, folder_name='concentration')
    print('Done Concentration Dataset')

