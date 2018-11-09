import task
import train
import configs

def transfer_learning():
    # If datasets don't exist, run these
    # task.save_proto(config=task.input_ProtoConfig(), seed=2)
    # task.save_proto(config=task.input_ProtoConfig(), seed=3)

    config = configs.FullConfig()
    config.max_epoch = 30
    config.target_acc = 0.53  # Make this more general, and lower
    config.train_orn2pn = True
    config.kc_norm_pre = 'layer_norm'
    config.data_dir = './datasets/proto/_100_generalization_onehot_s0'
    config.save_path = './files/transfer_layernorm'

    train.train(config, reload=False)

    config.train_orn2pn = False
    config.max_epoch = 15
    config.target_acc = None
    config.data_dir = './datasets/proto/_100_generalization_onehot_s1'

    train.train(config, reload=True)


if __name__ == '__main__':
    transfer_learning()