import os
import torch
import temp_meta.metamodel
import configs
import torch.nn as nn
import tools
from mamldataset import DataGenerator
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch
from collections import OrderedDict
from torchmeta.modules import MetaModule


def gradient_update_parameters(model,
                               loss,
                               update_lr,
                               first_order=False):
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))

    params = OrderedDict(model.meta_named_parameters())
    final_layer_params = OrderedDict(model.layer3.meta_named_parameters())

    new_params = dict()
    grads = torch.autograd.grad(loss,
                                final_layer_params.values(),
                                create_graph=not first_order)

    for name, param in params.items():
        new_params[name] = param
    for (name, param), grad in zip(final_layer_params.items(), grads):
        new_params[name] = param - update_lr * grad
    return new_params


def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points

    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.

    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(num_examples, num_classes)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    _, targets = torch.max(targets, dim=-1)
    return torch.mean(predictions.eq(targets).float())


def run_per_batch(model, loss, split_size, data_x, data_t, update_lr):
    pre_acc_av = torch.tensor(0., device=device)
    pre_loss_av = torch.tensor(0., device=device)
    post_acc_av = torch.tensor(0., device=device)
    post_loss_av = torch.tensor(0., device=device)
    val_acc_av = torch.tensor(0., device=device)
    val_loss_av = torch.tensor(0., device=device)
    l = data_x.shape[0]
    for task_ix, (x, t) in \
            enumerate(zip(data_x, data_t)):
        train_x, val_x = torch.split(x, split_size, dim=0)
        train_t, val_t = torch.split(t, split_size, dim=0)
        
        # Train
        pre_y = model(train_x)
        pre_loss = loss(pre_y, torch.max(train_t, dim=-1)[1])
    
        with torch.no_grad():
            pre_acc = get_accuracy(pre_y, train_t)
    
        model.zero_grad()
        params = gradient_update_parameters(
            model,
            pre_loss,
            update_lr=update_lr,
            first_order=False)

        with torch.no_grad():
            post_y = model(train_x, params=params)
            post_loss = loss(post_y, torch.max(train_t, dim=-1)[1])
            post_acc = get_accuracy(post_y, train_t)
    
        # Test
        val_y = model(val_x, params=params)
        val_loss = loss(val_y, torch.max(val_t, dim=-1)[1])
    
        with torch.no_grad():
            val_acc = get_accuracy(val_y, val_t)
            
        pre_acc_av += pre_acc
        pre_loss_av += pre_loss
        post_acc_av += post_acc
        post_loss_av += post_loss
        val_acc_av += val_acc
        val_loss_av += val_loss

    return pre_acc_av.div_(l), pre_loss_av.div_(l), \
        post_acc_av.div_(l), post_loss_av.div_(l), \
        val_acc_av.div_(l), val_loss_av.div_(l)


def train(config: configs.MetaConfig):
    dataset_config = tools.load_config(config.data_dir)
    dataset_config.update(config)
    config = dataset_config
    for item in config.__dict__.items():
        print(item)

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    tools.save_config(config, save_path=config.save_path)

    # model
    model = temp_meta.metamodel.model(config=config)
    model.to(device=device)
    meta_optimizer = torch.optim.Adam(model.parameters(),
                                      lr=config.meta_lr)

    print('MODEL VARIABLES')
    for k, v in model.named_parameters():
        print(k, v.shape)

    num_samples_per_class = config.meta_num_samples_per_class
    dim_output = config.N_CLASS  # TODO: this doesn't have to be
    num_class = config.meta_labels_per_class * dim_output
    data_generator = DataGenerator(
        dataset=config.data_dir,
        batch_size=num_samples_per_class * num_class * 2,
        meta_batch_size=config.meta_batch_size,
        num_samples_per_class=num_samples_per_class,
        num_class=num_class,
        dim_output=dim_output,
    )

    PRINT_INTERVAL = config.meta_print_interval
    loss = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for itr in range(config.metatrain_iterations):
        model.zero_grad()
        train_x_np, train_y_np = data_generator.generate('train')
        train_x_torch = torch.from_numpy(train_x_np).float().to(device)
        train_t_torch = torch.from_numpy(train_y_np).long().to(device)
        
        metatrain_train_pre_acc, metatrain_train_pre_loss, \
        metatrain_train_post_acc, metatrain_train_post_loss, \
        metatrain_val_acc, metatrain_val_loss = \
            run_per_batch(model, 
                          loss, 
                          num_class * num_samples_per_class, 
                          train_x_torch,
                          train_t_torch,
                          update_lr=config.meta_update_lr)

        metatrain_val_loss.backward()
        meta_optimizer.step()

        if itr % PRINT_INTERVAL == 0:
            print('Iteration ' + str(itr))

            if itr > 0:
                if config.save_every_epoch:
                    print('SAVING')
                    model.save_pickle(itr)
                    model.save()

                print('Meta-train')
                total_time = time.time() - start_time
                print('Time taken {:0.1f}s'.format(total_time))
                print('train_pre loss: {}, acc: {}'.format(
                    metatrain_train_pre_loss, metatrain_train_pre_acc))
                print('train_post loss: {}, acc: {}'.format(
                    metatrain_train_post_loss, metatrain_train_post_acc))
                print('val_post loss: {}, acc: {}'.format(
                    metatrain_val_loss, metatrain_val_acc))

            # TODO: uncomment this after done debugging
            # test_x_np, test_y_np = data_generator.generate('val')
            # test_x_torch = torch.from_numpy(test_x_np).float().to(device)
            # test_t_torch = torch.from_numpy(test_y_np).long().to(device)
            # metaval_train_pre_acc, metaval_train_pre_loss, \
            # metaval_train_post_acc, metaval_train_post_loss, \
            # metaval_val_acc, metaval_val_loss = \
            #     run_per_batch(model,
            #                   loss,
            #                   num_class * num_samples_per_class,
            #                   test_x_torch,
            #                   test_t_torch,
            #                   update_lr=config.meta_update_lr)
            #
            # print('Meta-val')
            # total_time = time.time() - start_time
            # print('Time taken {:0.1f}s'.format(total_time))
            # print('train_pre loss: {}, acc: {}'.format(
            #     metaval_train_pre_loss, metaval_train_pre_acc))
            # print('train_post loss: {}, acc: {}'.format(
            #     metaval_train_post_loss, metaval_train_post_acc))
            # print('val_post loss: {}, acc: {}'.format(
            #     metaval_val_loss, metaval_val_acc))


def main():
    import shutil
    import configs
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = configs.MetaConfig()
    config.meta_lr = .001
    config.N_CLASS = 10 #10
    config.save_every_epoch = True
    config.meta_batch_size = 32 #32
    config.meta_num_samples_per_class = 8 #16
    config.meta_print_interval = 100

    config.replicate_orn_with_tiling = True
    config.N_ORN_DUPLICATION = 10
    config.output_max_lr = 2.0 #2.0
    config.meta_update_lr = .01

    config.metatrain_iterations = 15000
    config.pn_norm_pre = 'batch_norm'
    config.kc_norm_pre = 'batch_norm'
    config.pn_norm_pre = None
    config.kc_norm_pre = None
    # config.initial_pn2kc = 0.05 #0.05

    config.data_dir = '../datasets/proto/meta_dataset'
    config.save_path = '../files/torch_metalearn'

    try:
        shutil.rmtree(config.save_path)
    except FileNotFoundError:
        pass
    train(config)

if __name__ == "__main__":
    main()