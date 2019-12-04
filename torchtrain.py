import os
import sys
from collections import defaultdict
import pickle
import json
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import task
from torchmodel import FullModel
from configs import FullConfig, SingleLayerConfig
import tools
from standard.analysis_pn2kc_training import _compute_sparsity

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def _log_full_model_train_pn2kc(log, model, res, config):
    w_glo = model.w_glo
    w_glo[w_glo < 1e-9] = 1e-9  # finite range for log
    kcs = res['kc'].cpu().numpy()  # (n_odor, n_neuron)

    coding_level = (kcs > 0).mean()
    coding_level_per_kc = kcs.mean(axis=0)
    coding_level_per_odor = kcs.mean(axis=1)
    log['coding_level'].append(coding_level)
    hist, _ = np.histogram(coding_level_per_kc, bins=log['activity_bins'])
    log['coding_level_per_kc'].append(hist)
    hist, _ = np.histogram(coding_level_per_odor, bins=log['activity_bins'])
    log['coding_level_per_odor'].append(hist)

    # Store distribution of flattened weigths
    log_hist, _ = np.histogram(np.log(w_glo.flatten()),
                               bins=log['log_bins'])
    hist, _ = np.histogram(w_glo.flatten(), bins=log['lin_bins'])
    log['log_hist'].append(log_hist)
    log['lin_hist'].append(hist)
    log['kc_w_sum'].append(w_glo.sum(axis=0))
    # Store sparsity computed with threshold

    sparsity_inferred, thres_inferred = _compute_sparsity(
        w_glo, dynamic_thres=True)
    K_inferred = sparsity_inferred[sparsity_inferred > 0].mean()
    bad_KC_inferred = np.sum(
        sparsity_inferred == 0) / sparsity_inferred.size
    log['sparsity_inferred'].append(sparsity_inferred)
    log['thres_inferred'].append(thres_inferred)
    log['K_inferred'].append(K_inferred)
    log['bad_KC_inferred'].append(bad_KC_inferred)

    if config.kc_prune_weak_weights:
        sparsity, thres = _compute_sparsity(w_glo, dynamic_thres=False,
                                            thres=config.kc_prune_threshold)
    else:
        sparsity, thres = _compute_sparsity(w_glo, dynamic_thres=False)
    K = sparsity[sparsity > 0].mean()
    bad_KC = np.sum(sparsity == 0) / sparsity.size
    log['sparsity'].append(sparsity)
    log['thres'].append(thres)
    log['K'].append(K)
    log['bad_KC'].append(bad_KC)

    print('KC coding level={}'.format(np.round(coding_level, 2)))
    print('Bad KCs (fixed, inferred) ={}, {}'.format(bad_KC,
                                                     bad_KC_inferred))
    print('K (fixed, inferred) ={}, {}'.format(K, K_inferred))
    return log


def logging(log, model, res, config):
    if config.model == 'full':
        w_orn = model.w_orn
        glo_score, _ = tools.compute_glo_score(w_orn, config.N_ORN)
        log['glo_score'].append(glo_score)
        print('Glo score ' + str(glo_score))

        sim_score, _ = tools.compute_sim_score(w_orn, config.N_ORN)
        log['sim_score'].append(sim_score)
        print('Sim score ' + str(sim_score))

        if config.train_pn2kc:
            log = _log_full_model_train_pn2kc(log, model, res, config)

    # consider json instead of pickle
    with open(os.path.join(config.save_path, 'log.pkl'), 'wb') as f:
        pickle.dump(log, f, protocol=pickle.HIGHEST_PROTOCOL)
    return log


def train(config, reload=False, save_everytrainloss=False):
    # Merge model config with config from dataset
    dataset_config = tools.load_config(config.data_dir)
    dataset_config.update(config)
    config = dataset_config
    for item in config.__dict__.items():
        print(item)

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    # Save config
    tools.save_config(config, save_path=config.save_path)

    # Load dataset
    train_x, train_y, val_x, val_y = task.load_data(config.dataset, config.data_dir)

    batch_size = config.batch_size

    model = FullModel(config=config)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_data = torch.from_numpy(train_x).float().to(device)
    train_target = torch.from_numpy(train_y).long().to(device)

    # Build validation dataset
    val_data = torch.from_numpy(val_x).float().to(device)
    val_target = torch.from_numpy(val_y).long().to(device)

    # Make custom logger
    log = defaultdict(list)

    finish_training = False

    start_epoch = 0

    loss_train = 0
    res = {'acc': np.nan}
    total_time, start_time = 0, time.time()

    log['lin_bins'] = np.linspace(0, 1, 1001)
    log['log_bins'] = np.linspace(-20, 5, 201)
    log['activity_bins'] = np.linspace(0, 1, 201)

    for ep in range(start_epoch, config.max_epoch):
        if config.save_every_epoch:
            model.save_pickle(ep)
            model.save(ep)

        # validation
        with torch.no_grad():
            model.eval()
            res_val = model(val_data, val_target)
        loss_val = res_val['loss'].item()

        print('[*' + '*'*50 + '*]')
        print('Epoch {:d}'.format(ep))
        print('Train/Validation loss {:0.2f}/{:0.2f}'.format(
            loss_train, loss_val))
        print('Train/Validation accuracy {:0.2f}/{:0.2f}'.format(
            res['acc'], res_val['acc']))
        log['epoch'].append(ep)
        log['train_loss'].append(loss_train)
        log['val_loss'].append(loss_val)
        log['train_acc'].append(res['acc'])
        log['val_acc'].append(res_val['acc'])

        log = logging(log, model, res_val, config)

        if ep > 0:
            time_spent = time.time() - start_time
            total_time += time_spent
            print('Time taken {:0.1f}s'.format(total_time))
            print('Examples/second {:d}'.format(int(train_x.shape[0]/time_spent)))
        start_time = time.time()

        try:
            model.train()
            random_idx = np.random.permutation(config.n_train)
            idx = 0
            while idx < config.n_train:
                batch_indices = random_idx[idx:idx+batch_size]
                idx += batch_size

                res = model(train_data[batch_indices],
                            train_target[batch_indices])
                optimizer.zero_grad()
                res['loss'].backward()
                optimizer.step()

            loss_train = res['loss'].item()

        except KeyboardInterrupt:
            print('Training interrupted by users')
            finish_training = True

        if finish_training:
            break
        sys.stdout.flush()

    print('Training finished')

    if 'save_log_only' in dir(config) and config.save_log_only is True:
        pass
    else:
        model.save_pickle()
        model.save()


def train_from_path(path):
    """Train from a path with a config file in it."""
    config = tools.load_config(path)
    train(config, reload=True)


if __name__ == '__main__':
    config = FullConfig()
    config.dataset = 'proto'
    config.data_dir = './datasets/proto/standard'
    config.model = 'full'
    config.save_path = './files/tmp_train'
    config.train_pn2kc = True
    config.sparse_pn2kc = False
    config.pn_norm_pre = 'batch_norm'

    train(config)


