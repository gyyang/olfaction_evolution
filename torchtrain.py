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

class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


def train_from_path(path):
    """Train from a path with a config file in it."""
    config = tools.load_config(path)
    train(config, reload=True)


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
    if 'n_batch' in dir(config):
        n_batch = config.n_batch
    else:
        n_batch = train_x.shape[0] // batch_size

    model = FullModel(config=config)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0)

    # Build training dataset
    dataset = MyDataset(train_x, train_y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    # Build validation dataset
    val_data = torch.from_numpy(val_x).float().to(device)
    val_target = torch.from_numpy(val_y).long().to(device)

    # Make custom logger
    log = defaultdict(list)
    log_name = os.path.join(config.save_path, 'log.pkl')  # Consider json instead of pickle

    glo_score_mode = 'tile' if config.replicate_orn_with_tiling else 'repeat'

    finish_training = False


    start_epoch = 0

    loss_train = 0
    acc = 0
    lr = config.lr
    acc_smooth = 0
    total_time, start_time = 0, time.time()
    w_bins = np.linspace(0, 1, 201)
    w_bins_log = np.linspace(-20, 5, 201)
    log['w_bins'] = w_bins
    log['w_bins_log'] = w_bins_log
    lin_bins = np.linspace(0, 1, 1001)
    log['lin_bins'] = lin_bins
    activity_bins = np.linspace(0, 1, 201)

    for ep in range(start_epoch, config.max_epoch):
        # validation
        with torch.no_grad():
            model.eval()
            loss_val, acc_val = model(val_data, val_target)

        if ep % config.save_epoch_interval == 0:
            print('[*' + '*'*50 + '*]')
            print('Epoch {:d}'.format(ep))
            print('Train/Validation loss {:0.2f}/{:0.2f}'.format(
                loss_train, loss_val.item()))
            print('Train/Validation accuracy {:0.2f}/{:0.2f}'.format(
                acc, acc_val))

            if ep > 0:
                time_spent = time.time() - start_time
                total_time += time_spent
                print('Time taken {:0.1f}s'.format(total_time))
                print('Examples/second {:d}'.format(int(train_x.shape[0]/time_spent)))
            start_time = time.time()

        try:
            model.train()
            for batch_idx, (data, target) in enumerate(loader):
                loss, acc = model(data.to(device), target.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_train = loss.item()

        except KeyboardInterrupt:
            print('Training interrupted by users')
            finish_training = True

        if finish_training:
            break

    print('Training finished')



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


