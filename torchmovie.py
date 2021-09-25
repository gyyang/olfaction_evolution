import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import collections as mc
import torch

import task
from torchmodel import FullModel
from configs import FullConfig
import tools

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    train_x, train_y, val_x, val_y = task.load_data(config.data_dir)

    batch_size = config.batch_size

    model = FullModel(config=config)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_data = torch.from_numpy(train_x).float().to(device)
    train_target = torch.from_numpy(train_y).long().to(device)

    n_save_every = 20
    ind_orn = list(range(0, 500, 50)) + list(range(1, 500, 50)) + list(range(2, 500, 50))
    weight_layer1, weight_layer2 = [], []

    k = 0
    for ep in range(config.max_epoch):
        if config.save_every_epoch:
            model.save_pickle(ep)
            model.save(ep)

        print('[*' + '*'*50 + '*]')
        print('Epoch {:d}'.format(ep))

        model.train()
        random_idx = np.random.permutation(config.n_train)
        idx = 0
        while idx < config.n_train:
            if (idx//batch_size) % n_save_every == 0:
                w_glo = model.w_glo
                w_orn = model.w_orn

                weight_layer1.append(w_orn[ind_orn, :])
                weight_layer2.append(w_glo[:, :30])
                k += 1

            batch_indices = random_idx[idx:idx+batch_size]
            idx += batch_size

            res = model(train_data[batch_indices],
                        train_target[batch_indices])
            optimizer.zero_grad()
            res['loss'].backward()
            optimizer.step()

    np.save(os.path.join(config.save_path, 'w_layer1'), np.array(weight_layer1))
    np.save(os.path.join(config.save_path, 'w_layer2'), np.array(weight_layer2))


def main_train():
    config = FullConfig()
    config.initial_pn2kc = 4. / config.N_PN  # explicitly set for clarity
    config.kc_prune_weak_weights = True
    config.kc_prune_threshold = 1. / config.N_PN
    config.kc_dropout_rate = 0.5
    config.save_path = './files/movie'
    config.max_epoch = 10
    train(config)

def main_plot(path):
    w1 = np.load(os.path.join(path, 'w_layer1.npy'))
    w2 = np.load(os.path.join(path, 'w_layer2.npy'))
    w1 = w1[:, :30, :]
    w2 = w2[:, :, :5]
    
    n_plot = 800
    # n_plot = 100
    w1 = w1[:n_plot]
    w2 = w2[:n_plot]
    
    w1 = w1[::2]
    w2 = w2[::2]

    # Normalize
    w1 /= np.max(w1)
    w2 /= np.max(w2)
    
    rect = [0.1, 0.1, 0.8, 0.8]
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_axes(rect)
    ax.set_xlim([-0.1, 2.1])
    ax.set_ylim([-1, 51])
    plt.axis('off')
    
    x1, y1 = np.meshgrid(range(w1.shape[1]), range(w1.shape[2]))
    x1, y1 = x1.flatten(), y1.flatten()
    x2, y2 = np.meshgrid(range(w2.shape[1]), range(w2.shape[2]))
    x2, y2 = x2.flatten(), y2.flatten()
    
    lines = list()
    lines += [[(0, x*49/29.), (1, y)] for x, y in zip(x1, y1)]
    lines += [[(1, x), (2, y*49/4.)] for x, y in zip(x2, y2)]
    lc = mc.LineCollection(lines, linewidths=2)
    ax.add_collection(lc)
    
    colors1 = np.array([[228,26,28],[77,175,74],[55,126,184]])/255.
    colors2 = np.array([[27,158,119],[217,95,2],[117,112,179],
                           [231,41,138],[102,166,30]])/255.
    
    ind1 = np.array([0]*10+[1]*10+[2]*10)
    ax.scatter([0]*w1.shape[1], np.arange(w1.shape[1])*49/29., color=colors1[ind1], s=4)
    ax.scatter([2]*w2.shape[2], np.arange(w2.shape[2])*49/4., color=colors2, s=4)

    y_text = 50
    fontsize = 14
    ax.text(-.05, y_text, 'ORNs', fontsize=fontsize)
    ax.text(.95, y_text, 'PNs', fontsize=fontsize)
    ax.text(1.95, y_text, 'KCs', fontsize=fontsize)
    epoch_text = ax.text(1.85, -4, '0.00 Epochs', fontsize=fontsize)

    # initialization function: plot the background of each frame
# =============================================================================
#     def init():
#         line.set_segments([])
#         return line,
# =============================================================================
    
    # animation function.  This is called sequentially
    def animate(i):
        n1, n2 = len(x1), len(x2)
        c = np.zeros((n1+n2, 4))
        c[:n1, :3] = colors1[x1//10]
        c[n1:, :3] = colors2[y2]
        w1_ = w1[i].T.flatten()
        w2_ = w2[i].T.flatten()
        c[:n1, 3] = w1_ / w1_.max()
        c[n1:, 3] = w2_ / w2_.max()
        lc.set_color(c)
        epoch_text.set_text(f'{i/40.:0.2f} Epochs')
        return ax
    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate,
                               frames=w1.shape[0], interval=20)
    writer = animation.writers['ffmpeg'](fps=30)
    # anim.save(os.path.join(path, 'movie.mp4'), writer=writer, dpi=600)
    anim.save(os.path.join(path, 'movie.mp4'), writer=writer, dpi=200)

if __name__ == '__main__':
    main_train()
    path = './files/movie'
    main_plot(path)
    w1 = np.load(os.path.join(path, 'w_layer1.npy'))
    w2 = np.load(os.path.join(path, 'w_layer2.npy'))    


