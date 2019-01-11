import pickle
import numpy as np
import tools
from pylab import *
import matplotlib.animation as animation

def plot_weights(weight):
    rect = [0.15, 0.15, 0.65, 0.65]
    rect_cb = [0.82, 0.15, 0.02, 0.65]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes(rect)
    vlim = .5
    im = ax.imshow(weight, cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                   interpolation='none')
    plt.axis('tight')
    plt.title('ORN-PN connectivity')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xlabel('To PNs', labelpad=-5)
    ax.set_ylabel('From ORNs', labelpad=-5)
    ax.set_xticks([0, weight.shape[1]])
    ax.set_yticks([0, weight.shape[0]])
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[-vlim, vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Weight', labelpad=-10)
    plt.tick_params(axis='both', which='major')
    plt.axis('tight')
    return fig, im

def ani_frame(weight):

    def update_img(n):
        tmp = weight[n, :, :]
        im.set_data(tmp)
        return im

    fig, im = plot_weights(weight[0, :, :])
    ani = animation.FuncAnimation(fig, update_img, weight.shape[0], interval=30)
    writer = animation.writers['ffmpeg'](fps=30)
    dpi = 100
    ani.save('demo.mp4', writer=writer, dpi=dpi)
    return ani

mpl.rcParams['font.size'] = 14
path = '/Users/peterwang/Desktop/PYTHON/olfaction_evolution' \
        '/vary_size_experiment/weight_over_time.pickle'

with open(path, 'rb') as handle:
    mat = pickle.load(handle)

ind_max = np.argmax(mat[-1], axis=0)
ind_sort = np.argsort(ind_max)
mat = np.stack(mat,axis=0)
mat = mat[:,:,ind_sort]
print(mat.shape)
mat = np.concatenate((mat[0:1000:10,::],
                      mat[1000:2000:20,::],
                      mat[2000:4000:40,::],
                      mat[4000:8000:80,::],
                      mat[8000:19500:200,::]),axis=0)
print(mat.shape)
ani_frame(mat)




