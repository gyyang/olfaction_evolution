import pickle
import numpy as np
import tools
from pylab import *
import matplotlib.animation as animation

def plot_weights(weight, xlabel, ylabel, title, vlim):
    rect = [0.15, 0.15, 0.65, 0.65]
    rect_cb = [0.82, 0.15, 0.02, 0.65]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes(rect)
    im = ax.imshow(weight, cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                   interpolation='none')
    plt.axis('tight')
    plt.title(title)
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xlabel(xlabel, labelpad=-5)
    ax.set_ylabel(ylabel, labelpad=-5)
    ax.set_xticks([0, weight.shape[1]])
    ax.set_yticks([0, weight.shape[0]])
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[-vlim, vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Weight', labelpad=-10)
    plt.tick_params(axis='both', which='major')
    plt.axis('tight')
    return fig, im

def ani_frame(weight, xlabel, ylabel, title, vlim = .25):

    def update_img(n):
        tmp = weight[n, :, :]
        im.set_data(tmp)
        return im

    fig, im = plot_weights(weight[0, :, :], xlabel, ylabel, title, vlim)
    ani = animation.FuncAnimation(fig, update_img, weight.shape[0], interval=30)
    writer = animation.writers['ffmpeg'](fps=30)
    dpi = 200
    ani.save(title + '.mp4', writer=writer, dpi=dpi)
    return ani

def w_orn_reshape(w_orn):
    ind_max = np.argmax(w_orn[-1], axis=1)
    ind_sort = np.argsort(ind_max)
    w_orn = np.stack(w_orn, axis=0)
    w_orn = w_orn[:, ind_sort,:]
    print(w_orn.shape)
    w_orn = np.concatenate((
        w_orn[0:50:1, ::],
        w_orn[50:100:2, ::],
        w_orn[100:1000:20, ::],
        w_orn[1000:6000:40, ::],
        ), axis=0)
    print(w_orn.shape)
    return w_orn

def w_glo_reshape(w_glo):
    w_glo = np.stack(w_glo, axis=0)
    w_glo = w_glo[:,:,:30]
    print(w_glo.shape)
    w_glo = np.concatenate((
        w_glo[0:100:2, ::],
        w_glo[100:500:5, ::],
        w_glo[500:1000:10, ::],
        w_glo[1000:3900:20, ::],
        ), axis=0)
    print(w_glo.shape)
    return w_glo

mpl.rcParams['font.size'] = 14


# path = r'C:\Users\Peter\PycharmProjects\olfaction_evolution\files_temp\temp\000000\weights_over_time.pickle'
# with open(path, 'rb') as handle:
#     mat = pickle.load(handle)
# unzipped = list(zip(*mat))
# w_orn, w_glo = np.stack(unzipped[0],axis=0), np.stack(unzipped[1],axis=0)
# w_orn = w_orn_reshape(w_orn)
# ani_frame(w_orn, ylabel = 'From ORNs', xlabel = 'to PNs', title= 'ORN-PN connectivity')

path = r'C:\Users\Peter\PycharmProjects\olfaction_evolution\files_temp\movie_kc\000000\weights_over_time.pickle'
with open(path, 'rb') as handle:
    mat = pickle.load(handle)
unzipped = list(zip(*mat))
w_orn, w_glo = np.stack(unzipped[0],axis=0), np.stack(unzipped[1],axis=0)
w_glo = w_glo_reshape(w_glo)
ani_frame(w_glo, ylabel = 'From PNs', xlabel = 'to KCs', title= 'PN-KC connectivity', vlim=.5)


