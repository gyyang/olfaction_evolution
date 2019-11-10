import numpy as np
from matplotlib import pyplot as plt

import tools

def orthogonality(path, ix, arg = 'ortho', vlim=None):
    w_orns = tools.load_pickle(path, 'w_orn')
    w_orn = w_orns[ix]

    out = np.zeros((w_orn.shape[1], w_orn.shape[1]))
    for i in range(w_orn.shape[1]):
        for j in range(w_orn.shape[1]):
            x = np.dot(w_orn[:,i], w_orn[:,j])
            y = np.corrcoef(w_orn[:,i], w_orn[:,j])[0,1]
            if arg == 'ortho':
                out[i,j] = x
            else:
                out[i,j] = y

    rect = [0.15, 0.15, 0.65, 0.65]
    rect_cb = [0.82, 0.15, 0.02, 0.65]
    fig = plt.figure(figsize=(2.6, 2.6))
    ax = fig.add_axes(rect)

    max = np.max(abs(out))
    if not vlim:
        vlim = np.round(max, decimals=1) if max > .1 else np.round(max, decimals=2)
    im = ax.imshow(out, cmap='RdBu_r', vmin= -vlim, vmax=vlim, interpolation='none', origin='upper')

    title_txt = 'orthogonality' if arg == 'ortho' else 'correlation'
    plt.title('ORN-PN ' + title_txt)
    ax.set_xlabel('PN', labelpad = -5)
    ax.set_ylabel('PN', labelpad = -5)

    plt.axis('tight')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xticks([0, out.shape[1]])
    ax.set_yticks([0, out.shape[0]])
    # plt.xlim([-.5, out.shape[1]+0.5])
    # plt.ylim([-.5, out.shape[1]+0.5])
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[-vlim, vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Weight', fontsize=7, labelpad=-10)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.axis('tight')

    txt = '_' + title_txt + '_'
    tools.save_fig(path, txt + str(ix))