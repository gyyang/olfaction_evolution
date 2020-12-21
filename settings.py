"""User specific settings."""

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'
mpl.rcParams['mathtext.fontset'] = 'stix'

seqcmap = mpl.cm.cool_r
try:
    import seaborn as sns
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sns.color_palette('deep'))
    # seqcmap = sns.color_palette("crest_r", as_cmap=True)
except ImportError as e:
    print('Seaborn not available, default to matplotlib color scheme')

use_torch = False
cluster_path = '/share/ctn/users/gy2259/olfaction_evolution'
