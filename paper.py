"""File that summarizes all key results."""

# MODE = 'train'
MODE = 'analysis'

# Reproducing glomeruli-like activity
import standard.experiment as standard_experiment
import standard.analysis as standard_analysis
save_path = './files/standard/orn2pn'
if MODE == 'train':
    standard_experiment.train_orn2pn(save_path)
else:
    standard_analysis.plot_progress(save_path)

# Varying #PN and #KC
if MODE == 'train':
    pass
else:
    pass

# Varying the noise level while varying #KC
if MODE == 'train':
    pass
else:
    pass

# Reproducing sparse PN-KC connectivity
if MODE == 'train':
    pass
else:
    pass

# Varying PN-KC connectivity sparseness
if MODE == 'train':
    pass
else:
    pass

# Reproducing random connectivity
if MODE == 'train':
    pass
else:
    pass

# The impact of various normalization
if MODE == 'train':
    pass
else:
    pass

# The impact of various task variants
if MODE == 'train':
    pass
else:
    pass