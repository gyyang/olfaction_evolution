"""File that summarizes all key results."""

MODE = 'train'
# MODE = 'analysis'

# Reproducing glomeruli-like activity
import standard_experiment.experiment as standard_experiment
if MODE == 'train':
    standard_experiment.train_orn2pn()
else:
    pass

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