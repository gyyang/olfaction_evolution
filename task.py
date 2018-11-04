import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

N_CLASS = 1000
N_ORN = 50

prototypes = np.random.rand(N_CLASS, N_ORN)


# TODO: Activate a proportion of ORN
# TODO: Sparse PN to KC, each KC receives 7/50
# TODO: (optional) fix the weight from PN to KC

N_TRAIN = 100000
N_VAL = 1000
train_odors = np.random.rand(N_TRAIN, N_ORN)
val_odors = np.random.rand(N_VAL, N_ORN)

def get_labels(odors):
    dist = euclidean_distances(prototypes, odors)
    return np.argmax(dist, axis=0)

train_labels = get_labels(train_odors)
val_labels = get_labels(val_odors)