import os

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt


PROTO_N_CLASS = 50
PROTO_N_ORN = 50
PROTO_PATH = os.path.join(os.getcwd(), 'datasets', 'proto')
if not os.path.exists(PROTO_PATH):
    os.makedirs(PROTO_PATH)

N_TRAIN = 1000000
N_VAL = 9192

N_COMBINATORIAL_CLASSES = 20
COMBINATORIAL_DENSITY = .3

def _generate_proto():
    """Activate all ORNs randomly."""
    N_CLASS = PROTO_N_CLASS
    N_ORN = PROTO_N_ORN

    seed = 0
    rng = np.random.RandomState(seed)

    prototypes = rng.rand(N_CLASS, N_ORN).astype(np.float32)
    train_odors = rng.rand(N_TRAIN, N_ORN).astype(np.float32)
    val_odors = rng.rand(N_VAL, N_ORN).astype(np.float32)

    def get_labels(odors):
        dist = euclidean_distances(prototypes, odors)
        return np.argmin(dist, axis=0).astype(np.int32)

    train_labels = get_labels(train_odors)
    val_labels = get_labels(val_odors)
    return train_odors, train_labels, val_odors, val_labels


def _generate_proto_threshold():
    """Activate all ORNs randomly.
    Only a fraction (as defined by variable PERCENTILE) of odors will
    generalize. If the similarity index (currently euclidean distance) is
    below the distance value at the percentile, the test class will not be
    any of the prototypes, but rather an additional 'default' label.

    default label will be labels(0), prototype classes will be labels(a
    1:N_CLASS)"""
    N_CLASS = PROTO_N_CLASS
    N_ORN = PROTO_N_ORN
    PERCENTILE = 50

    seed = 0
    rng = np.random.RandomState(seed)

    prototypes = rng.rand(N_CLASS-1, N_ORN).astype(np.float32)
    train_odors = rng.rand(N_TRAIN, N_ORN).astype(np.float32)
    val_odors = rng.rand(N_VAL, N_ORN).astype(np.float32)

    def get_labels(odors):
        dist = euclidean_distances(prototypes, odors)
        highest_match = np.max(dist, axis=0)
        threshold = np.percentile(highest_match.flatten(), PERCENTILE)
        default_class = threshold * np.ones((1, dist.shape[1]))
        dist = np.vstack((default_class, dist))
        return np.argmin(dist, axis=0).astype(np.int32)

    train_labels = get_labels(train_odors)
    # hist, bin_edges = np.histogram(train_labels, bins= N_CLASS, range=(0,N_CLASS-1))
    # plt.hist(train_labels, bins= N_CLASS, range=(1,N_CLASS), normed=True)
    val_labels = get_labels(val_odors)
    return train_odors, train_labels, val_odors, val_labels

def _convert_one_hot_label(labels, label_range):
    label_one_hot = np.zeros((labels.size, label_range))
    label_one_hot[np.arange(labels.size), labels] = 1
    return label_one_hot

def _generate_combinatorial_label(label_range, n_classes, density):
    masks = np.random.rand(label_range+1, n_classes)
    label_to_combinatorial = masks < density

    X = euclidean_distances(label_to_combinatorial)
    np.fill_diagonal(X, 1)
    assert np.any(X.flatten() == 0) == 0, "at least 2 combinatorial labels are the same"
    return label_to_combinatorial

def _convert_to_combinatorial_label(labels, label_to_combinatorial_encoding):
    return label_to_combinatorial_encoding[labels, :]

def save_proto_hard(argThreshold = 1, argCombinatorial = 0):
    folder_name = ''
    if argThreshold:
        folder_name += '_threshold'
        train_x, train_y, val_x, val_y = _generate_proto_threshold()
    else:
        folder_name += '_no-threshold'
        train_x, train_y, val_x, val_y = _generate_proto()

    if argCombinatorial:
        folder_name += '_combinatorial'
        key = _generate_combinatorial_label(PROTO_N_CLASS, N_COMBINATORIAL_CLASSES, COMBINATORIAL_DENSITY)
        train_y_modified = _convert_to_combinatorial_label(train_y, key)
        val_y_modified = _convert_to_combinatorial_label(val_y, key)
    else:
        folder_name += '_one-hot'
        train_y_modified = _convert_one_hot_label(train_y, PROTO_N_CLASS)
        val_y_modified = _convert_one_hot_label(val_y, PROTO_N_CLASS)

    path = os.path.join(PROTO_PATH, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    for result, name in zip([train_x, train_y_modified, val_x, val_y_modified],
                            ['train_x', 'train_y', 'val_x', 'val_y']):
        np.save(os.path.join(path, name), result)

def save_proto():
    """Save dataset in numpy format."""
    results = _generate_proto()
    for result, name in zip(results, ['train_x', 'train_y', 'val_x', 'val_y']):
        np.save(os.path.join(PROTO_PATH, name), result)


def load_proto(path=PROTO_PATH):
    """Load dataset from numpy format."""
    names = ['train_x', 'train_y', 'val_x', 'val_y']
    return [np.load(os.path.join(path, name+'.npy')) for name in names]


def generate_sparse_active():
    # TODO: TBF
    N_CLASS = 1000
    N_ORN = 50

    prototypes = np.random.rand(N_CLASS, N_ORN).astype(np.float32)

    N_TRAIN = 100000
    N_VAL = 1000
    train_odors = np.random.rand(N_TRAIN, N_ORN).astype(np.float32)
    val_odors = np.random.rand(N_VAL, N_ORN).astype(np.float32)

    def get_labels(odors):
        dist = euclidean_distances(prototypes, odors)
        return np.argmax(dist, axis=0).astype(np.int32)

    train_labels = get_labels(train_odors)
    val_labels = get_labels(val_odors)
    return train_odors, train_labels


class smallConfig():
    N_SAMPLES = 1000

    N_ORN = 30
    NEURONS_PER_ORN = 50
    NOISE_STD = .2

def generate_repeat():
    '''
    :return:
    x = noisy ORN channels. n_samples X n_orn * neurons_per_orn
    y = noiseless PN channels
    '''
    config = smallConfig()
    n_samples = config.N_SAMPLES
    neurons_per_orn = config.NEURONS_PER_ORN
    n_orn = config.N_ORN
    noise_std = config.NOISE_STD

    y = np.random.uniform(low=0, high=1, size= (n_samples, n_orn))
    x = np.repeat(y, repeats = neurons_per_orn, axis = 1)
    n = np.random.normal(loc=0, scale= noise_std, size= x.shape)
    x += n
    return x.astype(np.float32), y.astype(np.float32)


if __name__ == '__main__':
    proto_path = os.path.join(os.getcwd(), 'datasets', 'proto', '_threshold_combinatorial')
    save_proto_hard(1, 1)
    train_odors, train_labels, val_odors, val_labels = load_proto(proto_path)