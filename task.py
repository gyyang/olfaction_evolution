import os
import shutil

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt


PROTO_N_CLASS = 50  # TODO: make this dependent on dataset
PROTO_N_ORN = 50
PROTO_N_ORN_PER_PN = 10
PROTO_ORN_NOISE_STD = 0.5
PROTO_PATH = os.path.join(os.getcwd(), 'datasets', 'proto')
if not os.path.exists(PROTO_PATH):
    os.makedirs(PROTO_PATH)

N_TRAIN = 1000000
N_VAL = 9192

PERCENT_GENERALIZATION = 50
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


def _generate_proto_threshold(percent_generalization=PERCENT_GENERALIZATION):
    """Activate all ORNs randomly.

    Only a fraction (as defined by variable PERCENTILE) of odors will
    generalize. If the similarity index (currently euclidean distance) is
    below the distance value at the percentile, the test class will not be
    any of the prototypes, but rather an additional 'default' label.



    default label will be labels(0), prototype classes will be labels(a
    1:N_CLASS)
    """
    def get_labels(prototypes, odors):
        dist = euclidean_distances(prototypes, odors)
        highest_match = np.min(dist, axis=0)
        threshold = np.percentile(highest_match.flatten(), percent_generalization)
        default_class = threshold * np.ones((1, dist.shape[1]))
        dist = np.vstack((default_class, dist))
        return np.argmin(dist, axis=0).astype(np.int32)

    seed = 0
    rng = np.random.RandomState(seed)

    #TODO: add all parameters from input config file
    N_CLASS = PROTO_N_CLASS
    N_ORN = PROTO_N_ORN
    N_ORN_PER_PN = PROTO_N_ORN_PER_PN
    ORN_NOISE_STD = PROTO_ORN_NOISE_STD

    repeat = lambda x: np.repeat(x, repeats= N_ORN_PER_PN, axis=1)
    prototypes = repeat(rng.rand(N_CLASS-1, N_ORN).astype(np.float32))
    train_odors = repeat(rng.rand(N_TRAIN, N_ORN).astype(np.float32))
    val_odors = repeat(rng.rand(N_VAL, N_ORN).astype(np.float32))

    train_labels = get_labels(prototypes, train_odors)
    val_labels = get_labels(prototypes, val_odors)
    #noise is added after getting labels
    train_odors += np.random.normal(loc=0, scale=ORN_NOISE_STD, size=train_odors.shape)
    val_odors += np.random.normal(loc=0, scale=ORN_NOISE_STD, size=val_odors.shape)
    return train_odors, train_labels, val_odors, val_labels


def _convert_one_hot_label(labels, n_label):
    """Convert labels to one-hot labels."""
    label_one_hot = np.zeros((labels.size, n_label))
    label_one_hot[np.arange(labels.size), labels] = 1
    return label_one_hot


def _generate_combinatorial_label(label_range, n_classes, density):
    seed = 100
    rng = np.random.RandomState(seed)
    masks = rng.rand(label_range+1, n_classes)
    label_to_combinatorial = masks < density

    X = euclidean_distances(label_to_combinatorial)
    np.fill_diagonal(X, 1)
    assert np.any(X.flatten() == 0) == 0, "at least 2 combinatorial labels are the same"
    return label_to_combinatorial


def _convert_to_combinatorial_label(labels, label_to_combinatorial_encoding):
    return label_to_combinatorial_encoding[labels, :]


def save_proto(use_threshold=True, use_combinatorial=False,
               percent_generalization=PERCENT_GENERALIZATION):
    """Save dataset in numpy format."""

    folder_name = ''
    if use_threshold:
        folder_name += '_threshold'
        train_x, train_y, val_x, val_y = _generate_proto_threshold(percent_generalization)
    else:
        folder_name += '_no_threshold'
        train_x, train_y, val_x, val_y = _generate_proto()

    if use_combinatorial:
        folder_name += '_combinatorial'
        key = _generate_combinatorial_label(
            PROTO_N_CLASS, N_COMBINATORIAL_CLASSES, COMBINATORIAL_DENSITY)
        train_y = _convert_to_combinatorial_label(train_y, key)
        val_y = _convert_to_combinatorial_label(val_y, key)
    else:
        folder_name += '_onehot'
        train_y = _convert_one_hot_label(train_y, PROTO_N_CLASS)
        val_y = _convert_one_hot_label(val_y, PROTO_N_CLASS)

    path = os.path.join(PROTO_PATH, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)

    for result, name in zip([train_x, train_y, val_x, val_y],
                            ['train_x', 'train_y', 'val_x', 'val_y']):
        np.save(os.path.join(path, name), result)


def save_proto_all():
    """Generate all datasets."""
    for use_threshold in [True, False]:
        for use_combinatorial in [True, False]:
            save_proto(use_threshold, use_combinatorial)


def load_proto(path=PROTO_PATH):
    """Load dataset from numpy format."""
    names = ['train_x', 'train_y', 'val_x', 'val_y']
    return [np.load(os.path.join(path, name+'.npy')) for name in names]


def generate_repeat():
    '''
    :return:
    x = noisy ORN channels. n_samples X n_orn * neurons_per_orn
    y = noiseless PN channels
    '''
    N_SAMPLES = 1000

    N_ORN = 30
    NEURONS_PER_ORN = 50
    NOISE_STD = .2

    y = np.random.uniform(low=0, high=1, size= (N_SAMPLES, N_ORN))
    x = np.repeat(y, repeats=NEURONS_PER_ORN, axis=1)
    n = np.random.normal(loc=0, scale=NOISE_STD, size=x.shape)
    x += n
    return x.astype(np.float32), y.astype(np.float32)


def load_data(dataset, data_dir=None):
    """Load dataset."""
    if dataset == 'proto':
        if data_dir is None:
            train_x, train_y, val_x, val_y = load_proto()
        else:
            train_x, train_y, val_x, val_y = load_proto(data_dir)
    elif dataset == 'repeat':
        train_x, train_y = generate_repeat()
        val_x, val_y = generate_repeat()
    else:
        raise ValueError('Unknown dataset type ' + str(dataset))
    return train_x, train_y, val_x, val_y


if __name__ == '__main__':
    save_proto(use_threshold=True, use_combinatorial=False,
               percent_generalization=PERCENT_GENERALIZATION)
    # save_proto_all()
    # proto_path = os.path.join(PROTO_PATH, '_threshold_onehot')
    # train_odors, train_labels, val_odors, val_labels = load_proto(proto_path)

