import os
import shutil
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

class input_ProtoConfig(object):
    def __init__(self):
        self.path = os.path.join(os.getcwd(), 'datasets', 'proto')

        self.n_train = 1000000
        self.n_val = 9192

        self.N_CLASS = 50
        self.N_ORN = 50
        self.N_ORN_PER_PN = 1
        self.N_PN_PER_ORN = 1
        self.ORN_NOISE_STD = 0 #make sure this param is set to zero if N_ORN_PER_PN = 1

        self.percent_generalization = 50
        self.use_combinatorial = False
        self.n_combinatorial_classes = 20
        self.combinatorial_density = .3

def _generate_repeat(config=None):
    '''
    :return:
    x = noisy ORN channels. n_samples X n_orn * neurons_per_orn
    y = noiseless PN channels
    '''
    if config is None:
        config = input_ProtoConfig()

    N_SAMPLES = config.n_train
    N_ORN = config.N_ORN
    NEURONS_PER_ORN = config.N_ORN_PER_PN
    NOISE_STD = config.ORN_NOISE_STD

    y = np.random.uniform(low=0, high=1, size= (N_SAMPLES, N_ORN))
    x = np.repeat(y, repeats=NEURONS_PER_ORN, axis=1)
    n = np.random.normal(loc=0, scale=NOISE_STD, size=x.shape)
    x += n
    return x.astype(np.float32), y.astype(np.float32)

# TODO: is this function still needed? It is basically equivalent to 100% generalization
def _generate_proto(config=None):
    """Activate all ORNs randomly."""
    if config is None:
        config = input_ProtoConfig()

    N_CLASS = config.N_CLASS
    N_ORN = config.N_ORN
    N_TRAIN = config.n_train
    N_VAL = config.n_val

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

def _generate_proto_threshold(config=None):
    """Activate all ORNs randomly.

    Only a fraction (as defined by variable PERCENTILE) of odors will
    generalize. If the similarity index (currently euclidean distance) is
    below the distance value at the percentile, the test class will not be
    any of the prototypes, but rather an additional 'default' label.



    default label will be labels(0), prototype classes will be labels(a
    1:N_CLASS)
    """
    if config is None:
        config = input_ProtoConfig()

    seed = 0
    rng = np.random.RandomState(seed)

    N_CLASS = config.N_CLASS
    N_ORN = config.N_ORN
    N_ORN_PER_PN = config.N_ORN_PER_PN
    ORN_NOISE_STD = config.ORN_NOISE_STD
    GEN_THRES = config.percent_generalization
    N_TRAIN = config.n_train
    N_VAL = config.n_val

    def get_labels(prototypes, odors):
        dist = euclidean_distances(prototypes, odors)
        highest_match = np.min(dist, axis=0)
        threshold = np.percentile(highest_match.flatten(), GEN_THRES)
        default_class = threshold * np.ones((1, dist.shape[1]))
        dist = np.vstack((default_class, dist))
        return np.argmin(dist, axis=0)

    repeat = lambda x: np.repeat(x, repeats= N_ORN_PER_PN, axis=1)
    prototypes = repeat(rng.rand(N_CLASS-1, N_ORN))
    train_odors = repeat(rng.rand(N_TRAIN, N_ORN))
    val_odors = repeat(rng.rand(N_VAL, N_ORN))

    train_labels = get_labels(prototypes, train_odors)
    val_labels = get_labels(prototypes, val_odors)
    #noise is added after getting labels
    train_odors += np.random.normal(loc=0, scale=ORN_NOISE_STD, size=train_odors.shape)
    val_odors += np.random.normal(loc=0, scale=ORN_NOISE_STD, size=val_odors.shape)
    return train_odors, train_labels, val_odors, val_labels

def save_proto(config=None):
    """Save dataset in numpy format."""
    def _convert_one_hot_label(labels, n_class):
        """Convert labels to one-hot labels."""
        label_one_hot = np.zeros((labels.size, n_class))
        label_one_hot[np.arange(labels.size), labels] = 1
        return label_one_hot

    def _generate_combinatorial_label(n_class, n_comb_class, density):
        seed = 100
        rng = np.random.RandomState(seed)
        masks = rng.rand(n_class + 1, n_comb_class)
        label_to_combinatorial = masks < density

        X = euclidean_distances(label_to_combinatorial)
        np.fill_diagonal(X, 1)
        assert np.any(X.flatten() == 0) == 0, "at least 2 combinatorial labels are the same"
        return label_to_combinatorial

    def _convert_to_combinatorial_label(labels, label_to_combinatorial_encoding):
        return label_to_combinatorial_encoding[labels, :]

    if config is None:
        config = input_ProtoConfig()

    #make and save data
    folder_name = '_' + str(config.percent_generalization) + '_generalization'
    train_x, train_y, val_x, val_y = _generate_proto_threshold(config)

    if config.use_combinatorial:
        folder_name += '_combinatorial'
        key = _generate_combinatorial_label(
            config.N_CLASS, config.n_combinatorial_classes, config.combinatorial_density)
        train_y = _convert_to_combinatorial_label(train_y, key)
        val_y = _convert_to_combinatorial_label(val_y, key)
    else:
        folder_name += '_onehot'
        train_y = _convert_one_hot_label(train_y, config.N_CLASS)
        val_y = _convert_one_hot_label(val_y, config.N_CLASS)

    folder_path = os.path.join(config.path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)

    for result, name in zip([train_x.astype(np.float32), train_y.astype(np.int32),
                             val_x.astype(np.float32), val_y.astype(np.int32)],
                            ['train_x', 'train_y', 'val_x', 'val_y']):
        np.save(os.path.join(folder_path, name), result)

    #save parameters
    save_name = os.path.join(folder_path, 'parameters.txt')
    cur_dict = {k: v for k, v in config.__dict__.items()}
    with open(save_name, 'w') as f:
        for k, v in cur_dict.items():
            f.write('%s: %s \n' % (k, v))

def save_proto_all():
    """Generate all datasets."""
    config = input_ProtoConfig()
    for use_threshold in [True, False]:
        config.USE_THRESHOLD = use_threshold
        for use_combinatorial in [True, False]:
            config.use_combinatorial = use_combinatorial
            save_proto(config)

def load_data(dataset, data_dir):
    """Load dataset."""
    def _load_proto(path):
        """Load dataset from numpy format."""
        names = ['train_x', 'train_y', 'val_x', 'val_y']
        return [np.load(os.path.join(path, name + '.npy')) for name in names]

    if dataset == 'proto':
            train_x, train_y, val_x, val_y = _load_proto(data_dir)
    elif dataset == 'repeat':
        train_x, train_y = _generate_repeat()
        val_x, val_y = _generate_repeat()
    else:
        raise ValueError('Unknown dataset type ' + str(dataset))
    return train_x, train_y, val_x, val_y


if __name__ == '__main__':
    save_proto()
    # save_proto_all()
    # proto_path = os.path.join(PROTO_PATH, '_threshold_onehot')
    # train_odors, train_labels, val_odors, val_labels = load_proto(proto_path)

