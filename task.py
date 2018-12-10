import os
import shutil

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import tools
from configs import input_ProtoConfig


def _make_hallem_dataset(config=None):
    '''
    :param config:
    :return: hallem carlson dataset in matrix format

    110 odors * 24 ORNs, with spontaneous activity subtracted
    '''
    N_ODORS = 110
    N_ORNS = 24
    if config is None:
        config = input_ProtoConfig()
    file = config.hallem_path
    with open(file) as f:
        vec = f.readlines()
    vec = [int(x.strip()) for x in vec]
    mat = np.reshape(vec, (N_ODORS+1, N_ORNS),'F')
    spontaneous_activity = mat[-1,:]
    odor_activation = mat[:-1,:] - spontaneous_activity
    return odor_activation

def _generate_from_hallem(config=None):
    if config is None:
        config = input_ProtoConfig()
    odor_activation = _make_hallem_dataset(config)
    corr_coef = np.corrcoef(np.transpose(odor_activation))
    mask = ~np.eye(corr_coef.shape[0], dtype=bool)
    data = corr_coef[mask].flatten()
    y, x = np.histogram(data, bins=100)
    x = [(a + b) / 2 for a, b in zip(x[:-1], x[1:])]

    def gaus(x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    popt, pcov = curve_fit(gaus, x, y)

    # plt.plot(x, y, 'b+', label='data')
    plt.hist(data, bins=100, label='data')
    plt.figure()
    plt.plot(x, gaus(x, *popt), 'ro', label='fit')
    plt.show()


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
    NEURONS_PER_ORN = config.N_ORN_DUPLICATION
    NOISE_STD = config.ORN_NOISE_STD

    y = np.random.uniform(low=0, high=1, size= (N_SAMPLES, N_ORN))
    x = np.repeat(y, repeats=NEURONS_PER_ORN, axis=1)
    n = np.random.normal(loc=0, scale=NOISE_STD, size=x.shape)
    x += n
    return x.astype(np.float32), y.astype(np.float32)


def relabel(train_labels, val_labels, n_pre, n_post, rng=None):
    """Relabeing classes.

    Randomly relabel n_pre classes to n_post classes, assuming n_post<n_pre
    Assume that label 0 is still mapped to label 0

    Args:
        train_labels: a list of labels
        val_labels: a list of labels
        n_pre: the number of labels before relabeling
        n_post: the number of labels after relabeling
        rng: random number generator

    Returns:
        new_train_labels: a list of labels after relabeling
        new_val_labels: a list of labels after relabeling
    """
    if rng is None:
        rng = np.random.RandomState()

    # Generate the mapping from previous labels to new labels
    # TODO: Consider balancing the number of old labels each new label gets
    labelmap = rng.choice(range(1, n_post), size=(n_pre))
    labelmap[0] = 0  # 0 still mapped to 0

    new_train_labels = np.array([labelmap[l] for l in train_labels])
    new_val_labels = np.array([labelmap[l] for l in val_labels])
    return new_train_labels, new_val_labels


def _generate_proto_threshold(config=None, seed=0):
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

    rng = np.random.RandomState(seed)

    if config.relabel:
        N_PROTO = config.n_trueclass
    else:
        N_PROTO = config.N_CLASS
    N_ORN = config.N_ORN
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

    def add_bias(matrix, bias):
        """Add correlated bias."""
        bias_vector = rng.normal(0, bias, size=matrix.shape[0])
        matrix += bias_vector.reshape(-1,1)
        return matrix

    lamb = 1
    bias = 0

    prototypes = rng.uniform(0, lamb, (N_PROTO-1, N_ORN))
    train_odors = rng.uniform(0, lamb, (N_TRAIN, N_ORN))
    val_odors = rng.uniform(0, lamb, (N_VAL, N_ORN))
    prototypes = add_bias(prototypes, bias)
    train_odors = add_bias(train_odors, bias)
    val_odors = add_bias(val_odors, bias)
    prototypes.clip(min=0)
    train_odors.clip(min=0)
    val_odors.clip(min=0)
    train_odors = train_odors.astype(np.float32)
    val_odors = val_odors.astype(np.float32)

    if config.distort_input:
        # Distort the distance metric with random MLP
        Ms = [rng.randn(N_ORN, N_ORN) / np.sqrt(N_ORN) for _ in range(5)]

        relu = lambda x: x * (x > 0.)

        def transform(x):
            for M in Ms:
                # x = np.tanh(np.dot(x, M))
                x = relu(np.dot(x, M))
                x = x / np.std(x) * 0.3
            return x

        prototypes_distort = transform(prototypes)
        train_odors_distort = transform(train_odors)
        val_odors_distort = transform(val_odors)
        train_labels = get_labels(prototypes_distort, train_odors_distort)
        val_labels = get_labels(prototypes_distort, val_odors_distort)

    else:
        train_labels = get_labels(prototypes, train_odors)
        val_labels = get_labels(prototypes, val_odors)

    if config.shuffle_label:
        # Shuffle the labels
        rng.shuffle(train_labels)
        rng.shuffle(val_labels)

    if config.relabel:
        train_labels, val_labels = relabel(
            train_labels, val_labels, N_PROTO, config.N_CLASS, rng)

    # Repeat odors for duplication of ORNs
    if not config.replicate_orn_with_tiling:
        N_ORN_PER_PN = config.N_ORN_DUPLICATION
        repeat = lambda x: np.repeat(x, repeats=N_ORN_PER_PN, axis=1)
        train_odors = repeat(train_odors)
        val_odors = repeat(val_odors)

        # noise is added after getting labels
        ORN_NOISE_STD = config.ORN_NOISE_STD
        train_odors += rng.normal(loc=0, scale=ORN_NOISE_STD, size=train_odors.shape)
        val_odors += rng.normal(loc=0, scale=ORN_NOISE_STD, size=val_odors.shape)

    assert train_odors.dtype == np.float32

    return train_odors, train_labels, val_odors, val_labels


def _gen_folder_name(config, seed):
    """Automatically generate folder name."""
    auto_folder_name = '_' + str(
        config.percent_generalization) + '_generalization'

    # Convert labels
    if config.use_combinatorial:
        auto_folder_name += '_combinatorial'
    else:
        auto_folder_name += '_onehot'

    auto_folder_name += '_s' + str(seed)
    return auto_folder_name


def save_proto(config=None, seed=0, folder_name=None):
    """Save dataset in numpy format."""
    def _convert_one_hot_label(labels, n_class):
        """Convert labels to one-hot labels."""
        label_one_hot = np.zeros((labels.size, n_class))
        label_one_hot[np.arange(labels.size), labels] = 1
        return label_one_hot

    def _generate_combinatorial_label(n_class, n_comb_class, density):
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

    # make and save data
    train_x, train_y, val_x, val_y = _generate_proto_threshold(config, seed=seed)

    # Convert labels
    if config.use_combinatorial:
        key = _generate_combinatorial_label(
            config.N_CLASS, config.n_combinatorial_classes, config.combinatorial_density)
        train_y = _convert_to_combinatorial_label(train_y, key)
        val_y = _convert_to_combinatorial_label(val_y, key)
    else:
        train_y = _convert_one_hot_label(train_y, config.N_CLASS)
        val_y = _convert_one_hot_label(val_y, config.N_CLASS)

    if folder_name is None:
        folder_name = _gen_folder_name(config, seed)

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
    tools.save_config(config, folder_path)
    return folder_path


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
    # save_proto()
    # save_proto_all()
    # proto_path = os.path.join(PROTO_PATH, '_threshold_onehot')
    # train_odors, train_labels, val_odors, val_labels = load_proto(proto_path)
    # _make_hallem_dataset()
    _generate_from_hallem()
