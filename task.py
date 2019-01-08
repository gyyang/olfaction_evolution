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


def _convert_one_hot_label(labels, n_class):
    """Convert labels to one-hot labels."""
    label_one_hot = np.zeros((labels.size, n_class))
    label_one_hot[np.arange(labels.size), labels] = 1
    return label_one_hot


def _generate_combinatorial_label(n_class, n_comb_class, density, rng):
    masks = rng.rand(n_class + 1, n_comb_class)
    label_to_combinatorial = masks < density

    X = euclidean_distances(label_to_combinatorial)
    np.fill_diagonal(X, 1)
    assert np.any(X.flatten() == 0) == 0, "at least 2 combinatorial labels are the same"
    return label_to_combinatorial


def _convert_to_combinatorial_label(labels, label_to_combinatorial_encoding):
    return label_to_combinatorial_encoding[labels, :]


def _generate_proto_threshold(
        n_orn,
        n_class,
        percent_generalization,
        n_train,
        n_val,
        label_type,
        vary_concentration,
        distort_input,
        shuffle_label,
        relabel,
        n_trueclass,
        n_combinatorial_classes=None,
        combinatorial_density=None,
        n_class_valence=None,
        seed=0):
    """Activate all ORNs randomly.

    Only a fraction (as defined by variable PERCENTILE) of odors will
    generalize. If the similarity index (currently euclidean distance) is
    below the distance value at the percentile, the test class will not be
    any of the prototypes, but rather an additional 'default' label.

    default label will be labels(0), prototype classes will be labels(a
    1:N_CLASS)

    Args:
        n_orn: int, number of ORN types
        n_class: int, number of output class
        percent_generalization: float, percentage of odors that generalize
        n_train: int, number of training examples
        n_val: int, number of validation examples
        label_type: str, one of 'one_hot', 'sparse', 'combinatorial'
        vary_concentration: bool. if True, prototypes are all unit vectors,
            concentrations are varied independently from odor identity
        distort_input: bool. if True, distort the input space
        shuffle_label: bool. if True, shuffle the class label for each example
        relabel: bool. if True, true classes are relabeled to get the output classes
        n_trueclass: int, the number of True classes
        n_combinatorial_classes: int, the number of combinatorial classes
        combinatorial_density: float, the density of combinatorial code
        n_class_valence: int, the number of valence class
        seed: int, random seed to generate the dataset

    Returns:
        train_odors: np array (n_train, n_orn)
        train_labels: np array (n_train, n_class)
        val_odors: np array (n_val, n_orn)
        val_labels: np array (n_val, n_class)
    """
    rng = np.random.RandomState(seed)

    # the number of prototypes
    n_proto = n_trueclass if relabel else n_class

    def get_labels(prototypes, odors):
        dist = euclidean_distances(prototypes, odors)
        highest_match = np.min(dist, axis=0)
        threshold = np.percentile(highest_match.flatten(), percent_generalization)
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

    prototypes = rng.uniform(0, lamb, (n_proto-1, n_orn))
    train_odors = rng.uniform(0, lamb, (n_train, n_orn))
    val_odors = rng.uniform(0, lamb, (n_val, n_orn))
    prototypes = add_bias(prototypes, bias)
    train_odors = add_bias(train_odors, bias)
    val_odors = add_bias(val_odors, bias)
    prototypes.clip(min=0)
    train_odors.clip(min=0)
    val_odors.clip(min=0)
    train_odors = train_odors.astype(np.float32)
    val_odors = val_odors.astype(np.float32)

    # ORN activity for computing labels
    train_odors_forlabels, val_odors_forlabels = train_odors, val_odors

    if distort_input:
        # Distort the distance metric with random MLP
        Ms = [rng.randn(n_orn, n_orn) / np.sqrt(n_orn) for _ in range(5)]
        relu = lambda x: x * (x > 0.)

        def _transform(x):
            for M in Ms:
                # x = np.tanh(np.dot(x, M))
                x = relu(np.dot(x, M))
                x = x / np.std(x) * 0.3
            return x

        prototypes = _transform(prototypes)
        train_odors_forlabels = _transform(train_odors_forlabels)
        val_odors_forlabels = _transform(val_odors_forlabels)

    if vary_concentration:
        # normalize prototypes and train/val_odors_forlabels to unit vectors
        def _normalize(x):
            norm = np.linalg.norm(x, axis=1)
            x = (x.T/norm).T
            return x

        prototypes = _normalize(prototypes)
        train_odors_forlabels = _normalize(train_odors_forlabels)
        val_odors_forlabels = _normalize(val_odors_forlabels)

    train_labels = get_labels(prototypes, train_odors_forlabels)
    val_labels = get_labels(prototypes, val_odors_forlabels)

    if shuffle_label:
        # Shuffle the labels
        rng.shuffle(train_labels)
        rng.shuffle(val_labels)

    if relabel:
        train_labels, val_labels = relabel(
            train_labels, val_labels, n_proto, n_class, rng)

    assert train_odors.dtype == np.float32

    if label_type == 'multi_head_sparse':
        # generating labels for valence
        train_labels_valence = rng.randint(
            n_class_valence, size=(n_train,), dtype=np.int32)
        val_labels_valence = rng.randint(
            n_class_valence, size=(n_val,), dtype=np.int32)

    # Convert labels
    if label_type == 'combinatorial':
        key = _generate_combinatorial_label(
            n_class, n_combinatorial_classes,
            combinatorial_density, rng)
        train_labels = _convert_to_combinatorial_label(train_labels, key)
        val_labels = _convert_to_combinatorial_label(val_labels, key)
    elif label_type == 'one_hot':
        train_labels = _convert_one_hot_label(train_labels, n_class)
        val_labels = _convert_one_hot_label(val_labels, n_class)
    elif label_type == 'sparse':
        pass
    elif label_type == 'multi_head_sparse':
        train_labels = np.stack([train_labels, train_labels_valence]).T
        val_labels = np.stack([val_labels, val_labels_valence]).T
    else:
        raise ValueError('Unknown label type: ', str(label_type))

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

    if config is None:
        config = input_ProtoConfig()

    # make and save data
    train_x, train_y, val_x, val_y = _generate_proto_threshold(
        n_orn=config.N_ORN,
        n_class=config.N_CLASS,
        percent_generalization=config.percent_generalization,
        n_train=config.n_train,
        n_val=config.n_val,
        label_type=config.label_type,
        vary_concentration=config.vary_concentration,
        distort_input=config.distort_input,
        shuffle_label=config.shuffle_label,
        relabel=config.relabel,
        n_trueclass=config.n_trueclass,
        n_combinatorial_classes=config.n_combinatorial_classes,
        combinatorial_density=config.combinatorial_density,
        n_class_valence=config.n_class_valence,
        seed=0)

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
    # _generate_from_hallem()
    save_proto()