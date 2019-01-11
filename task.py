import os
import shutil

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import tools
from configs import input_ProtoConfig


def _get_labels(prototypes, odors, percent_generalization):
    dist = euclidean_distances(prototypes, odors)
    highest_match = np.min(dist, axis=0)
    threshold = np.percentile(highest_match.flatten(), percent_generalization)
    default_class = threshold * np.ones((1, dist.shape[1]))
    dist = np.vstack((default_class, dist))
    return np.argmin(dist, axis=0)


def _mask_orn_activation(prototypes):
    mask = np.zeros_like(prototypes)
    n_samples = mask.shape[0]
    n_orn = mask.shape[1]
    mask = np.zeros_like(prototypes, dtype=int)
    n_orn_active = np.random.random_integers(1, n_orn, size=n_samples)
    for i in range(n_samples):
        mask[i, :n_orn_active[i]] = 1
        np.random.shuffle(mask[i, :])
    out = np.multiply(prototypes, mask)
    return out

def _relabel(train_labels, val_labels, n_pre, n_post, rng=None):
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

def junk_code():
    # def add_bias(matrix, bias):
    #     """Add correlated bias."""
    #     bias_vector = rng.normal(0, bias, size=matrix.shape[0])
    #     matrix += bias_vector.reshape(-1,1)
    #     return matrix
    # 
    # lamb = 1
    # bias = 0
    # prototypes = add_bias(prototypes, bias)
    # train_odors = add_bias(train_odors, bias)
    # val_odors = add_bias(val_odors, bias)
    # 
    # prototypes.clip(min=0)
    # train_odors.clip(min=0)
    # val_odors.clip(min=0)
    pass

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
        realistic_orn_mean,
        realistic_orn_mask,
        n_combinatorial_classes=None,
        combinatorial_density=None,
        n_class_valence=None,
        n_proto_valence=None,
        has_special_odors=False,
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
    multi_head = label_type == 'multi_head_sparse'

    # the number of prototypes
    n_proto = n_trueclass if relabel else n_class

    max_activation = 1
    if multi_head:
        ratio = int(n_proto /  n_orn)
        n_good_odor = n_bad_odor = n_proto_valence
        p_good_odor = p_bad_odor = 1.0* (n_proto_valence/n_proto) * ratio
        n_train_good = int(p_good_odor*n_train)
        n_val_good = int(p_good_odor*n_val)
        n_train_bad = int(p_bad_odor * n_train)
        n_val_bad = int(p_bad_odor * n_val)
        n_train_neutral = n_train - n_train_good - n_train_bad
        n_val_neutral = n_val - n_val_good - n_val_bad

    if multi_head and has_special_odors:
        # TODO(gryang): make this code not so ugly
        n_neutral_odor = n_proto - 1 - (n_good_odor + n_bad_odor)
        prototypes_neutral = rng.uniform(0, max_activation, (n_neutral_odor, n_orn))
        prototypes_good = np.zeros((n_good_odor, n_orn))
        prototypes_good[range(n_good_odor), range(n_good_odor)] = 5.
        prototypes_bad = np.zeros((n_bad_odor, n_orn))
        prototypes_bad[range(n_bad_odor), range(n_good_odor, n_good_odor+n_bad_odor)] = 5.
        prototypes = np.concatenate((prototypes_neutral, prototypes_good, prototypes_bad), axis=0)

        train_odors_neutral = rng.uniform(0, max_activation, (n_train_neutral, n_orn))
        ind = rng.randint(n_good_odor, size=(n_train_good))
        train_odors_good = prototypes_good[ind] + rng.uniform(0, 1, (n_train_good, n_orn))
        ind = rng.randint(n_bad_odor, size=(n_train_bad))
        train_odors_bad = prototypes_bad[ind] + rng.uniform(0, 1, (n_train_bad, n_orn))
        train_odors = np.concatenate((train_odors_neutral, train_odors_good, train_odors_bad), axis=0)
        train_labels_valence = np.array([0]*n_train_neutral+[1]*n_train_good+[2]*n_train_bad)
        ind_shuffle = np.arange(n_train)
        rng.shuffle(ind_shuffle)
        train_odors = train_odors[ind_shuffle, :]
        train_labels_valence = train_labels_valence[ind_shuffle]

        val_odors_neutral = rng.uniform(0, max_activation, (n_val_neutral, n_orn))
        ind = rng.randint(n_good_odor, size=(n_val_good))
        val_odors_good = prototypes_good[ind] + rng.uniform(0, 1, (n_val_good, n_orn))
        ind = rng.randint(n_bad_odor, size=(n_val_bad))
        val_odors_bad = prototypes_bad[ind] + rng.uniform(0, 1, (n_val_bad, n_orn))
        val_odors = np.concatenate(
            (val_odors_neutral, val_odors_good, val_odors_bad), axis=0)
        val_labels_valence = np.array([0]*n_val_neutral+[1]*n_val_good+[2]*n_val_bad)
        ind_shuffle = np.arange(n_val)
        rng.shuffle(ind_shuffle)
        val_odors = val_odors[ind_shuffle, :]
        val_labels_valence = val_labels_valence[ind_shuffle]

    else:
        prototypes = rng.uniform(0, max_activation, (n_proto-1, n_orn))
        train_odors = rng.uniform(0, max_activation, (n_train, n_orn))
        val_odors = rng.uniform(0, max_activation, (n_val, n_orn))

    if realistic_orn_mask:
        prototypes = _mask_orn_activation(prototypes)
        train_odors = _mask_orn_activation(train_odors)
        val_odors = _mask_orn_activation(val_odors)

    if realistic_orn_mean:
        prototypes *= np.random.uniform(0, 1, prototypes.shape[0]).reshape(-1,1)
        train_odors *= np.random.uniform(0, 1, train_odors.shape[0]).reshape(-1,1)
        val_odors *= np.random.uniform(0, 1, val_odors.shape[0]).reshape(-1,1)

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

    train_labels = _get_labels(prototypes, train_odors_forlabels, percent_generalization)
    val_labels = _get_labels(prototypes, val_odors_forlabels, percent_generalization)

    debug = False
    if debug:
        plt.hist(np.sum(train_odors, axis=1))
        plt.show()
        sums = np.sum(train_odors, axis=1, keepdims=True)
        lol = np.divide(train_odors, sums)
        plt.hist(np.sum(lol, axis=1))
        plt.show()
        plt.hist(train_labels, bins=100)
        plt.show()
        return

    if shuffle_label:
        # Shuffle the labels
        rng.shuffle(train_labels)
        rng.shuffle(val_labels)

    if relabel:
        train_labels, val_labels = relabel(
            train_labels, val_labels, n_proto, n_class, rng)

    assert train_odors.dtype == np.float32

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
        if not has_special_odors:
            # labels 0-4 will be good, 5-9 will be bad, others will be neutral
            train_labels_valence = np.zeros_like(train_labels)
            train_labels_valence[(0<=train_labels)*(train_labels<5)] = 1
            train_labels_valence[(5 <= train_labels) * (train_labels < 10)] = 2
            val_labels_valence = np.zeros_like(val_labels)
            val_labels_valence[(0 <= val_labels) * (val_labels < 5)] = 1
            val_labels_valence[(5 <= val_labels) * (val_labels < 10)] = 2

        train_labels = np.stack([train_labels, train_labels_valence]).T
        val_labels = np.stack([val_labels, val_labels_valence]).T
    else:
        raise ValueError('Unknown label type: ', str(label_type))

    return train_odors, train_labels, val_odors, val_labels


def _gen_folder_name(config, seed):
    """Automatically generate folder name."""
    auto_folder_name = '_s' + str(seed)
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
        realistic_orn_mean= config.realistic_orn_mask,
        realistic_orn_mask= config.realistic_orn_mean,
        n_combinatorial_classes=config.n_combinatorial_classes,
        combinatorial_density=config.combinatorial_density,
        n_class_valence=config.n_class_valence,
        n_proto_valence=config.n_proto_valence,
        has_special_odors=config.has_special_odors,
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