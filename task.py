import os
import shutil

import numpy as np
import scipy.stats as stats
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import tools
from configs import input_ProtoConfig, InputAutoEncode


def _get_labels(prototypes, odors, percent_generalization, weights=None):
    dist = euclidean_distances(prototypes, odors)
    if percent_generalization < 100:
        highest_match = np.min(dist, axis=0)
        threshold = np.percentile(highest_match.flatten(), percent_generalization)
        default_class = (1e-6+threshold) * np.ones((1, dist.shape[1]))
        dist = np.vstack((default_class, dist))

    if weights is not None:
        assert dist.shape[0] == weights.shape[0], 'not the same dimension'
        weights = np.repeat(weights.reshape(-1,1), dist.shape[1], axis=1)
        dist = weights * dist
    return np.argmin(dist, axis=0)


def _spread_orn_activity(prototypes, spread = 0):
    '''
    :param prototypes: (n_samples, n_neurons)
    :param spread: varies from [0, 1). 0 means no spread, 1 means maximum spread.
    :return:
    '''
    assert spread >= 0 and spread < 1, 'spread is not within range of [0, 1)'
    mask_degree = (1 - spread) / 2
    n_samples = prototypes.shape[0]
    n_orn = prototypes.shape[1]
    low, high = mask_degree, 1 - mask_degree
    print(low, high)

    low_samples = np.random.uniform(0, low, n_samples)
    high_samples = np.random.uniform(high, 1, n_samples)
    samples = np.concatenate((low_samples, high_samples))
    activity = np.random.choice(samples, size=n_samples, replace=False)
    out = prototypes * np.repeat(activity.reshape(-1,1), n_orn, axis=1)
    return out


def _mask_orn_activation_row(prototypes, spread=None):
    '''
    :param prototypes:
    :param spread: varies from [0, 1). 0 means no spread, 1 means maximum spread.
    :return:
    '''
    assert spread >= 0 and spread < 1, 'spread is not within range of [0, 1)'

    n_samples, n_orn = prototypes.shape
    mask_degree = np.round(n_orn * (1 - spread) / 2).astype(int)
    # Small number of ORNs active
    list_of_numbers = list(range(1, mask_degree))
    # Large number of ORNs active
    list_of_numbers = list_of_numbers + list(range(n_orn - mask_degree, n_orn))
    print(list_of_numbers)

    # For each sample odor, how many ORNs will be active
    n_orn_active = np.random.choice(list_of_numbers, size=n_samples, replace=True)
    mask = np.zeros_like(prototypes, dtype=int)
    for i in range(n_samples):
        mask[i, :n_orn_active[i]] = 1  # set only this number of ORNs active
        np.random.shuffle(mask[i, :])
    out = np.multiply(prototypes, mask)
    return out


def _mask_orn_activation_column(prototypes, probs):
    '''
    :param prototypes:
    :param spread:  varies from [0, 1). 0 means no spread, 1 means maximum spread.
    :return:
    '''
    n_samples = prototypes.shape[0]
    n_orn = prototypes.shape[1]
    mask = np.zeros_like(prototypes)
    for i in range(n_orn):
        mask[:,i] = np.random.uniform(0, 1, n_samples) < probs[i]
    out = np.multiply(prototypes, mask)
    return out


def _relabel(train_labels, val_labels, n_pre, n_post, rng=None, random=False):
    """Relabeling classes.

    Randomly relabel n_pre classes to n_post classes, assuming n_post<n_pre

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
    if random:
        if rng is None:
            rng = np.random.RandomState()
        # Generate the mapping from previous labels to new labels
        labelmap = rng.choice(range(n_post), size=(n_pre))
    else:
        if not (n_pre/n_post).is_integer():
            print('n_pre/n_post is not an integer, making uneven classes')
        labelmap = np.tile(np.arange(n_post), int(np.ceil(n_pre/n_post)))
        labelmap = labelmap[:n_pre]

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


def _normalize(x):
    norm = np.linalg.norm(x, axis=1)
    x = (x.T / norm).T
    x[np.isnan(x)] = 0
    return x


def _sample_input(n_sample, dim, rng, corr=None):
    """Sample inputs, default uniform.

    For generating multi-variate random variables with uniform (0, 1) marginal
    and specified correlation, see for references:
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.281&rep=rep1&type=pdf
    https://stats.stackexchange.com/questions/66610/
    generate-pairs-of-random-numbers-uniformly-distributed-and-correlated

    Args:
        corr: if not None, correlation of multi-dimensional random variables

    Return:
        Y: numpy array, (n_sample, dim)
    """
    if corr is not None:
        mean = np.zeros(dim)
        cov = np.ones((dim, dim)) * 2 * np.sin(corr * np.pi / 6)
        np.fill_diagonal(cov, 1)
        Y = rng.multivariate_normal(mean, cov, n_sample)
        Y = stats.norm.cdf(Y)
    else:
        Y = rng.uniform(0, 1, (n_sample, dim))
    return Y


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
        is_spread_orn_activity,
        spread_orn_activity,
        mask_orn_activation_row,
        mask_orn_activation_column,
        n_combinatorial_classes=None,
        combinatorial_density=None,
        n_class_valence=None,
        n_proto_valence=None,
        has_special_odors=False,
        special_odor_activation=0,
        n_or_per_orn=1,
        orn_corr=None,
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
        n_proto_valence: int, the number of valence class
        orn_corr: None or float between 0 or 1, the correlation between
            activity of different ORNs
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
    if percent_generalization < 100:
        n_proto -= 1

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
        # special_odor_activation = 5.
        n_neutral_odor = n_proto - (n_good_odor + n_bad_odor)
        prototypes_neutral = rng.uniform(0, max_activation, (n_neutral_odor, n_orn))
        prototypes_good = np.zeros((n_good_odor, n_orn))
        prototypes_good[range(n_good_odor), range(n_good_odor)] = special_odor_activation
        prototypes_bad = np.zeros((n_bad_odor, n_orn))
        prototypes_bad[range(n_bad_odor), range(n_good_odor, n_good_odor+n_bad_odor)] = special_odor_activation
        prototypes = np.concatenate((prototypes_neutral, prototypes_good, prototypes_bad), axis=0)

        train_odors_neutral = rng.uniform(0, max_activation, (n_train_neutral, n_orn))
        ind = rng.randint(n_good_odor, size=(n_train_good))
        # TODO(gryang): This should be changed
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
        if orn_corr is not None:
            raise ValueError('orn_corr not None not supported for multi_head')
    else:
        prototypes = _sample_input(n_proto, n_orn, rng=rng, corr=orn_corr)
        train_odors = _sample_input(n_train, n_orn, rng=rng, corr=orn_corr)
        val_odors = _sample_input(n_val, n_orn, rng=rng, corr=orn_corr)

        prototypes *= max_activation
        train_odors *= max_activation
        val_odors *= max_activation

    if n_proto == n_train:
        train_odors = prototypes

    if mask_orn_activation_row[0]:
        print('mask_row')
        mask_degree = mask_orn_activation_row[1]
        prototypes = _mask_orn_activation_row(prototypes, spread=mask_degree)
        train_odors = _mask_orn_activation_row(train_odors, spread=mask_degree)
        val_odors = _mask_orn_activation_row(val_odors, spread=mask_degree)

    if mask_orn_activation_column[0]:
        print('mask_col')
        spread = mask_orn_activation_column[1]
        assert spread >= 0 and spread < 1, 'spread is not between the values of [0,1)'
        mask_degree = (1 - spread) / 2
        low, high = mask_degree, 1 - mask_degree
        low_samples = np.random.uniform(0, low, n_orn)
        high_samples = np.random.uniform(high, 1, n_orn)
        samples = np.concatenate((low_samples, high_samples))
        probs = np.random.choice(samples, size=n_orn, replace=False)

        prototypes = _mask_orn_activation_column(prototypes, probs)
        train_odors = _mask_orn_activation_column(train_odors, probs)
        val_odors = _mask_orn_activation_column(val_odors, probs)

    if is_spread_orn_activity:
        print('mean')
        spread = spread_orn_activity
        prototypes = _spread_orn_activity(prototypes, spread)
        train_odors = _spread_orn_activity(train_odors, spread)
        val_odors = _spread_orn_activity(val_odors, spread)

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
        print('concentration')
        # normalize prototypes and train/val_odors_forlabels to unit vectors
        prototypes = _normalize(prototypes)
        train_odors_forlabels = _normalize(train_odors_forlabels)
        val_odors_forlabels = _normalize(val_odors_forlabels)

    train_labels = _get_labels(prototypes, train_odors_forlabels, percent_generalization)
    val_labels = _get_labels(prototypes, val_odors_forlabels, percent_generalization)

    #make label distribution more uniform
    sculpt = False
    if sculpt:
        cutoff = 8 * (1 / n_proto)
        weights = np.ones(n_proto)
        i = 0
        while True:
            print(i)
            i+=1
            hist = np.histogram(train_labels, bins=n_proto, density=True)[0]
            has_greater = np.max(hist) > cutoff
            if has_greater:
                ix = np.argmax(hist)
                weights[ix] *= 1.1
                train_labels = _get_labels(prototypes, train_odors_forlabels, percent_generalization, weights)
            else:
                break
        print(weights)
        val_labels = _get_labels(prototypes, val_odors_forlabels, percent_generalization, weights)

    if shuffle_label:
        # Shuffle the labels
        rng.shuffle(train_labels)
        rng.shuffle(val_labels)

    if relabel:
        print('relabeling ' + str(n_proto) + ' classes into ' + str(n_class))
        train_labels, val_labels = _relabel(
            train_labels, val_labels, n_proto, n_class, rng)

    assert train_odors.dtype == np.float32

    # Convert labels
    if label_type == 'combinatorial':
        key = _generate_combinatorial_label(
            n_class, n_combinatorial_classes,
            combinatorial_density, rng)
        train_labels = _convert_to_combinatorial_label(train_labels, key)
        val_labels = _convert_to_combinatorial_label(val_labels, key)

        plt.imshow(key)
        plt.show()
    elif label_type == 'one_hot':
        train_labels = _convert_one_hot_label(train_labels, n_class)
        val_labels = _convert_one_hot_label(val_labels, n_class)
    elif label_type == 'sparse':
        pass
    elif label_type == 'multi_head_sparse':
        if not has_special_odors:
            # labels 0-4 will be good, 5-9 will be bad, others will be neutral
            print('no special odors')
            # good_ix, bad_ix = 5, 10
            good_ix, bad_ix = 50, 100
            train_labels_valence = np.zeros_like(train_labels)
            train_labels_valence[(0<=train_labels)*(train_labels< good_ix)] = 1
            train_labels_valence[(good_ix <= train_labels) * (train_labels < bad_ix)] = 2
            val_labels_valence = np.zeros_like(val_labels)
            val_labels_valence[(0 <= val_labels) * (val_labels < good_ix)] = 1
            val_labels_valence[(good_ix <= val_labels) * (val_labels < bad_ix)] = 2
            #
            # innate_generalization = 100
            # prototypes_valence = rng.uniform(0, max_activation, (n_proto_valence-1, n_orn))
            # train_labels_valence = _get_labels(prototypes_valence, train_odors_forlabels, innate_generalization)
            # val_labels_valence = _get_labels(prototypes_valence, val_odors_forlabels, innate_generalization)

        train_labels = np.stack([train_labels, train_labels_valence]).T
        val_labels = np.stack([val_labels, val_labels_valence]).T
    else:
        raise ValueError('Unknown label type: ', str(label_type))

    debug = False
    if debug:
        plt.hist(np.sum(train_odors, axis=1), density=True)
        plt.show()
        plt.hist(train_labels, bins= n_proto, density=True)
        plt.show()

    if n_or_per_orn > 1:
        # mix_or_per_orn_mode = 'random'
        mix_or_per_orn_mode = 'circulant'
        if mix_or_per_orn_mode == 'random':
            # Randoml mix OR per ORN
            mask = np.zeros((n_orn, n_orn))
            mask[:n_or_per_orn] = 1./n_or_per_orn
            for i in range(n_orn):
                np.random.shuffle(mask[:, i])  # shuffling in-place
        else:
            from scipy.linalg import circulant
            tmp = np.zeros(n_orn)
            tmp[:n_or_per_orn] = 1./n_or_per_orn
            mask = circulant(tmp)

        train_odors = np.dot(train_odors, mask)
        val_odors = np.dot(val_odors, mask)
        prototypes = np.dot(prototypes, mask)

    return train_odors, train_labels, val_odors, val_labels, prototypes


def _gen_folder_name(config, seed):
    """Automatically generate folder name."""
    auto_folder_name = '_s' + str(seed)
    return auto_folder_name


def save_proto(config=None, seed=0, folder_name=None):
    """Save dataset in numpy format."""

    if config is None:
        config = input_ProtoConfig()

    # make and save data
    train_x, train_y, val_x, val_y, prototypes = _generate_proto_threshold(
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
        is_spread_orn_activity=config.is_spread_orn_activity,
        spread_orn_activity=config.spread_orn_activity,
        mask_orn_activation_row=config.mask_orn_activation_row,
        mask_orn_activation_column=config.mask_orn_activation_column,
        n_combinatorial_classes=config.n_combinatorial_classes,
        combinatorial_density=config.combinatorial_density,
        n_class_valence=config.n_class_valence,
        n_proto_valence=config.n_proto_valence,
        has_special_odors=config.has_special_odors,
        special_odor_activation=config.special_odor_activation,
        n_or_per_orn=config.n_or_per_orn,
        orn_corr=config.orn_corr,
        seed=seed)

    if folder_name is None:
        folder_name = _gen_folder_name(config, seed)

    folder_path = os.path.join(config.path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)

    vars = [train_x.astype(np.float32), train_y.astype(np.int32),
            val_x.astype(np.float32), val_y.astype(np.int32),
            prototypes.astype(np.float32)]
    varnames = ['train_x', 'train_y', 'val_x', 'val_y', 'prototype']
    for result, name in zip(vars, varnames):
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


def save_autoencode(config=None, seed=0, folder_name=None):
    """Save dataset in numpy format."""

    if config is None:
        config = InputAutoEncode()

    # make and save data
    rng = np.random.RandomState(seed)

    prototypes = (rng.rand(config.n_class, config.n_orn) < config.proto_density).astype(np.float32)

    train_ind = rng.choice(np.arange(config.n_class), size=(config.n_train,))
    train_x = prototypes[train_ind]
    train_y = prototypes[train_ind]
    # flip the matrix element if the corresponding element in flip_matrix is 1
    flip_matrix = rng.rand(*train_x.shape) < config.p_flip
    train_x = abs(flip_matrix - train_x)

    val_ind = rng.choice(np.arange(config.n_class), size=(config.n_val,))
    val_x = prototypes[val_ind]
    val_y = prototypes[val_ind]
    flip_matrix = rng.rand(*val_x.shape) < config.p_flip
    val_x = abs(flip_matrix - val_x)

    folder_path = os.path.join(config.path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.makedirs(folder_path)

    vars = [train_x, train_y, val_x, val_y, prototypes]
    varnames = ['train_x', 'train_y', 'val_x', 'val_y', 'prototype']
    for result, name in zip(vars, varnames):
        np.save(os.path.join(folder_path, name), result)

    #save parameters
    tools.save_config(config, folder_path)
    return folder_path


def load_data(data_dir):
    """Load dataset."""
    if not os.path.exists(data_dir):
        # datasets are usually stored like path/datasets/proto/name
        paths = ['.'] + os.path.normpath(data_dir).split(os.path.sep)[-3:]
        data_dir = os.path.join(*paths)

    def _load_proto(path):
        """Load dataset from numpy format."""
        names = ['train_x', 'train_y', 'val_x', 'val_y']
        return [np.load(os.path.join(path, name + '.npy')) for name in names]

    train_x, train_y, val_x, val_y = _load_proto(data_dir)
    return train_x, train_y, val_x, val_y


if __name__ == '__main__':
    pass
