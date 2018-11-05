import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


PROTO_N_CLASS = 1000
PROTO_N_ORN = 50

def _generate_proto():
    """Activate all ORNs randomly."""
    N_CLASS = PROTO_N_CLASS
    N_ORN = PROTO_N_ORN

    prototypes = np.random.rand(N_CLASS, N_ORN).astype(np.float32)

    N_TRAIN = 1000000
    N_VAL = 9192
    train_odors = np.random.rand(N_TRAIN, N_ORN).astype(np.float32)
    val_odors = np.random.rand(N_VAL, N_ORN).astype(np.float32)

    def get_labels(odors):
        dist = euclidean_distances(prototypes, odors)
        return np.argmax(dist, axis=0).astype(np.int32)

    train_labels = get_labels(train_odors)
    val_labels = get_labels(val_odors)
    return train_odors, train_labels, val_odors, val_labels


def save_proto():
    train_odors, train_labels, val_odors, val_labels = _generate_proto()
    np.save('./datasets/proto/train_x', train_odors)
    np.save('./datasets/proto/train_y', train_labels)
    np.save('./datasets/proto/val_x', val_odors)
    np.save('./datasets/proto/val_y', val_labels)


def load_proto():
    train_odors = np.load('./datasets/proto/train_x.npy')
    train_labels = np.load('./datasets/proto/train_y.npy')
    val_odors = np.load('./datasets/proto/val_x.npy')
    val_labels = np.load('./datasets/proto/val_y.npy')
    return train_odors, train_labels, val_odors, val_labels


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
    # save_proto()
    train_odors, train_labels, val_odors, val_labels = load_proto()