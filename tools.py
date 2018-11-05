import numpy as np


def compute_glo_score(w_orn):
    """Compute the glomeruli score in numpy.

    This function returns the glomeruli score, a number between 0 and 1 that
    measures how close the connectivity is to glomeruli connectivity.

    For one glomeruli neuron, first we compute the average connections from
    each ORN group. Then we sort the absolute connection weights by ORNs.
    The glomeruli score is simply:
        (Max weight - Second max weight) / (Max weight + Second max weight)

    Args:
        w_orn: numpy array (n_neuron, n_orn). This matrix has to be organized
        in the following way: the n_neuron neurons are grouped into n_orn groups
        index 0, ..., n_neuron_per_orn - 1 is the 0-th group, and so on

    Return:
        avg_glo_score: scalar, average glomeruli score
        glo_scores: numpy array (n_orn,), all glomeruli scores
    """
    n_neuron, n_orn = w_orn.shape
    n_neuron_per_orn = n_neuron // n_orn
    w_orn_by_orn = np.reshape(w_orn, (n_orn, n_neuron_per_orn, n_orn))
    w_orn_by_orn = w_orn_by_orn.mean(axis=1)
    w_orn_by_orn = abs(w_orn_by_orn)

    glo_scores = list()
    for i in range(n_orn):
        w_tmp = w_orn_by_orn[:, i]  # all projections to the i-th PN neuron
        indsort = np.argsort(w_tmp)[::-1]
        w_max = w_tmp[indsort[0]]
        w_second = w_tmp[indsort[1]]
        glo_score = (w_max - w_second) / (w_max + w_second)
        glo_scores.append(glo_score)

    avg_glo_score = np.mean(glo_scores)
    return avg_glo_score, glo_scores