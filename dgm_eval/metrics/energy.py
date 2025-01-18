import numpy as np
from scipy import linalg
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

def euclidian_distance(x, y):
    return np.linalg.norm(x - y, axis=-1)


def prepare_i_j_pairs(x, y):
    """Prepares pairs with indexes as \sum_{i, j = 1}^n.

    \sum_{i, j = 1}^n = \sum_i^n \sum_j^n

    Example:
    \sum_{i, j = 1}^3 i+j = \sum_i^3 \sum_j^3 i + j =
    (1 + 1) + (1 + 2) + (1 + 3) +
    (2 + 1) + (2 + 2) + (2 + 3) +
    (3 + 1) + (3 + 2) + (3 + 3)

    This function creates exactly those pairs between two arrays.
    e.g. x = [1, 2, 3], y = [1, 2, 3].
    In x every value is repeated n_elements times,
    while y is repeated n_elements times as whole, so:
    x = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    y = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
    Then arbitrary operation between two arrays can be applied.
    Works for multidimensional arrays.
    """
    n_elements = x.shape[0]
    return np.repeat(x, n_elements, axis=0), np.tile(y, (n_elements, 1))

def energy_distance(x, y):
    return (
        (
            2 * np.sum(euclidian_distance(*prepare_i_j_pairs(x, y)))
            - np.sum(euclidian_distance(*prepare_i_j_pairs(x, x)))
            - np.sum(euclidian_distance(*prepare_i_j_pairs(y, y)))
        )
        / (2 * np.sum(euclidian_distance(*prepare_i_j_pairs(x, y))))
    )

def compute_energy_with_reps(reps1, reps2):
    #print(reps1, reps2)
    """
    Params:
    -- reps1   : activations of a representative data set (usually train)
    -- reps2   : activations of generated data set
    Returns:
    --   : Some Sort Of Energy Distance.
    """

    return energy_distance(reps1, reps2)


def pairwise_distances_sum(x, y):
    """Compute the sum of pairwise distances between two sets of vectors."""
    sum_dist = 0.0
    for i in range(x.shape[0]):
        sum_dist += np.sum(np.linalg.norm(x[i] - y, axis=-1))
    return sum_dist

'''
def energy_distance_naive(x, y):
    """Compute the energy distance between two distributions."""
    # Sum of distances between x and y
    d_xy = pairwise_distances_sum(x, y)

    # Sum of distances within x
    d_xx = pairwise_distances_sum(x, x)

    # Sum of distances within y
    d_yy = pairwise_distances_sum(y, y)

    # Energy distance formula
    return (2 * d_xy - d_xx - d_yy) / (2 * d_xy)
'''

def compute_energy_with_reps_naive(reps1, reps2):
    print(reps1, reps2)
    """
    Params:
    -- reps1   : activations of a representative data set (usually train)
    -- reps2   : activations of generated data set
    Returns:
    --   : Some Sort Of Energy Distance.
    """

    return energy_distance_naive(reps1, reps2)


def batch_pairwise_distance(x, y, batch_size):
    """Compute pairwise distances between x and y in batches."""
    n_x = x.shape[0]
    n_y = y.shape[0]
    sum_dist = 0.0

    for i in range(0, n_x, batch_size):
        x_batch = x[i:i + batch_size]
        # Compute pairwise distances for the current batch
        distances = np.linalg.norm(x_batch[:, np.newaxis, :] - y, axis=-1)
        sum_dist += np.sum(distances)

    return sum_dist


def energy_distance_naive(x, y, batch_size=100):
    """Compute energy distance between two distributions with batching."""
    # Sum of distances between x and y
    d_xy = batch_pairwise_distance(x, y, batch_size)

    # Sum of distances within x
    d_xx = batch_pairwise_distance(x, x, batch_size)

    # Sum of distances within y
    d_yy = batch_pairwise_distance(y, y, batch_size)

    # Energy distance formula
    return (2 * d_xy - d_xx - d_yy) / (2 * d_xy)


def incremental_pairwise_sum(x, y):
    """Incrementally compute the sum of pairwise distances between x and y."""
    sum_dist = 0.0
    for i in range(len(x)):
        dist = np.linalg.norm(x[i] - y, axis=1)  # Compute distances for one element
        sum_dist += np.sum(dist)
    return sum_dist

'''
def energy_distance_naive(x, y):
    """Compute energy distance incrementally to save memory."""
    # Sum of distances between x and y
    d_xy = incremental_pairwise_sum(x, y)

    # Sum of distances within x
    d_xx = incremental_pairwise_sum(x, x)

    # Sum of distances within y
    d_yy = incremental_pairwise_sum(y, y)

    # Energy distance formula
    return (2 * d_xy - d_xx - d_yy) / (2 * d_xy)
'''