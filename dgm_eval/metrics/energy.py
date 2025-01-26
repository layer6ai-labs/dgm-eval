import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

Array = jnp.ndarray

def euclidian_distance(x, y):
    return np.linalg.norm(x - y, axis=-1)


def prepare_i_j_pairs(x, y):
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


def compute_energy_with_reps_naive(reps1, reps2):
    """
    Params:
    -- reps1   : activations of a representative data set (usually train)
    -- reps2   : activations of generated data set
    Returns:
    --   : Some Sort Of Energy Distance.
    """

    return energy_distance_naive(reps1, reps2)

def energy_distance_naive(x, y, batch_size=1):
    """Compute energy distance between two distributions with batching."""
    # Sum of distances between x and y
    d_xy = batch_pairwise_distance(x, y, batch_size)

    # Sum of distances within x
    d_xx = batch_pairwise_distance(x, x, batch_size)

    # Sum of distances within y
    d_yy = batch_pairwise_distance(y, y, batch_size)

    # Energy distance formula
    return (2 * d_xy - d_xx - d_yy) / (2 * d_xy)

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

@jax.jit  # Tell JAX that batch_size is static
def batch_pairwise_distance_jax(x, y):
    """Compute pairwise distances between x and y in batches."""
    n_x = x.shape[0]
    sum_dist = 0.0

    # Replace the Python loop with JAX-compatible logic
    def loop_body(i, acc):
        # Use lax.dynamic_slice for dynamic slicing
        x_batch = lax.dynamic_slice(x, (i, 0), (1, x.shape[1]))  # Slicing the batch
        distances = jnp.linalg.norm(x_batch[:, jnp.newaxis, :] - y, axis=-1)
        return acc + jnp.sum(distances)

    sum_dist = jax.lax.fori_loop(0, n_x, lambda i, acc: loop_body(i, acc), sum_dist)
    return sum_dist


@jax.jit
def energy_distance_naive_jax(x, y, batch_size=100):
    """Compute energy distance between two distributions with batching."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    # Sum of distances between x and y
    d_xy = batch_pairwise_distance_jax(x, y)

    # Sum of distances within x
    d_xx = batch_pairwise_distance_jax(x, x)

    # Sum of distances within y
    d_yy = batch_pairwise_distance_jax(y, y)

    # Energy distance formula
    return (2 * d_xy - d_xx - d_yy) / (2 * d_xy)

@jax.jit
def compute_energy_with_reps_naive_jax(reps1, reps2):
    """
    Params:
    -- reps1   : activations of a representative data set (usually train)
    -- reps2   : activations of generated data set
    Returns:
    --   : Some Sort Of Energy Distance.
    """
    return energy_distance_naive_jax(reps1, reps2)

