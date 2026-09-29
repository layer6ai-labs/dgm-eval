"""
Sykes 2024
https://arxiv.org/abs/2405.01611
"""

import logging

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.metrics import pairwise_distances_chunked
from sklearn.neighbors import NearestNeighbors

try:  # np.trapz was removed in NumPy 2.x; np.trapezoid exists since 2.0
    from numpy import trapezoid as _trapezoid
except ImportError:  # NumPy < 2.0
    from numpy import trapz as _trapezoid

from dgm_eval.utils import strat_tts_optional_labs

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s.%(funcName)s: %(message)s",
    force=True,
)

LAMBDAS = np.tan(np.linspace(0, np.pi / 2, 152))[1:-1]
PR_CURVE_CLFS = ["cov", "ipr", "kde", "knn", "Cov", "iPR", "KDE", "kNN"]

WORKING_MEMORY = 512

HANDLE_LABELS = ["filter_dataset", "filter_probes", "filter_centers"]


####################################################################################################
# UTILS
####################################################################################################


def resolve_k(k, n):
    if k is None:
        k = int(np.sqrt(n))
        logger.debug(f"k undefined. Setting it to sqrt of num samples: {k}")
    elif k == -1:
        k = int(np.sqrt(n))
        logger.debug(f"k = -1. Setting it to sqrt of num samples: {k}")
    return k


def kth_nn_radius(
    query: np.ndarray,
    points: np.ndarray,
    k: int,
    exclude_self: bool = False,
    working_memory: int = WORKING_MEMORY,
) -> np.ndarray:
    """Radius of the ball at query[i] reaching its k-th NN among points. -> (|query|,)"""

    k = k + 1 if exclude_self else k

    out = np.empty(query.shape[0], dtype=np.float64)
    offset = 0
    for chunk in pairwise_distances_chunked(
        query,
        points,
        working_memory=working_memory,
    ):
        # partition = O(n) selection of the k smallest, no full sort needed
        part = np.partition(chunk, k - 1, axis=1)[:, :k]
        out[offset : offset + chunk.shape[0]] = part.max(axis=1)
        offset += chunk.shape[0]

    return out


def count_points_in_probe_ball(
    points: np.ndarray,
    probes: np.ndarray,
    radii: np.ndarray | float,
    working_memory: int = WORKING_MEMORY,
) -> np.ndarray:

    radii = np.broadcast_to(np.asarray(radii, dtype=np.float64), (probes.shape[0],))

    out = np.zeros(probes.shape[0], dtype=np.float64)
    offset = 0
    for chunk in pairwise_distances_chunked(
        probes,
        points,
        working_memory=working_memory,
    ):
        r = radii[offset : offset + chunk.shape[0], None]
        out[offset : offset + chunk.shape[0]] = (chunk <= r).sum(axis=1)
        offset += chunk.shape[0]
    return out


def count_points_in_probe_ball_labelwise(
    probes,
    probes_labs,
    points,
    points_labs,
    radii,
):
    """
    Label-filtered counting (radii/bandwidths stay pooled, probes and centers are
    filtered by label).
    """
    radii = np.broadcast_to(np.asarray(radii, dtype=np.float64), (probes.shape[0],))
    out = np.zeros(probes.shape[0], dtype=np.float64)
    for y in np.unique(probes_labs):
        Mz = probes_labs == y
        Mp = points_labs == y
        if Mz.any() and Mp.any():
            out[Mz] = count_points_in_probe_ball(points[Mp], probes[Mz], radii[Mz])
    return out


def count_balls_containing_probe(
    probes: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    working_memory: int = WORKING_MEMORY,
) -> np.ndarray:

    assert centers.shape[0] == radii.shape[0]

    out = np.zeros(probes.shape[0], dtype=np.float64)
    offset = 0
    for chunk in pairwise_distances_chunked(
        probes,
        centers,
        working_memory=working_memory,
    ):
        out[offset : offset + chunk.shape[0]] = (chunk <= radii[None, :]).sum(axis=1)
        offset += chunk.shape[0]
    return out


def count_balls_containing_probes_labelwise(
    probes,
    probes_labs,
    centers,
    center_labs,
    radii,
):
    """
    Label-filtered counting (radii/bandwidths stay pooled, probes and centers are
    filtered by label).
    """
    out = np.zeros(probes.shape[0], dtype=np.float64)
    for y in np.unique(probes_labs):
        Mz = probes_labs == y
        Mc = center_labs == y
        if Mz.any() and Mc.any():
            out[Mz] = count_balls_containing_probe(probes[Mz], centers[Mc], radii[Mc])
    return out


def kth_nearest_neighbours(
    query: np.ndarray,
    points: np.ndarray,
    k: int,
    working_memory: int = WORKING_MEMORY,
):
    """k nearest points per query, sorted by distance. -> (distances, indices), each (|query|, k)"""
    distances = np.empty((query.shape[0], k), dtype=np.float64)
    indices = np.empty((query.shape[0], k), dtype=np.int64)
    offset = 0
    for chunk in pairwise_distances_chunked(
        query,
        points,
        working_memory=working_memory,
    ):
        part_idx = np.argpartition(chunk, k - 1, axis=1)[:, :k]
        part_dist = np.take_along_axis(chunk, part_idx, axis=1)
        order = np.argsort(part_dist, axis=1)
        n = chunk.shape[0]
        indices[offset : offset + n] = np.take_along_axis(part_idx, order, axis=1)
        distances[offset : offset + n] = np.take_along_axis(part_dist, order, axis=1)
        offset += n
    return distances, indices


# --------------------------------------------------------------------------------
# SCORES
# --------------------------------------------------------------------------------


def gaussian_score(
    z: np.ndarray,
    mu_p: np.ndarray,
    mu_q: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Bayes-optimal score q(z)/p(z).
    """
    dim = z.shape[1]

    p_z = multivariate_normal.pdf(z, mean=mu_p, cov=sigma * np.eye(dim))
    q_z = multivariate_normal.pdf(z, mean=mu_q, cov=sigma * np.eye(dim))

    return q_z / p_z


def log_likelihood_ratio_score_gmm(
    x: np.ndarray,
    means: np.ndarray,
    P_weights: np.ndarray,
    Q_weights: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Bayes-optimal score log(p(z)/q(z)) for the GMM setup with shared means and isotropic
    unit variance.
    """
    dim = x.shape[1]

    # ||x - μ_l·1_d||² = ||x||² - 2·μ_l·Σx_i + d·μ_l² (vectorised over l)

    # x: (n, d)
    # x_norm: (n,) the squared norm of each x (||x||²)
    # x_sum: (n,) the sum of each x (Σ_1^d x_i)
    # means: (K,) the means of each component
    # exponand: (n, K) the exponand for each x and each mean

    x_norm = np.sum(x**2, axis=1)
    x_sum = np.sum(x, axis=1)

    norm = (
        x_norm[:, None]
        - 2.0 * means[None, :] * x_sum[:, None]
        + dim * means[None, :] ** 2
    )
    exponand = -(1 / 2) / sigma**2 * norm

    log_p = logsumexp(exponand, b=P_weights, axis=1)
    log_q = logsumexp(exponand, b=Q_weights, axis=1)

    return log_p - log_q


def cov_score(z: np.ndarray, X: np.ndarray, Y: np.ndarray, k: int) -> np.ndarray:
    # manual flag for testing purposes
    allow_exclusion = False
    z_is_X = allow_exclusion and np.array_equal(z, X)
    z_is_Y = allow_exclusion and np.array_equal(z, Y)

    z_to_X_knn_radii = kth_nn_radius(z, X, k, exclude_self=z_is_X)
    z_to_Y_knn_radii = kth_nn_radius(z, Y, k, exclude_self=z_is_Y)

    n_X_in_Y_ball = count_points_in_probe_ball(X, z, z_to_Y_knn_radii)
    n_Y_in_X_ball = count_points_in_probe_ball(Y, z, z_to_X_knn_radii)

    return n_X_in_Y_ball, n_Y_in_X_ball


def cov_score_FC(z, Yz, Xr, Yr, Xg, Yg, k):
    z_to_real_radii = kth_nn_radius(z, Xr, k, exclude_self=False)
    z_to_fake_radii = kth_nn_radius(z, Xg, k, exclude_self=False)

    n_real_in_fake_ball = count_points_in_probe_ball_labelwise(
        z, Yz, Xr, Yr, z_to_fake_radii
    )
    n_fake_in_real_ball = count_points_in_probe_ball_labelwise(
        z,
        Yz,
        Xg,
        Yg,
        z_to_real_radii,
    )

    return n_real_in_fake_ball, n_fake_in_real_ball


def ipr_score(z: np.ndarray, X: np.ndarray, Y: np.ndarray, k: int) -> np.ndarray:
    # manual flag for testing purposes
    allow_exclusion = False

    X_to_X_knn_radii = kth_nn_radius(X, X, k, exclude_self=allow_exclusion)
    Y_to_Y_knn_radii = kth_nn_radius(Y, Y, k, exclude_self=allow_exclusion)

    p_hat = count_balls_containing_probe(z, X, X_to_X_knn_radii)
    q_hat = count_balls_containing_probe(z, Y, Y_to_Y_knn_radii)

    return p_hat, q_hat


def ipr_score_FC(z, Yz, Xr, Yr, Xg, Yg, k):
    real_to_real_knn_radii = kth_nn_radius(Xr, Xr, k, exclude_self=False)
    fake_to_fake_knn_radii = kth_nn_radius(Xg, Xg, k, exclude_self=False)

    p_hat = count_balls_containing_probes_labelwise(
        z,
        Yz,
        Xr,
        Yr,
        real_to_real_knn_radii,
    )
    q_hat = count_balls_containing_probes_labelwise(
        z,
        Yz,
        Xg,
        Yg,
        fake_to_fake_knn_radii,
    )

    return p_hat, q_hat


def kde_score(z, X, Y, k):
    # manual flag for testing purposes
    allow_exclusion = False

    X_to_X_knn_radii = kth_nn_radius(X, X, k, exclude_self=allow_exclusion)
    Y_to_Y_knn_radii = kth_nn_radius(Y, Y, k, exclude_self=allow_exclusion)

    bandwidth_X = np.mean(X_to_X_knn_radii)
    bandwidth_Y = np.mean(Y_to_Y_knn_radii)

    p_hat = count_points_in_probe_ball(X, z, bandwidth_X)
    q_hat = count_points_in_probe_ball(Y, z, bandwidth_Y)

    return p_hat, q_hat


def kde_score_FC(z, Yz, Xr, Yr, Xg, Yg, k):
    bandwidth_r = np.mean(kth_nn_radius(Xr, Xr, k, exclude_self=False))
    bandwidth_g = np.mean(kth_nn_radius(Xg, Xg, k, exclude_self=False))

    p_hat = count_points_in_probe_ball_labelwise(z, Yz, Xr, Yr, bandwidth_r)
    q_hat = count_points_in_probe_ball_labelwise(z, Yz, Xg, Yg, bandwidth_g)

    return p_hat, q_hat


def knn_score(z: np.ndarray, X: np.ndarray, Y: np.ndarray, k: int) -> np.ndarray:
    # manual flag for testing purposes
    allow_exclusion = False
    self_query = np.array_equal(z, X) or np.array_equal(z, Y)
    n_neighbors = k + 1 if allow_exclusion and self_query else k

    XY = np.vstack([X, Y])
    labels = np.array([0] * len(X) + [1] * len(Y))  # 0: X, 1: Y

    # _, indices = kth_nearest_neighbours(z, XY, k)  # (|z|, k)

    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(XY)
    _, indices = nn.kneighbors(z)  # (|z|, k)

    if allow_exclusion and self_query:
        indices = indices[:, 1:]

    neighbour_labels = labels[indices]
    n_X = (neighbour_labels == 0).sum(axis=1).astype(float)
    n_Y = (neighbour_labels == 1).sum(axis=1).astype(float)

    return n_X, n_Y


def knn_score_FC(z, Yz, Xr, Yr, Xg, Yg, k):
    X = np.vstack([Xr, Xg])
    Y = np.concatenate([Yr, Yg])
    src = np.concatenate(
        [
            np.zeros(Xr.shape[0], dtype=int),
            np.ones(Xg.shape[0], dtype=int),
        ]
    )

    nn = NearestNeighbors(n_neighbors=k).fit(X)
    _, indices = nn.kneighbors(z)  # (|z|, k)

    neigh_src = src[indices]  # (|z|, k)
    neigh_lab = Y[indices]  # (|z|, k)

    neigh_is_real = neigh_src == 0
    neigh_is_fake = neigh_src == 1
    match_label = neigh_lab == Yz[:, None]

    n_real_in_ball = (neigh_is_real & match_label).sum(axis=1).astype(float)
    n_fake_in_ball = (neigh_is_fake & match_label).sum(axis=1).astype(float)

    return n_real_in_ball, n_fake_in_ball


# # --------------------------------------------------------------------------------
# # Coverage
# # (Naem 2020)
# # https://arxiv.org/abs/2002.09797
# # --------------------------------------------------------------------------------


# def cov_score(z: np.ndarray, X: np.ndarray, Y: np.ndarray, k: int) -> np.ndarray:
#     """
#     For each probe z:
#       - compute radius r_Y(z) = dist to k-th NN of z in Y
#       - compute radius r_X(z) = dist to k-th NN of z in X
#       - score = #{x∈X inside B^Y_kNN(z)} / #{y∈Y inside B^X_kNN(z)}
#     """

#     nn_X = NearestNeighbors(n_neighbors=k).fit(X)
#     nn_Y = NearestNeighbors(n_neighbors=k).fit(Y)

#     dist_z_to_X, _ = nn_X.kneighbors(z)
#     dist_z_to_Y, _ = nn_Y.kneighbors(z)
#     r_X = dist_z_to_X[:, -1]  # dist from z to k-th neighbor in X
#     r_Y = dist_z_to_Y[:, -1]  # dist from z to k-th neighbor in Y

#     # distance from every x∈X to every z, and every y∈Y to every z
#     dist_X_to_z, _ = nn_X.kneighbors(z, n_neighbors=len(X))  # (|z|, |X|)
#     dist_Y_to_z, _ = nn_Y.kneighbors(z, n_neighbors=len(Y))  # (|z|, |Y|)

#     # #{x∈X : dist(x,z) <= r_Y(z)}
#     # #{y∈Y : dist(y,z) <= r_X(z)}
#     n_X_in_Y_ball = (dist_X_to_z <= r_Y[:, None]).sum(axis=1).astype(float)
#     n_Y_in_X_ball = (dist_Y_to_z <= r_X[:, None]).sum(axis=1).astype(float)

#     return n_X_in_Y_ball / (n_Y_in_X_ball + 1e-12)


# # --------------------------------------------------------------------------------
# # Improved PR (adapative bandwidth Kernel Density Estimator)
# # (Kynkaanniemi 2019)
# # https://arxiv.org/abs/1904.06991
# # --------------------------------------------------------------------------------


# # def ipr_score(z: np.ndarray, X: np.ndarray, Y: np.ndarray, k: int) -> np.ndarray:
# #     """
# #     For each probe point z, compute p̂(z) and q̂(z) via manifold indicator,
# #     then return the log-ratio score (monotone in p̂/q̂, avoids division by zero).
# #     """

# #     nn_X = NearestNeighbors(n_neighbors=k).fit(X)
# #     nn_Y = NearestNeighbors(n_neighbors=k).fit(Y)

# #     distances_X, _ = nn_X.kneighbors(X)
# #     distances_Y, _ = nn_Y.kneighbors(Y)
# #     knn_radii_X = distances_X[:, -1]  # dist from x to k-th neighbour in X
# #     knn_radii_Y = distances_Y[:, -1]  # dist from y to k-th neighbour in Y

# #     # distances from each z to every x / y
# #     dist_z_to_X, _ = nn_X.kneighbors(z, n_neighbors=len(X))  # (|z|, |X|)
# #     dist_z_to_Y, _ = nn_Y.kneighbors(z, n_neighbors=len(Y))  # (|z|, |Y|)

# #     p_hat = (dist_z_to_X <= knn_radii_X[None, :]).sum(axis=1).astype(float)
# #     q_hat = (dist_z_to_Y <= knn_radii_Y[None, :]).sum(axis=1).astype(float)

# #     return p_hat / (q_hat + 1e-12) # (|z|,)


# def ipr_score(
#     z: np.ndarray,  # (|z|, d)
#     X: np.ndarray,  # (|X|, d)
#     Y: np.ndarray,  # (|Y|, d)
#     k: int,
#     n_jobs: int = 8,
# ) -> np.ndarray:
#     """
#     For each probe point z, compute p̂(z) and q̂(z) via manifold indicator,
#     then return the score (monotone in p̂/q̂, avoids division by zero).
#     """

#     nn_X = NearestNeighbors(
#         n_neighbors=k + 1,  # do not count itself
#         algorithm="ball_tree",  # tree indexing
#         n_jobs=n_jobs,
#     ).fit(X)
#     nn_Y = NearestNeighbors(
#         n_neighbors=k + 1,  # do not count itself
#         algorithm="ball_tree",  # tree indexing
#         n_jobs=n_jobs,
#     ).fit(Y)

#     distances_X, _ = nn_X.kneighbors(X)
#     distances_Y, _ = nn_Y.kneighbors(Y)
#     knn_radii_X = distances_X[:, -1]  # dist from x to k-th neighbour in X, (|X|,)
#     knn_radii_Y = distances_Y[:, -1]  # dist from y to k-th neighbour in Y, (|Y|,)

#     # distances from each z to every x / y
#     # dist_z_to_X, _ = nn_X.kneighbors(z, n_neighbors=len(X))  # (|z|, |X|)
#     # dist_z_to_Y, _ = nn_Y.kneighbors(z, n_neighbors=len(Y))  # (|z|, |Y|)

#     # Replace kneighbors(n_neighbors=len(X)) with radius_neighbors for efficiency
#     # use global max radius for parallelism, then filter per point
#     dists_z_X, idx_z_X = nn_X.radius_neighbors(
#         z,
#         radius=knn_radii_X.max(),
#         return_distance=True,
#     )
#     dists_z_Y, idx_z_Y = nn_Y.radius_neighbors(
#         z,
#         radius=knn_radii_Y.max(),
#         return_distance=True,
#     )

#     # Count matches via per-point radius filtering
#     n_z = len(z)
#     p_hat = np.zeros(n_z, dtype=np.float64)  # (|z|,)
#     q_hat = np.zeros(n_z, dtype=np.float64)  # (|z|,)

#     for i in range(n_z):
#         # keep only x's where dist <= that x's own radius (the manifold indicator)
#         p_hat[i] = (dists_z_X[i] <= knn_radii_X[idx_z_X[i]]).sum()
#         q_hat[i] = (dists_z_Y[i] <= knn_radii_Y[idx_z_Y[i]]).sum()

#     return p_hat / (q_hat + 1e-12)  # (|z|,)

# # --------------------------------------------------------------------------------
# # KNN (KNN Classifier)
# # (Park & Kim 2023)
# # https://arxiv.org/abs/2309.01590
# # --------------------------------------------------------------------------------


# def knn_score(z: np.ndarray, X: np.ndarray, Y: np.ndarray, k: int) -> np.ndarray:
#     """
#     For each probe z, find k nearest neighbours in X∪Y.
#     Score = (# that come from X) - (# that come from Y).
#     High score → neighbourhood is mostly real → z is real-like.
#     """
#     XY = np.vstack([X, Y])
#     labels = np.array([0] * len(X) + [1] * len(Y))  # 0: X, 1: Y

#     # for each z, count how many of its k nearest neighbours are real vs fake
#     nn = NearestNeighbors(n_neighbors=k).fit(XY)
#     _, indices = nn.kneighbors(z)  # (|z|, k)

#     neighbour_labels = labels[indices]
#     n_X = (neighbour_labels == 0).sum(axis=1).astype(float)
#     n_Y = (neighbour_labels == 1).sum(axis=1).astype(float)

#     return n_X / (n_Y + 1e-12)


# # --------------------------------------------------------------------------------
# # Parzen (fixed bandwidth Kernel Density Estimator)
# # --------------------------------------------------------------------------------


# def kde_score(
#     z: np.ndarray,
#     X: np.ndarray,
#     Y: np.ndarray,
#     k: int,
# ) -> np.ndarray:
#     """
#     Fixed-bandwidth Parzen classifier.
#     Same as IPR but with a single global radius per dataset
#     instead of per-sample adaptive radii.
#     rho_X = mean kNN radius over X
#     rho_Y = mean kNN radius over Y
#     """

#     nn_X = NearestNeighbors(n_neighbors=k).fit(X)
#     nn_Y = NearestNeighbors(n_neighbors=k).fit(Y)

#     distances_X, _ = nn_X.kneighbors(X)
#     distances_Y, _ = nn_Y.kneighbors(Y)
#     knn_radii_X = distances_X[:, -1]  # dist from x to k-th neighbour in X
#     knn_radii_Y = distances_Y[:, -1]  # dist from y to k-th neighbour in Y

#     rho_X = np.mean(knn_radii_X)  # single global bandwidth for P
#     rho_Y = np.mean(knn_radii_Y)  # single global bandwidth for Q

#     # distances from each z to every x / y
#     dist_z_to_X, _ = nn_X.kneighbors(z, n_neighbors=len(X))  # (|z|, |X|)
#     dist_z_to_Y, _ = nn_Y.kneighbors(z, n_neighbors=len(Y))  # (|z|, |Y|)

#     p_hat = (dist_z_to_X <= rho_X).sum(axis=1).astype(float)  # (|z|,)
#     q_hat = (dist_z_to_Y <= rho_Y).sum(axis=1).astype(float)  # (|z|,)

#     return p_hat / (q_hat + 1e-12)


SCORES = {
    "cov": cov_score,
    "ipr": ipr_score,
    "kde": kde_score,
    "knn": knn_score,
    "Cov": cov_score,
    "iPR": ipr_score,
    "KDE": kde_score,
    "kNN": knn_score,
}
SCORES_FC = {
    "cov": cov_score_FC,
    "ipr": ipr_score_FC,
    "kde": kde_score_FC,
    "knn": knn_score_FC,
    "Cov": cov_score_FC,
    "iPR": ipr_score_FC,
    "KDE": kde_score_FC,
    "kNN": knn_score_FC,
}


# --------------------------------------------------------------------------------
# PR CURVES
# --------------------------------------------------------------------------------


def classification(score_vs_X: np.ndarray, score_vs_Y: np.ndarray, gamma: float):
    if gamma == np.inf:
        # always real: FPR=0, FNR=1
        return np.ones(score_vs_X.shape[0], dtype=bool)
    elif gamma == 0.0:
        # never real: FPR=1, FNR=0
        return np.zeros(score_vs_X.shape[0], dtype=bool)
    elif gamma >= 1.0:
        return (gamma * score_vs_X) >= score_vs_Y
    elif gamma < 1.0:
        return (gamma * score_vs_X) > score_vs_Y
    else:
        raise ValueError(f"gamma={gamma} is not a valid threshold.")


def pr_curve_from_scores(
    scores_X_vs_X: np.ndarray,
    scores_X_vs_Y: np.ndarray,
    scores_Y_vs_X: np.ndarray,
    scores_Y_vs_Y: np.ndarray,
    lambdas: np.ndarray,
):
    """
    Shared core: risk-minimizing precision/recall curve from classifier scores.

    `scores_*_vs_X` / `scores_*_vs_Y` are the two arrays returned by
    `SCORE[clf](z, X, Y, k)` for a query set z (score of z against X, score of z
    against Y). Used with z=X and z=Y respectively.

    Returns:
        alphas, betas: precision/recall, each of length len(lambdas) + 2
        (the extra 2 entries are the gamma=0 / gamma=inf boundary points).
    """
    gammas = np.concatenate([[0.0], lambdas, [np.inf]])
    fpr = np.array(
        [1.0 - classification(scores_X_vs_X, scores_X_vs_Y, g).mean() for g in gammas]
    )
    fnr = np.array(
        [classification(scores_Y_vs_X, scores_Y_vs_Y, g).mean() for g in gammas]
    )

    # Precision/Recall
    risk = lambdas[:, None] * fpr[None, :] + fnr[None, :]  # (L, L)
    min_risk = risk.min(axis=1)  # (L,)

    # recall as lambda -> 0: min fpr among gammas achieving fnr == 0 (never empty: gamma=0 qualifies)
    zero_fnr = fnr == 0
    beta_at_0 = fpr[zero_fnr].min()

    # precision as lambda -> inf: min fnr among gammas achieving fpr == 0 (never empty: gamma=inf qualifies)
    zero_fpr = fpr == 0
    alpha_at_inf = fnr[zero_fpr].min()

    alphas = np.concatenate([[0.0], min_risk, [alpha_at_inf]])
    betas = np.concatenate([[beta_at_0], min_risk / lambdas, [0.0]])

    return alphas, betas


def pr_curve(
    X: np.ndarray,
    Y: np.ndarray,
    lambdas: np.ndarray,
    clf: str,
    k: int,
    split: bool = False,
):
    if split:
        X_train, X_test = np.array_split(X, 2)
        Y_train, Y_test = np.array_split(Y, 2)
    else:
        X_train, X_test = X, X
        Y_train, Y_test = Y, Y

    if k == -1:
        k = int(np.sqrt(len(X_train)))
        print(f"Using k = sqrt(n) for {clf} (n={len(X_train)}, k={k})")

    # Matching scores
    scores_X_vs_X, scores_X_vs_Y = SCORES[clf](X_test, X_train, Y_train, k)  # (N,)
    scores_Y_vs_X, scores_Y_vs_Y = SCORES[clf](Y_test, X_train, Y_train, k)  # (N,)

    return pr_curve_from_scores(
        scores_X_vs_X, scores_X_vs_Y, scores_Y_vs_X, scores_Y_vs_Y, lambdas
    )


# --------------------------------------------------------------------------------
# Compute handle for PR curve computation, optionally with per-label breakdown
# --------------------------------------------------------------------------------


def nan_pr_curve(
    n_lambdas: int, n_real: int, n_fake: int, param_k: int, split: bool
) -> dict:
    """NaN placeholder curve, shaped like a real pr_curve_from_scores output (+2 boundary points)."""
    return {
        "P": np.full(n_lambdas + 2, np.nan),
        "R": np.full(n_lambdas + 2, np.nan),
        "n_real": n_real,
        "n_fake": n_fake,
        "param_k": param_k,
        "param_split": split,
    }


def compute_pr_curve(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
    clf: str,
    nearest_k: int,
    split: bool = False,
    real_labs=None,
    fake_labs=None,
    derive_labelwise=False,
    lambdas: np.ndarray | None = None,
):
    """
    Compute PR curve.

    ** Labels **
    Optionally computes PR curves per-label.
    Labels are assumed to be integers in [0, n_labels-1].
    Ball computation is carried out once on pooled features,
    then per-label scores are derived.

    Uses the same risk-minimizing classification core as `pr_curve`, so results from
    the two functions agree given the same inputs.

    Args:
        real_feats: Real features (N_real, D)
        fake_feats: Fake features (N_fake, D)
        lambdas: Lambda thresholds for risk computation
        clf: Classifier name ('cov', 'ipr', 'kde', 'knn')
        nearest_k: Number of neighbors. None or -1 sets k = sqrt(n_real)
        real_labs: Real sample labels, shape (N_real,). Required if derive_labelwise=True
        fake_labs: Fake sample labels, shape (N_fake,). Required if derive_labelwise=True
        derive_labelwise: If True, compute basis once and derive per-label metrics
        split: If True, use train/test halves of feats for fitting the classifier and evaluating

    Returns:
        dict with keys:
            - 'P': overall precisions, length len(lambdas) + 2
            - 'R': overall recalls, length len(lambdas) + 2
            - 'label-i': per-label dict (if derive_labelwise=True)
    """

    ### Check inputs ###

    if clf not in SCORES:
        raise ValueError(f"Clf {clf} not supported. Choose from {list(SCORES.keys())}.")

    if derive_labelwise and (real_labs is None or fake_labs is None):
        raise ValueError("Arg `derive_labelwise` needs `real_labs` and `fake_labs`.")

    if lambdas is None:
        lambdas = LAMBDAS

    ### Integrate inputs ###

    Nr, Ng = int(real_feats.shape[0]), int(fake_feats.shape[0])
    n_lambdas = len(lambdas)

    if nearest_k is None:
        nearest_k = int(np.sqrt(Nr))
        logger.warning(f"k is None. Setting it to sqrt of num samples: {nearest_k}")
    elif nearest_k == -1:
        nearest_k = int(np.sqrt(Nr))
        logger.info(f"k = -1. Setting it to sqrt of num samples: {nearest_k}")

    if Nr < nearest_k + 1 or Ng < nearest_k + 1:
        logger.warning(
            f"Insufficient samples (real: {Nr}, fake: {Ng}, "
            f"need: {nearest_k + 1}). Returning NaN curve."
        )
        return nan_pr_curve(n_lambdas, Nr, Ng, nearest_k, split)

    if split:
        Xr_train, Xr_test, Yr_train, Yr_test = strat_tts_optional_labs(
            real_feats,
            real_labs,
        )
        Xg_train, Xg_test, Yg_train, Yg_test = strat_tts_optional_labs(
            fake_feats,
            fake_labs,
        )
    else:
        Xr_train, Xr_test = real_feats, real_feats
        Yr_train, Yr_test = real_labs, real_labs
        Xg_train, Xg_test = fake_feats, fake_feats
        Yg_train, Yg_test = fake_labs, fake_labs

    ### Compute metric ###

    scores_real_vs_X, scores_real_vs_Y = SCORES[clf](
        Xr_test,
        Xr_train,
        Xg_train,
        nearest_k,
    )
    scores_fake_vs_X, scores_fake_vs_Y = SCORES[clf](
        Xg_test,
        Xr_train,
        Xg_train,
        nearest_k,
    )
    alphas, betas = pr_curve_from_scores(
        scores_real_vs_X,
        scores_real_vs_Y,
        scores_fake_vs_X,
        scores_fake_vs_Y,
        lambdas,
    )
    d = {
        "P": alphas,
        "R": betas,
        "n_real": Nr,
        "n_fake": Ng,
        "param_k": nearest_k,
        "param_split": split,
    }

    if derive_labelwise:  # Filter probes per label and compute metric
        n_labels = int(max(Yr_test.max(), Yg_test.max())) + 1

        for y in range(n_labels):
            label_key = f"label-{y}"
            logger.info(f"--- {label_key} (rcf) ---")

            Mr_y = Yr_test == y
            Mg_y = Yg_test == y

            Nr_y = int(np.sum(Mr_y))
            Ng_y = int(np.sum(Mg_y))

            if Nr_y < nearest_k + 1 or Ng_y < nearest_k + 1:
                logger.warning(
                    f"Insufficient samples (real: {Nr_y}, fake: {Ng_y}, "
                    f"need: {nearest_k + 1}). Returning NaN curve."
                )
                d[label_key] = nan_pr_curve(n_lambdas, Nr_y, Ng_y, nearest_k, split)
                continue

            alphas_y, betas_y = pr_curve_from_scores(
                scores_real_vs_X[Mr_y],
                scores_real_vs_Y[Mr_y],
                scores_fake_vs_X[Mg_y],
                scores_fake_vs_Y[Mg_y],
                lambdas,
            )

            d[label_key] = {
                "P": alphas_y,
                "R": betas_y,
                "n_real": Nr_y,
                "n_fake": Ng_y,
                "param_k": nearest_k,
                "param_split": split,
            }

    return d


def compute_pr_curve_cond(
    Xr,
    Xg,
    Yr: np.ndarray = None,
    Yg: np.ndarray = None,
    clf: str = "ipr",
    nearest_k: int | None = None,
    split: bool = True,
    handle_labels: str | None = None,
    lambdas: np.ndarray = None,
):
    """
    Compute the pooled PR curve and, if `handle_labels` is set, a per-label PR
    curve for each label.

    Args:
        Xr, Xg: pooled real / fake features, shape (N_real, dim) / (N_fake, dim)
        Yr, Yg: integer labels in [0, n_labels-1], shape (N_real,) / (N_fake,).
        clf: classifier name ('cov', 'ipr', 'kde', 'knn')
        nearest_k: number of neighbors. None or -1 sets k = sqrt(n_real)
        split: if True, use train/test halves for fitting / evaluating
        handle_labels: None (no per-label computation) or one of HANDLE_LABELS
            'filter_dataset' (FD: unconditional estimator on label-y samples only),
            'filter_probes' (FP: unconditional classifier, probes masked by label),
            'filter_centers' (FC: pooled radii, centers masked by label)
        lambdas: lambda thresholds for risk computation

    Returns:
        dict with keys:
            - 'P', 'R': overall (unconditional, pooled) precision/recall curves
            - 'label-i': per-label dict, (only if `handle_labels` is not None)
    """

    ### Check inputs ###

    if clf not in SCORES:
        raise ValueError(f"Clf {clf} not supported. Choose from {list(SCORES.keys())}.")

    if handle_labels is not None and handle_labels not in HANDLE_LABELS:
        raise ValueError(f"handle_labels={handle_labels!r} must be in {HANDLE_LABELS}.")

    if handle_labels is not None and (Yr is None or Yg is None):
        raise ValueError(f"handle_labels={handle_labels!r} needs `Yr` and `Yg`.")

    if lambdas is None:
        lambdas = LAMBDAS

    ### Integrate inputs ###

    Nr, Ng = int(Xr.shape[0]), int(Xg.shape[0])
    n_lambdas = len(lambdas)

    k = resolve_k(nearest_k, Nr)

    # Return NaN if insufficient samples: FP/FC keep pooled radii, so this is
    # the only place they need k+1 samples (FD re-checks per label below).
    if Nr < k + 1 or Ng < k + 1:
        logger.warning(
            f"Insufficient samples (real: {Nr}, fake: {Ng}, "
            f"need: {k + 1}). Returning NaN curve."
        )
        return nan_pr_curve(n_lambdas, Nr, Ng, k, split)

    if split:
        Xr_train, Xr_test, Yr_train, Yr_test = strat_tts_optional_labs(Xr, Yr)
        Xg_train, Xg_test, Yg_train, Yg_test = strat_tts_optional_labs(Xg, Yg)
    else:
        Xr_train, Xr_test = Xr, Xr
        Yr_train, Yr_test = Yr, Yr
        Xg_train, Xg_test = Xg, Xg
        Yg_train, Yg_test = Yg, Yg

    ### Compute unconditional/pooled metric ###

    scores_real_vs_X, scores_real_vs_Y = SCORES[clf](
        Xr_test,
        Xr_train,
        Xg_train,
        k,
    )
    scores_fake_vs_X, scores_fake_vs_Y = SCORES[clf](
        Xg_test,
        Xr_train,
        Xg_train,
        k,
    )
    alphas, betas = pr_curve_from_scores(
        scores_real_vs_X,
        scores_real_vs_Y,
        scores_fake_vs_X,
        scores_fake_vs_Y,
        lambdas,
    )
    d = {
        "P": alphas,
        "R": betas,
        "n_real": Nr,
        "n_fake": Ng,
        "param_k": k,
        "param_split": split,
    }

    if handle_labels is None:
        return d

    ### Compute per-label metric ###

    Ny = int(max(Yr.max(), Yg.max())) + 1

    if handle_labels == "filter_dataset":
        # FD: scores are computed per label, inside the loop below.
        pass

    elif handle_labels == "filter_probes":
        # FP: reuse the pooled scores above, only the probes get masked by label below.
        real_vs_X, real_vs_Y = scores_real_vs_X, scores_real_vs_Y
        fake_vs_X, fake_vs_Y = scores_fake_vs_X, scores_fake_vs_Y

    elif handle_labels == "filter_centers":
        # FC: pooled radii/bandwidths/neighbourhoods, centers masked by label.
        real_vs_X, real_vs_Y = SCORES_FC[clf](
            Xr_test,
            Yr_test,
            Xr_train,
            Yr_train,
            Xg_train,
            Yg_train,
            k,
        )
        fake_vs_X, fake_vs_Y = SCORES_FC[clf](
            Xg_test,
            Yg_test,
            Xr_train,
            Yr_train,
            Xg_train,
            Yg_train,
            k,
        )

    else:
        raise ValueError(f"handle_labels={handle_labels!r} must be in {HANDLE_LABELS}.")

    for y in range(Ny):
        label_key = f"label-{y}"
        logger.debug(f"--- {label_key} ({handle_labels}) ---")

        Mr_test_y = Yr_test == y
        Mg_test_y = Yg_test == y
        Nr_y, Ng_y = int(np.sum(Mr_test_y)), int(np.sum(Mg_test_y))

        if handle_labels == "filter_dataset":
            # FD: the label-y dataset is a standalone unconditional problem,
            # so k and the k+1-samples check are resolved on it.
            Mr_train_y = Yr_train == y
            Mg_train_y = Yg_train == y
            k_y = resolve_k(nearest_k, int(np.sum(Yr == y)))

            n_min = min(np.sum(Mr_train_y), np.sum(Mg_train_y))
            if Nr_y == 0 or Ng_y == 0 or n_min < k_y + 1:
                logger.debug(
                    f"{label_key}: insufficient samples (real: {Nr_y}, fake: {Ng_y}, "
                    f"need: {k_y + 1} to fit). Returning NaN curve."
                )
                d[label_key] = nan_pr_curve(n_lambdas, Nr_y, Ng_y, k_y, split)
                continue

            real_vs_X_y, real_vs_Y_y = SCORES[clf](
                Xr_test[Mr_test_y],
                Xr_train[Mr_train_y],
                Xg_train[Mg_train_y],
                k_y,
            )
            fake_vs_X_y, fake_vs_Y_y = SCORES[clf](
                Xg_test[Mg_test_y],
                Xr_train[Mr_train_y],
                Xg_train[Mg_train_y],
                k_y,
            )
        else:
            # FP/FC: radii are pooled, so a label only needs probes to evaluate,
            # not k+1 samples of its own (unlike `pr_curve.compute_pr_curve`'s 'rcf').
            k_y = k
            if Nr_y == 0 or Ng_y == 0:
                logger.warning(
                    f"{label_key}: no probes (real: {Nr_y}, fake: {Ng_y}). Returning NaN curve."
                )
                d[label_key] = nan_pr_curve(n_lambdas, Nr_y, Ng_y, k_y, split)
                continue

            real_vs_X_y, real_vs_Y_y = real_vs_X[Mr_test_y], real_vs_Y[Mr_test_y]
            fake_vs_X_y, fake_vs_Y_y = fake_vs_X[Mg_test_y], fake_vs_Y[Mg_test_y]

        alphas_y, betas_y = pr_curve_from_scores(
            real_vs_X_y,
            real_vs_Y_y,
            fake_vs_X_y,
            fake_vs_Y_y,
            lambdas,
        )

        d[label_key] = {
            "P": alphas_y,
            "R": betas_y,
            "n_real": Nr_y,
            "n_fake": Ng_y,
            "param_k": k_y,
            "param_split": split,
        }

    return d


####################################################################################################
# PR curve distillation with scalar metrics
####################################################################################################


def check_alphas_betas(alphas, betas):
    alphas = np.asarray(alphas, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)

    if alphas.shape != betas.shape:
        raise ValueError(
            "alphas and betas must have same shape. "
            f"Got {alphas.shape} vs {betas.shape}."
        )

    if alphas.size == 0:
        raise ValueError("Empty curve.")

    if np.any(alphas < 0) or np.any(betas < 0):
        raise ValueError("alphas and betas must be non-negative.")

    if np.any(np.diff(alphas) < -1e-12):
        raise ValueError("alphas must be in non-decreasing order.")
    if np.any(np.diff(betas) > 1e-12):
        raise ValueError("betas must be in non-increasing order.")

    if abs(alphas[0]) > 1e-12:
        raise ValueError("alphas[0] must be 0 (alpha is 0 at lambda = 0).")
    if abs(betas[-1]) > 1e-12:
        raise ValueError("betas[-1] must be 0 (beta is 0 at lambda = inf).")

    return alphas, betas


def auc(alphas, betas):
    """
    When AUC=0, distributions have disjoint support.
    When AUC=1, distributions are identical.
    """
    alphas, betas = check_alphas_betas(alphas, betas)
    return float(_trapezoid(alphas[::-1], betas[::-1]))


def f_score(alphas, betas, b=8.0):
    """
    Proposed by Sajjadi 2018.
    """
    alphas, betas = check_alphas_betas(alphas, betas)

    b2 = float(b) ** 2
    num = b2 + 1.0

    valid = (alphas > 0) & (betas > 0)
    if not np.any(valid):
        return 0.0
    den = b2 / alphas[valid] + 1.0 / betas[valid]

    f = num / den

    return float(f.max())


def f_scores(alphas, betas, b=8.0):
    """
    When b is large, F and F_inv mainly capture mode dropping and mode invention.
    """
    return f_score(alphas, betas, b), f_score(alphas, betas, 1.0 / b)


def pr_median(alphas, betas, return_lambda=False):
    """
    Point (alpha_lbar, beta_lbar) on the curve such that the ray alpha = lbar * beta
    splits the area under the PR curve into two equal halves.

    The points are already ordered by increasing slope alpha/beta, so the region
    under the closed curve is the union of the triangles (O, p_i, p_{i+1}).
    The cumulative area is walked until half the total is reached; the crossing
    point is interpolated linearly on that segment (the area swept from O is
    linear along a segment). Ties give degenerate (zero-area) triangles only when
    p_i and p_{i+1} lie on the same ray, which is harmless.

    Returns (alpha_bar, beta_bar), plus lbar if return_lambda=True.
    """
    alphas, betas = check_alphas_betas(alphas, betas)

    # tri >= 0 given the ordering
    tri = 0.5 * (betas[:-1] * alphas[1:] - betas[1:] * alphas[:-1])
    tri = np.clip(tri, 0.0, None)  # guard against round-off
    total = tri.sum()
    if total <= 0:
        return (0.0, 0.0, 0.0) if return_lambda else (0.0, 0.0)

    half = total / 2.0
    cum = np.cumsum(tri)
    k = min(int(np.searchsorted(cum, half)), tri.size - 1)
    prev = cum[k - 1] if k > 0 else 0.0
    t = (half - prev) / tri[k] if tri[k] > 0 else 0.0
    alpha_bar = alphas[k] + t * (alphas[k + 1] - alphas[k])
    beta_bar = betas[k] + t * (betas[k + 1] - betas[k])

    if return_lambda:
        lbar = np.inf if beta_bar == 0 else alpha_bar / beta_bar
        return float(alpha_bar), float(beta_bar), float(lbar)

    return float(alpha_bar), float(beta_bar)


def precision_at_recall(alphas, betas, eps=0.05):
    """
    alpha@eps = max alpha such that (alpha, eps) is on the PRD curve, i.e. the
    precision read on the curve at recall beta = eps.

    Scans every segment [p_i, p_{i+1}] whose recall range contains eps and
    interpolates alpha there. A vertical segment located exactly at beta = eps
    (tie in betas) contributes its upper end, hence the max.
    Returns 0 if eps > beta_0 (max recall).
    """
    a, b = check_alphas_betas(alphas, betas)
    if eps > b[0]:
        return 0.0
    b0, b1, a0, a1 = b[:-1], b[1:], a[:-1], a[1:]
    hit = (b0 >= eps) & (b1 <= eps)  # betas are non-increasing
    vertical = hit & (b0 == b1)
    sloped = hit & ~vertical
    cands = []
    if np.any(vertical):
        cands.append(np.maximum(a0[vertical], a1[vertical]))
    if np.any(sloped):
        t = (b0[sloped] - eps) / (b0[sloped] - b1[sloped])
        cands.append(a0[sloped] + t * (a1[sloped] - a0[sloped]))
    return float(np.max(np.concatenate(cands)))


def recall_at_precision(alphas, betas, eps=0.05):
    """
    beta@eps = max beta such that (eps, beta) is on the PRD curve, i.e. the
    recall read on the curve at precision alpha = eps.
    By symmetry, reversing the arrays swaps the roles of alpha and beta while
    keeping the required ordering. Returns 0 if eps > alpha_inf (max precision).
    """
    return precision_at_recall(betas[::-1], alphas[::-1], eps)


def pr_at_eps(alphas, betas, eps=0.05):
    """Returns the pair (alpha@eps, beta@eps)."""
    return (
        precision_at_recall(alphas, betas, eps),
        recall_at_precision(alphas, betas, eps),
    )
