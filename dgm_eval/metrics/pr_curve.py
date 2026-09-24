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

from dgm_eval.utils import strat_tts_optional_labs

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s.%(funcName)s: %(message)s",
    force=True,
)

LAMBDAS = np.tan(np.linspace(0, np.pi / 2, 152))[1:-1]

PR_CURVE_CLFS = ["cov", "ipr", "kde", "knn", "Cov", "iPR", "KDE", "kNN"]


# --------------------------------------------------------------------------------
# UTILS
# --------------------------------------------------------------------------------

WORKING_MEMORY = 512


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


def count_points_in_ball(
    query: np.ndarray,
    center: np.ndarray,
    radius: np.ndarray | float,
    working_memory: int = WORKING_MEMORY,
) -> np.ndarray:

    # scalar -> one radius per center, read-only view, no copy
    radius = np.broadcast_to(np.asarray(radius, dtype=np.float64), (center.shape[0],))

    out = np.zeros(center.shape[0], dtype=np.float64)
    offset = 0
    for chunk in pairwise_distances_chunked(
        center,
        query,
        working_memory=working_memory,
    ):
        r = radius[offset : offset + chunk.shape[0], None]
        out[offset : offset + chunk.shape[0]] = (chunk <= r).sum(axis=1)
        offset += chunk.shape[0]
    return out


def count_balls_containing(
    query: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    working_memory: int = WORKING_MEMORY,
) -> np.ndarray:

    assert centers.shape[0] == radii.shape[0]

    out = np.zeros(query.shape[0], dtype=np.float64)
    offset = 0
    for chunk in pairwise_distances_chunked(
        query,
        centers,
        working_memory=working_memory,
    ):
        out[offset : offset + chunk.shape[0]] = (chunk <= radii[None, :]).sum(axis=1)
        offset += chunk.shape[0]
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


def cov_score(
    z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    As per experiments to match Fig 16, authors seem to not exlude self.
    """
    # manual flag for testing purposes
    allow_exclusion = False

    z_to_X_knn_radii = kth_nn_radius(
        z,
        X,
        k,
        exclude_self=allow_exclusion and np.array_equal(z, X),
    )
    z_to_Y_knn_radii = kth_nn_radius(
        z,
        Y,
        k,
        exclude_self=allow_exclusion and np.array_equal(z, Y),
    )

    n_X_in_Y_ball = count_points_in_ball(X, z, z_to_Y_knn_radii)
    n_Y_in_X_ball = count_points_in_ball(Y, z, z_to_X_knn_radii)

    return n_X_in_Y_ball, n_Y_in_X_ball


def ipr_score(
    z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    As per experiments to match Fig 16, authors seem to not exlude self.
    """
    # manual flag for testing purposes
    allow_exclusion = False

    X_to_X_knn_radii = kth_nn_radius(X, X, k, exclude_self=allow_exclusion)
    Y_to_Y_knn_radii = kth_nn_radius(Y, Y, k, exclude_self=allow_exclusion)

    p_hat = count_balls_containing(z, X, X_to_X_knn_radii)
    q_hat = count_balls_containing(z, Y, Y_to_Y_knn_radii)

    return p_hat, q_hat


def kde_score(z, X, Y, k):
    # manual flag for testing purposes
    allow_exclusion = False

    X_to_X_knn_radii = kth_nn_radius(X, X, k, exclude_self=allow_exclusion)
    Y_to_Y_knn_radii = kth_nn_radius(Y, Y, k, exclude_self=allow_exclusion)

    bandwidth_X = np.mean(X_to_X_knn_radii)
    bandwidth_Y = np.mean(Y_to_Y_knn_radii)

    p_hat = count_points_in_ball(X, z, bandwidth_X)
    q_hat = count_points_in_ball(Y, z, bandwidth_Y)

    return p_hat, q_hat


def knn_score(
    z: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    As per experiments to match Fig 16, authors seem to not exlude self.
    """
    # manual flag for testing purposes
    allow_exclusion = False

    XY = np.vstack([X, Y])
    labels = np.array([0] * len(X) + [1] * len(Y))  # 0: X, 1: Y

    # exclude self
    self_query = np.array_equal(z, X) or np.array_equal(z, Y)
    n_neighbors = k + 1 if allow_exclusion and self_query else k

    # _, indices = kth_nearest_neighbours(z, XY, k)  # (|z|, k)

    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(XY)
    _, indices = nn.kneighbors(z)  # (|z|, k)

    if allow_exclusion and self_query:
        indices = indices[:, 1:]

    neighbour_labels = labels[indices]
    n_X = (neighbour_labels == 0).sum(axis=1).astype(float)
    n_Y = (neighbour_labels == 1).sum(axis=1).astype(float)

    return n_X, n_Y


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


SCORE = {
    "cov": cov_score,
    "ipr": ipr_score,
    "kde": kde_score,
    "parzen": kde_score,
    "knn": knn_score,
    "Cov": cov_score,
    "iPR": ipr_score,
    "KDE": kde_score,
    "kNN": knn_score,
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
    scores_X_vs_X, scores_X_vs_Y = SCORE[clf](X_test, X_train, Y_train, k)  # (N,)
    scores_Y_vs_X, scores_Y_vs_Y = SCORE[clf](Y_test, X_train, Y_train, k)  # (N,)

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
        clf: Classifier name ('cov', 'ipr', 'knn', 'parzen')
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

    if lambdas is None:
        lambdas = LAMBDAS

    n_real, n_fake = int(real_feats.shape[0]), int(fake_feats.shape[0])
    n_lambdas = len(lambdas)

    # Checks
    if derive_labelwise and (real_labs is None or fake_labs is None):
        raise ValueError("Arg `derive_labelwise` needs `real_labs` and `fake_labs`.")

    if nearest_k is None:
        nearest_k = int(np.sqrt(n_real))
        logger.warning(f"k is None. Setting it to sqrt of num samples: {nearest_k}")
    elif nearest_k == -1:
        nearest_k = int(np.sqrt(n_real))
        logger.info(f"k = -1. Setting it to sqrt of num samples: {nearest_k}")

    # Return NaN if insufficient samples
    if n_real < nearest_k + 1 or n_fake < nearest_k + 1:
        logger.warning(
            f"Insufficient samples (real: {n_real}, fake: {n_fake}, "
            f"need: {nearest_k + 1}). Returning NaN curve."
        )
        return nan_pr_curve(n_lambdas, n_real, n_fake, nearest_k, split)

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

    scores_real_vs_X, scores_real_vs_Y = SCORE[clf](
        Xr_test, Xr_train, Xg_train, nearest_k
    )  # (N_real,)
    scores_fake_vs_X, scores_fake_vs_Y = SCORE[clf](
        Xg_test, Xr_train, Xg_train, nearest_k
    )  # (N_fake,)

    precisions, recalls = pr_curve_from_scores(
        scores_real_vs_X,
        scores_real_vs_Y,
        scores_fake_vs_X,
        scores_fake_vs_Y,
        lambdas,
    )

    d = {
        "P": precisions,
        "R": recalls,
        "n_real": n_real,
        "n_fake": n_fake,
        "param_k": nearest_k,
        "param_split": split,
    }

    if derive_labelwise:
        n_labels = np.max([Yr_train, Yr_test, Yg_train, Yg_test]) + 1

        for label_idx in range(n_labels):
            label_key = f"label-{label_idx}"
            logger.info(f"--- {label_key} (rcf) ---")

            # Masks
            Mr_k = Yr_test == label_idx
            Mg_k = Yg_test == label_idx
            Xr_test_k = Xr_test[Mr_k]
            Xg_test_k = Xg_test[Mg_k]

            # Number of real/fake
            Nr_k = np.sum(Mr_k)
            Ng_k = np.sum(Mg_k)

            # Return NaN if insufficient samples for this label
            if np.sum(Nr_k) < nearest_k + 1 or np.sum(Ng_k) < nearest_k + 1:
                logger.warning(
                    f"Insufficient samples (real: {Nr_k}, fake: {Ng_k}, "
                    f"need: {nearest_k + 1}). Returning NaN curve."
                )
                d[label_key] = nan_pr_curve(n_lambdas, Nr_k, Ng_k, nearest_k, split)
                continue

            alphas_k, betas_k = pr_curve_from_scores(
                scores_real_vs_X[Mr_k],
                scores_real_vs_Y[Mr_k],
                scores_fake_vs_X[Mg_k],
                scores_fake_vs_Y[Mg_k],
                lambdas,
            )

            d[label_key] = {
                "P": alphas_k,
                "R": betas_k,
                "n_real": Nr_k,
                "n_fake": Ng_k,
                "param_k": nearest_k,
                "param_split": split,
            }

    return d
