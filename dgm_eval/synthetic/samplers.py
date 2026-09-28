import numpy as np

from dgm_eval.synthetic import SyntheticConfig

SEED = 42


def sample_gaussian(
    mu_p: np.ndarray,
    mu_q: np.ndarray,
    dim: int,
    n: int,
    rng: np.random.Generator = None,
) -> tuple:

    if rng is None:
        rng = np.random.default_rng(SEED)

    dim = len(mu_p)
    X = rng.normal(loc=mu_p, scale=1.0, size=(n, dim))  # real
    Y = rng.normal(loc=mu_q, scale=1.0, size=(n, dim))  # generated

    return (X, Y)


def sample_gmm(
    P_means: list,
    Q_means: list,
    P_weights: list,
    Q_weights: list,
    dim: int,
    n: int,
    return_labels: bool = False,
    rng: np.random.Generator = None,
) -> tuple:

    if rng is None:
        rng = np.random.default_rng(SEED)

    def sample_mixture(means, weights):
        components = rng.choice(len(means), size=n, p=weights)
        samples = [rng.normal(loc=means[c], scale=1.0, size=dim) for c in components]
        return np.array(samples), components

    Xr, Yr = sample_mixture(P_means, P_weights)
    Xg, Yg = sample_mixture(Q_means, Q_weights)

    return (Xr, Xg, Yr, Yg) if return_labels else (Xr, Xg)


SAMPLERS = {
    "gaussian": sample_gaussian,
    "gmm": sample_gmm,
}


def sample_from_config(
    config: SyntheticConfig,
    return_labels: bool = False,
    rng: np.random.Generator = None,
) -> tuple:

    return SAMPLERS[config.type](
        **config.params,
        dim=config.dim,
        n=config.n_samples,
        rng=np.random.default_rng(config.seed) if rng is None else rng,
        **{"return_labels": True} if return_labels else {},
    )
