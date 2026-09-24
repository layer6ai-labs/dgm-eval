import numpy as np

from dgm_eval.synthetics import SyntheticConfig

SEED = 42


def sample_gaussian(
    mu_p: np.ndarray,
    mu_q: np.ndarray,
    dim: int,
    n: int,
    rng: np.random.Generator = None,
    return_params: bool = False,
) -> tuple:
    """Sample from two Gaussian distributions P and Q.

    Args:
        mu_p: Mean of distribution P (real data)
        mu_q: Mean of distribution Q (generated data)
        dim: Dimensionality
        n: Number of samples per distribution
        rng: Random number generator (uses SEED if None)
        return_params: Whether to return the parameters as well

    Returns:
        (X, Y) or (X, Y, mu_p, mu_q) if return_params=True
        X: samples from P (n, dim)
        Y: samples from Q (n, dim)
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    dim = len(mu_p)
    X = rng.normal(loc=mu_p, scale=1.0, size=(n, dim))  # real
    Y = rng.normal(loc=mu_q, scale=1.0, size=(n, dim))  # generated

    return (X, Y, mu_p, mu_q) if return_params else (X, Y)


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


def generate_from_config(
    config: SyntheticConfig,
    return_labels: bool = False,
    rng: np.random.Generator = None,
) -> tuple:

    rng = np.random.default_rng(config.seed)

    if config.type == "gaussian":
        return sample_gaussian(
            mu_p=config.params["mu_p"],
            mu_q=config.params["mu_q"],
            dim=config.dim,
            n=config.n_samples,
            rng=rng,
        )

    elif config.type == "gmm":
        return sample_gmm(
            P_means=config.params["P_means"],
            Q_means=config.params["Q_means"],
            P_weights=config.params["P_weights"],
            Q_weights=config.params["Q_weights"],
            dim=config.dim,
            n=config.n_samples,
            return_labels=return_labels,
            rng=rng,
        )

    else:
        raise ValueError(f"Unknown config type: {config.type}")
