"""
Classes that build SyntheticConfig corresponding to different data distributions.

Use a Distributions class when fixing one distribution's shape and then
generating several configs off it (varying n_samples/seed/n_per_class, not
shape itself). Construct the Distribution object once,  then call .to_config()
repeatedly:
```
dist = GaussianDistributions(mu_p, mu_q)  # shape fixed once
for seed in seeds:
    configs[f"gaussian_seed={seed}"] = dist.to_config(
        f"gaussian_seed={seed}",
        n_samples= 1_000
        seed=seed,
    )
```
"""

from abc import ABC, abstractmethod

import numpy as np

from .configs import SyntheticConfig


class Distributions(ABC):
    """Abstract base class for distributions."""

    @abstractmethod
    def to_config(self, name: str, dim: int, n_samples: int, seed: int = 42):
        """Convert distribution to SyntheticConfig."""


class GaussianDistributions(Distributions):
    """Gaussian distribution with fixed means for P and Q."""

    def __init__(self, mu_p: np.ndarray, mu_q: np.ndarray):
        """
        Initialize Gaussian distribution.

        Args:
            mu_p: Mean of P (real data), shape (dim,)
            mu_q: Mean of Q (generated data), shape (dim,)
        """
        if len(mu_p) != len(mu_q):
            raise ValueError("mu_p and mu_q must have same dimensionality")
        self.mu_p = np.asarray(mu_p)
        self.mu_q = np.asarray(mu_q)

    @property
    def dim(self) -> int:
        return len(self.mu_p)

    def to_config(self, name: str, n_samples: int, seed: int = 42):
        """Convert to SyntheticConfig.

        Args:
            name: Name for the config
            n_samples: Number of samples to generate
            seed: Random seed

        Returns:
            SyntheticConfig
        """

        sc = SyntheticConfig(
            name=name,
            type="gaussian",
            dim=self.dim,
            n_samples=n_samples,
            seed=seed,
            params={
                "mu_p": self.mu_p,
                "mu_q": self.mu_q,
            },
        )

        return sc


class GMMDistributions(Distributions):
    """Gaussian Mixture Model distribution.

    Component means are scalars, broadcast across all `dim` coordinates
    (mirrors `ClassPairDistributions`) rather than full dim-vectors.
    """

    def __init__(
        self,
        P_means: list,
        Q_means: list,
        P_weights: list,
        Q_weights: list,
    ):
        self._check_init_args(P_means, Q_means, P_weights, Q_weights)

        self.P_means = np.asarray(P_means, dtype=float)
        self.Q_means = np.asarray(Q_means, dtype=float)

        self.P_weights = np.asarray(P_weights)
        self.Q_weights = np.asarray(Q_weights)

        self.num_components = len(self.P_means)

    def _check_init_args(self, P_means, Q_means, P_weights, Q_weights):
        if (
            len(P_means) != len(Q_means)
            or len(P_means) != len(P_weights)
            or len(Q_means) != len(Q_weights)
        ):
            raise ValueError(
                f"Number of weights must match number of means. "
                f"Got {len(P_means)} means for P, {len(P_weights)} weights for P, "
                f"got {len(Q_means)} means for Q, {len(Q_weights)} weights for Q."
            )

        if not np.isclose(P_weights.sum(), 1.0):
            raise ValueError(f"P_weights must sum to 1, got {P_weights.sum()}")
        if not np.isclose(Q_weights.sum(), 1.0):
            raise ValueError(f"Q_weights must sum to 1, got {Q_weights.sum()}")

    @property
    def dim(self) -> int:
        """Dimensionality is determined at to_config() time (means are scalars)."""
        return None

    def to_config(self, name: str, dim: int, n_samples: int, seed: int = 42):
        """Convert to SyntheticConfig.

        Args:
            name: Name for the config
            dim: Dimensionality of samples (each scalar mean is broadcast across this)
            n_samples: Number of samples to generate
            seed: Random seed

        Returns:
            SyntheticConfig
        """

        sc = SyntheticConfig(
            name=name,
            type="gmm",
            dim=dim,
            n_samples=n_samples,
            seed=seed,
            params={
                "P_means": self.P_means.tolist(),
                "Q_means": self.Q_means.tolist(),
                "P_weights": self.P_weights.tolist(),
                "Q_weights": self.Q_weights.tolist(),
            },
        )

        return sc
