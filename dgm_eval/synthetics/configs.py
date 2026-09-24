from dataclasses import dataclass, field
from typing import Any


@dataclass
class SyntheticConfig:
    """Configuration for generating synthetic data.

    Attributes:
        name: Human-readable name for this configuration
        type: Data generation type ("gaussian", "gmm")
        dim: Dimensionality of samples
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        params: Dict of type-specific parameters
    """

    name: str
    type: str  # "gaussian", "gmm"
    dim: int
    n_samples: int
    seed: int = 42
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate config consistency."""
        valid_types = {"gaussian", "gmm"}
        if self.type not in valid_types:
            raise ValueError(f"type must be one of {valid_types}, got {self.type}")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.type,
            "dim": self.dim,
            "n_samples": self.n_samples,
            "seed": self.seed,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SyntheticConfig":
        """Create config from dictionary."""
        return cls(
            name=d["name"],
            type=d["type"],
            dim=d["dim"],
            n_samples=d["n_samples"],
            seed=d.get("seed", 42),
            params=d.get("params", {}),
        )


@dataclass
class SyntheticDataset:
    """Container for synthetic data with metadata.

    Attributes:
        X: Real samples (n, dim)
        Y: Generated samples (n, dim)
        config: SyntheticConfig used to generate the data
        metadata: Optional dict with additional info (e.g., per-sample labels, class ids)
    """

    X: Any  # np.ndarray
    Y: Any  # np.ndarray
    config: SyntheticConfig
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate consistency."""
        if self.X.shape != self.Y.shape:
            raise ValueError(
                f"X and Y must have same shape, got {self.X.shape} vs {self.Y.shape}"
            )
        if self.X.shape[1] != self.config.dim:
            raise ValueError(
                f"Sample dimensionality {self.X.shape[1]} "
                f"doesn't match config.dim {self.config.dim}"
            )
