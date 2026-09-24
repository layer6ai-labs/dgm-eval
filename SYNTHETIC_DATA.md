# Synthetic Data Experiments Guide

This guide explains how to use the synthetic data infrastructure to run experiments.

## Overview

The synthetic data system separates concerns into:

1. **Configuration** (`dgm_eval/synthetics/configs.py`): `SyntheticConfig` / `SyntheticDataset` dataclasses
2. **Distributions** (`dgm_eval/synthetics/distributions.py`): objects that fix a distribution's shape and build configs from it
3. **Sampling** (`dgm_eval/synthetics/samplers.py`): sampling functions and `generate_from_config`
4. **Experiment Utilities** (`dgm_eval/experiment_utils.py`): helpers for running experiments at scale
5. **Experiments** (`xp/`): your experiment logic

Public API (`from dgm_eval.synthetics import ...`):

| Name | Kind | Purpose |
|---|---|---|
| `SyntheticConfig` | dataclass | Serializable description of a P/Q pair |
| `SyntheticDataset` | dataclass | Container for `X`, `Y`, `config`, `metadata` (validates shapes) |
| `GaussianDistributions` | distribution | Two isotropic Gaussians |
| `GMMDistributions` | distribution | Two isotropic Gaussian mixtures |
| `generate_from_config` | function | Sample `(X, Y)` from a config |
| `sample_gaussian`, `sample_gmm` | functions | Low-level samplers |

Two data types are supported: `"gaussian"` and `"gmm"`. Any other `type` raises a `ValueError` in `SyntheticConfig`.

## Quick Start

### 1. Generate Synthetic Data

```python
import numpy as np
from dgm_eval.synthetics import SyntheticConfig, generate_from_config

config = SyntheticConfig(
    name="my_experiment",
    type="gaussian",
    dim=32,
    n_samples=1000,
    seed=42,
    params={
        "mu_p": np.zeros(32),       # Real data mean
        "mu_q": np.ones(32) * 0.5,  # Generated data mean
    },
)

X, Y = generate_from_config(config)  # X ~ P (real), Y ~ Q (generated), both (n_samples, dim)
```

### 2. Compute Metrics

```python
from dgm_eval.metrics import compute_prdc, compute_pr_curve

prdc = compute_prdc(X, Y, nearest_k=10)
print(f"Precision: {prdc['P']:.4f}, Recall: {prdc['R']:.4f}")

curve = compute_pr_curve(X, Y, clf="kNN", nearest_k=4)
# curve["P"], curve["R"]: precision / recall along the curve
```

### 3. Run an Experiment

See the scripts in `xp/` (e.g. `xp/xp_bound_violation.py`, `xp/xp_label_noising.py`) for complete examples.

## Data Types

All distributions are isotropic with unit variance (`scale=1.0`).

### 1. Gaussian (`type="gaussian"`)

P = N(`mu_p`, I), Q = N(`mu_q`, I).

| param | shape | meaning |
|---|---|---|
| `mu_p` | `(dim,)` | mean of P (real) |
| `mu_q` | `(dim,)` | mean of Q (generated) |

The sample dimensionality is taken from `len(mu_p)`, so keep `config.dim` consistent with it.

Using the distribution object (`dim` is inferred from the means):

```python
from dgm_eval.synthetics import GaussianDistributions

dist = GaussianDistributions(mu_p=np.zeros(64), mu_q=np.ones(64))
config = dist.to_config("my_config", n_samples=5000, seed=42)
```

### 2. Gaussian Mixture Model (`type="gmm"`)

P and Q are mixtures of K unit-variance Gaussians, each with its own component means and mixing weights.

| param | shape | meaning |
|---|---|---|
| `P_means` | `(K,)` | component means of P |
| `Q_means` | `(K,)` | component means of Q |
| `P_weights` | `(K,)` | mixing weights of P (sum to 1) |
| `Q_weights` | `(K,)` | mixing weights of Q (sum to 1) |

Component means are **scalars broadcast across all `dim` coordinates**: component `c` is N(`means[c]` · 1, I). The separation between two components is therefore `|m_i - m_j| · sqrt(dim)` in Euclidean distance.

```python
from dgm_eval.synthetics import GMMDistributions

K = 3
means = np.arange(K) * 2.0
weights = np.full(K, 1 / K)

dist = GMMDistributions(
    P_means=means,
    Q_means=means + 0.5,              # shift every Q component
    P_weights=np.array([0.5, 0.3, 0.2]),
    Q_weights=np.array([0.2, 0.3, 0.5]),
)
config = dist.to_config("gmm_config", dim=2, n_samples=5000, seed=42)
```

`GMMDistributions.dim` is `None`: dimensionality is only set when calling `to_config(..., dim=...)`. Pass the weights as NumPy arrays; the sum-to-one check calls `.sum()` on them.

#### Component labels

For GMMs, `generate_from_config` can also return the component index of each sample, which is useful for class-conditional metrics:

```python
X, Y, labels_X, labels_Y = generate_from_config(config, return_labels=True)

prdc = compute_prdc(X, Y, real_labs=labels_X, fake_labs=labels_Y, derive_labelwise=True)
```

`return_labels` is ignored for `type="gaussian"`, which always returns `(X, Y)`.

### Low-level samplers

The samplers can also be called directly without a config:

```python
from dgm_eval.synthetics import sample_gaussian, sample_gmm

rng = np.random.default_rng(0)
X, Y = sample_gaussian(mu_p, mu_q, dim=64, n=1000, rng=rng)
X, Y, lX, lY = sample_gmm(P_means, Q_means, P_weights, Q_weights, dim=2, n=1000,
                          return_labels=True, rng=rng)
```

If `rng` is `None`, they default to `np.random.default_rng(42)`.

### Why use a Distributions object?

Use one when the distribution's shape stays fixed and you generate several configs from it (varying `n_samples` / `seed` / `dim`):

```python
dist = GaussianDistributions(mu_p, mu_q)
configs = {
    f"n={n}": dist.to_config(f"n={n}", n_samples=n, seed=42)
    for n in [100, 1_000, 10_000]
}
```

## Experiment Utilities

All helpers in `dgm_eval.experiment_utils` call `generate_from_config(config)` and pass `(X, Y)` to your metric function. Component labels are **not** passed, so for label-aware experiments call `generate_from_config(..., return_labels=True)` yourself (see `xp/xp_label_noising.py`).

### Run Multiple Configs + Metrics

```python
from dgm_eval.experiment_utils import run_experiment_grid

configs = {
    "config_a": SyntheticConfig(...),
    "config_b": SyntheticConfig(...),
}

metrics = {
    "prdc_P": lambda X, Y: compute_prdc(X, Y, nearest_k=5)["P"],
    "prdc_R": lambda X, Y: compute_prdc(X, Y, nearest_k=5)["R"],
}

results = run_experiment_grid(configs, metrics)
# results[config_name][metric_name] = value
```

### Sweep Over Random Seeds

```python
from dgm_eval.experiment_utils import run_with_seed_sweep, aggregate_results_across_seeds

metric_fn = lambda X, Y: compute_prdc(X, Y, nearest_k=5)["P"]
seed_results = run_with_seed_sweep(config, metric_fn, seeds=[42, 123, 456, 789])
# seed_results[seed] = metric value

mean, std = aggregate_results_across_seeds(seed_results)
```

`aggregate_results_across_seeds` handles scalars, arrays (per-element mean/std), and nested dicts (recursively), and ignores NaNs.

### Sweep Over a Config Parameter

`vary_parameter(base_config, param_name, values, metric_fn)` overrides one key of `config.params` and returns `results[value] = metric value`. The values are used as dict keys, so they must be hashable (e.g. tuples for GMM means):

```python
from dgm_eval.experiment_utils import vary_parameter

base_config = GMMDistributions(
    P_means=np.array([0.0, 2.0]),
    Q_means=np.array([0.0, 2.0]),
    P_weights=np.array([0.5, 0.5]),
    Q_weights=np.array([0.5, 0.5]),
).to_config("q_weight_sweep", dim=8, n_samples=2000, seed=42)

q_weights = [(0.5, 0.5), (0.7, 0.3), (0.9, 0.1)]
metric_fn = lambda X, Y: compute_prdc(X, Y, nearest_k=5)["R"]

results = vary_parameter(base_config, "Q_weights", q_weights, metric_fn)
# results[(0.7, 0.3)] = recall
```

For array-valued parameters such as the Gaussian `mu_q`, build the configs in a loop instead:

```python
dim = 64
shifts = [0.0, 0.5, 1.0, 1.5, 2.0]
configs = {
    f"shift={s}": GaussianDistributions(np.zeros(dim), np.ones(dim) * s / np.sqrt(dim))
    .to_config(f"shift={s}", n_samples=2000)
    for s in shifts
}
results = run_experiment_grid(configs, {"R": metric_fn})
```

### Full Experiment with Seed Sweep + Aggregation

```python
from dgm_eval.experiment_utils import run_experiment_with_seed_sweep

results = run_experiment_with_seed_sweep(configs, metrics, seeds=[42, 123, 456], aggregate=True)
# results[config_name][metric_name] = {"mean": ..., "std": ...}
# aggregate=False -> results[config_name][metric_name][seed] = value
```

### Nested Sweeps (Multiple Parameter Dimensions)

For sweeping over configs and other parameters (classifiers, k values, ...). The metric function receives a third argument, the dict of the current parameter values:

```python
from dgm_eval.experiment_utils import run_nested_sweep

sweep_params = {
    "clf": ["Cov", "iPR", "kNN", "kde"],
    "k": [4, -1],  # -1 -> sqrt(n)
}

def metric(X, Y, params):
    return compute_pr_curve(X, Y, params["clf"], params["k"])["P"]

results = run_nested_sweep(configs, sweep_params, metric, seeds=[42, 123, 456], aggregate=True)
# results[config_name]["clf=Cov_k=4"] = {"mean": ..., "std": ...}
```

Every combination of parameters is run for each config and each seed, then aggregated (mean ± std) across seeds.

### Other Helpers

- `binary_label_noising(labels, p, rng=None)`: flips a fraction `p` of binary (0/1) labels.
- `plot_distributions(Xr, Xg, Yr=None, Yg=None, n_show=500, out=None)`: histograms of the projection on the diagonal axis `1/sqrt(dim)`, plus scatter plots of dims 0-1, pooled and per class. Saves to `out` if given, otherwise shows the figure.

## Example: Small-n Degeneracy (Nested Sweep)

```python
import numpy as np
from dgm_eval.experiment_utils import run_nested_sweep
from dgm_eval.metrics import compute_pr_curve
from dgm_eval.synthetics import GaussianDistributions

dim, shift = 64, 1.5
mu_p = np.zeros(dim)
dists = {
    "mismatch": GaussianDistributions(mu_p, np.ones(dim) * shift / np.sqrt(dim)),
    "null": GaussianDistributions(mu_p, mu_p.copy()),
}

configs = {
    f"{case}_n={n}": dist.to_config(f"{case}_n={n}", n_samples=n)
    for case, dist in dists.items()
    for n in [10, 50, 100, 500]
}

sweep_params = {"clf": ["Cov", "iPR", "kNN", "kde"], "k": [4, -1]}

def metric(X, Y, params):
    return compute_pr_curve(X, Y, params["clf"], params["k"])["P"]

results = run_nested_sweep(configs, sweep_params, metric, seeds=[42, 123, 456])

for config_name in sorted(results):
    for combo, stats in results[config_name].items():
        print(config_name, combo, stats["mean"], stats["std"])
```

## Example: Class-Conditional GMM

```python
import numpy as np
from dgm_eval.metrics import compute_pr_curve
from dgm_eval.synthetics import GMMDistributions, generate_from_config

K = 2
means = np.arange(K) * 2.75
weights = np.full(K, 1 / K)

config = GMMDistributions(means, means, weights, weights).to_config(
    "gmm_2d", dim=2, n_samples=10_000, seed=0
)
Xr, Xg, Yr, Yg = generate_from_config(config, return_labels=True)

overall = compute_pr_curve(Xr, Xg, "kNN", 4)
per_class = [compute_pr_curve(Xr[Yr == c], Xg[Yg == c], "kNN", 4) for c in range(K)]
```

## File Structure

```
dgm-eval/
├── dgm_eval/
│   ├── synthetics/
│   │   ├── __init__.py           # Exports
│   │   ├── configs.py            # SyntheticConfig, SyntheticDataset
│   │   ├── distributions.py      # GaussianDistributions, GMMDistributions
│   │   └── samplers.py           # sample_gaussian, sample_gmm, generate_from_config
│   ├── experiment_utils.py       # Experiment helpers
│   ├── metrics/                  # Metric functions
│   └── ...
├── xp/                           # Experiment scripts
└── ...
```

## Reproducibility

`generate_from_config` always seeds its RNG from `config.seed` (default `42`). Its `rng` argument is currently ignored. The same config gives the same data.

Configs serialize with `config.to_dict()` / `SyntheticConfig.from_dict(d)`. For Gaussian configs, `mu_p` / `mu_q` are stored as NumPy arrays, so convert them with `.tolist()` before dumping to JSON. GMM configs already store lists.

## Tips

1. **Serialize configs**: save `config.to_dict()` alongside results.
2. **Aggregate over seeds**: report mean ± std.
3. **Mind the scale**: shifts are per coordinate, so the Euclidean separation grows as `sqrt(dim)`. Divide by `sqrt(dim)` to keep separation fixed across dimensions.
4. **Save results**: store result dicts as JSON/pickle for later analysis.
