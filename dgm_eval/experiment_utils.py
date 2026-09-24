"""Utilities for running synthetic data experiments."""

import itertools
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from dgm_eval.synthetics import SyntheticConfig, generate_from_config

###############################################################################
# UTILS
###############################################################################


def binary_label_noising(
    Y: np.ndarray,
    p: float,
    rng: np.random.Generator = None,
) -> tuple:
    """
    Flip a fraction p of binary labels in Y.
    """

    if rng is None:
        rng = np.random.default_rng(42)

    n_samples = Y.shape[0]
    n_noisy = int(p * n_samples)
    noisy_indices = rng.choice(n_samples, size=n_noisy, replace=False)

    Y_noisy = Y.copy()
    Y_noisy[noisy_indices] = 1 - Y_noisy[noisy_indices]  # flip labels

    return Y_noisy


###############################################################################
# RUNS
###############################################################################


def run_with_seed_sweep(
    config: SyntheticConfig,
    metric_fn: Callable[[np.ndarray, np.ndarray], Any],
    seeds: list[int],
) -> dict[int, Any]:
    """
    Run a metric computation across multiple random seeds.

    Args:
        config: SyntheticConfig (seed field will be overridden)
        metric_fn: Function that takes (X, Y) and returns metric value(s)
        seeds: List of random seeds to sweep over

    Returns:
        Dict mapping seed -> metric result
    """
    results = {}
    for seed in seeds:
        config_copy = SyntheticConfig(
            name=config.name,
            type=config.type,
            dim=config.dim,
            n_samples=config.n_samples,
            seed=seed,
            params=config.params,
        )
        X, Y = generate_from_config(config_copy)
        results[seed] = metric_fn(X, Y)
    return results


def vary_parameter(
    base_config: SyntheticConfig,
    param_name: str,
    values: list[Any],
    metric_fn: Callable[[np.ndarray, np.ndarray], Any],
) -> dict[Any, Any]:
    """Grid search over a parameter in the config.

    Args:
        base_config: Base SyntheticConfig
        param_name: Name of parameter to vary (in config.params)
        values: List of values to try
        metric_fn: Function that takes (X, Y) and returns metric value(s)

    Returns:
        Dict mapping parameter value -> metric result
    """
    results = {}
    for value in values:
        config_copy = SyntheticConfig(
            name=base_config.name,
            type=base_config.type,
            dim=base_config.dim,
            n_samples=base_config.n_samples,
            seed=base_config.seed,
            params={**base_config.params, param_name: value},
        )
        X, Y = generate_from_config(config_copy)
        results[value] = metric_fn(X, Y)
    return results


def aggregate_results_across_seeds(
    results: dict[int, Any],
) -> tuple[Any, Any]:
    """Aggregate metric results across multiple seeds.

    Handles:
    - Scalar values: compute mean and std
    - Array values: compute mean and std per element
    - Dict values (e.g., per-class metrics): recursively aggregate

    Args:
        results: Dict mapping seed -> metric result

    Returns:
        (mean, std) aggregated across seeds
    """
    if not results:
        raise ValueError("results dict is empty")

    values = list(results.values())

    # Handle nested dicts (e.g., per-class metrics)
    if isinstance(values[0], dict):
        mean, std = {}, {}
        for k in values[0]:
            mean[k], std[k] = aggregate_results_across_seeds(
                {seed: v[k] for seed, v in results.items()}
            )
        return mean, std

    # Convert to array and aggregate
    arr = np.array(values)

    # Array-valued metrics: aggregate per-element
    if arr.ndim >= 2:
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
    # Scalar metrics: aggregate globally
    else:
        return np.nanmean(arr), np.nanstd(arr)


def run_experiment_grid(
    configs: dict[str, SyntheticConfig],
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], Any]],
) -> dict[str, dict[str, Any]]:
    """Run multiple metrics on multiple configs.

    Args:
        configs: Dict mapping config_name -> SyntheticConfig
        metrics: Dict mapping metric_name -> metric_fn

    Returns:
        Dict[config_name][metric_name] -> result
    """
    results = defaultdict(dict)

    for config_name, config in configs.items():
        X, Y = generate_from_config(config)
        for metric_name, metric_fn in metrics.items():
            results[config_name][metric_name] = metric_fn(X, Y)

    return dict(results)


def run_experiment_with_seed_sweep(
    configs: dict[str, SyntheticConfig],
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], Any]],
    seeds: list[int],
    aggregate: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run multiple metrics on multiple configs, swept over seeds.

    Args:
        configs: Dict mapping config_name -> SyntheticConfig
        metrics: Dict mapping metric_name -> metric_fn
        seeds: List of random seeds
        aggregate: If True, return aggregated (mean, std). If False, return all results.

    Returns:
        If aggregate=True: Dict[config_name][metric_name] -> (mean, std)
        If aggregate=False: Dict[config_name][metric_name][seed] -> result
    """
    results = defaultdict(lambda: defaultdict(dict))

    for config_name, config in configs.items():
        for metric_name, metric_fn in metrics.items():
            seed_results = run_with_seed_sweep(config, metric_fn, seeds)
            if aggregate:
                mean, std = aggregate_results_across_seeds(seed_results)
                results[config_name][metric_name] = {"mean": mean, "std": std}
            else:
                results[config_name][metric_name] = seed_results

    return dict(results)


def run_nested_sweep(
    configs: dict[str, SyntheticConfig],
    sweep_params: dict[str, list[Any]],
    metric_fn: Callable[[np.ndarray, np.ndarray, dict[str, Any]], Any],
    seeds: list[int],
    aggregate: bool = True,
) -> dict[str, dict[str, Any]]:
    """
    Run metrics with nested parameter sweeps.

    Useful for sweeping over configs and additional parameters (e.g., classifier type,
    k value) without having to manually nest loops.

    Args:
        configs: Dict mapping config_name -> SyntheticConfig
        sweep_params: Dict mapping param_name -> list of values to sweep
        metric_fn: Function(X, Y, params_dict) -> result
                  where params_dict contains current values from sweep_params
        seeds: List of random seeds for each config
        aggregate: If True, aggregate results across seeds

    Returns:
        Dict[config_name][param_combo] -> metric_result or {"mean": ..., "std": ...}

    Example:
        configs = {"n=100": config1, "n=200": config2}
        sweep_params = {"clf": ["Cov", "iPR"], "k": [4, -1]}

        def metric(X, Y, params):
            clf = params["clf"]
            k = params["k"]
            return compute_pr_curve(X, Y, clf, k)

        results = run_nested_sweep(configs, sweep_params, metric, seeds=[42, 123])
    """
    results = defaultdict(dict)

    # Generate all combinations of sweep parameters
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[name] for name in param_names]
    param_combos = itertools.product(*param_values)

    for config_name, config in configs.items():
        X, Y = generate_from_config(config)

        for combo in param_combos:
            params_dict = dict(zip(param_names, combo))
            combo_key = "_".join(f"{k}={v}" for k, v in params_dict.items())

            # Sweep over seeds for this combo
            seed_results = {}
            for seed in seeds:
                # Update seed in config for reproducibility
                config_copy = SyntheticConfig(
                    name=config.name,
                    type=config.type,
                    dim=config.dim,
                    n_samples=config.n_samples,
                    seed=seed,
                    params=config.params,
                )
                X, Y = generate_from_config(config_copy)
                seed_results[seed] = metric_fn(X, Y, params_dict)

            if aggregate:
                mean, std = aggregate_results_across_seeds(seed_results)
                results[config_name][combo_key] = {"mean": mean, "std": std}
            else:
                results[config_name][combo_key] = seed_results

    return dict(results)


################################################################################
# PLOT UTILS
################################################################################


def plot_distributions(
    Xr,
    Xg,
    Yr=None,
    Yg=None,
    n_show: int = 500,
    out: str | None = None,
):
    K = len(np.unique([Yr, Yg])) if Yr is not None and Yg is not None else 1
    dim = Xr.shape[1]
    signal_axis = np.ones(dim) / np.sqrt(dim)

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, K + 1, figsize=(3 * (K + 1), 6.0))
    n_show = min(n_show, Xr.shape[0], Xg.shape[0])

    # --- top row: signal-axis projections (true separation) ---

    ax = axes[0, K if K > 1 else 0]
    ax.hist(
        Xr @ signal_axis,
        bins=30,
        alpha=0.5,
        density=True,
        color="gray",
        label="real (pool)",
    )
    ax.hist(
        Xg @ signal_axis,
        bins=30,
        alpha=0.5,
        density=True,
        color=cmap(K if K > 1 else 0),
        label="gen. (pool)",
    )
    ax.set_title("pooled", fontsize=10)
    ax.legend(fontsize=7)

    if K > 1:
        for k in range(K):
            ax = axes[0, k]
            Xr_k = Xr[Yr == k]
            Xg_k = Xg[Yg == k]

            # Project the data onto the signal axis
            proj_Xr_k = Xr_k @ signal_axis
            proj_Xg_k = Xg_k @ signal_axis
            ax.hist(
                proj_Xr_k,
                bins=20,
                alpha=0.5,
                density=True,
                color="gray",
                label="real",
            )
            ax.hist(
                proj_Xg_k,
                bins=20,
                alpha=0.5,
                density=True,
                color=cmap(k),
                label="gen.",
            )
            ax.axvline(
                proj_Xr_k.mean(),
                color="gray",
                linestyle="--",
                linewidth=1.0,
            )
            ax.axvline(
                proj_Xg_k.mean(),
                color=cmap(k),
                linestyle="--",
                linewidth=1.0,
            )
            ax.set_title(f"class {k}", fontsize=10)
            if k == 0:
                ax.set_ylabel(
                    "Density, projection on\nclass discriminative axis", fontsize=9
                )
                ax.legend(fontsize=7)

    # --- bottom row: raw dims 0-1 scatter (visually understates separation by sqrt(dim)) ---

    ax = axes[1, K if K > 1 else 0]
    ax.scatter(
        *Xr[:, :2].T,
        s=8,
        alpha=0.4,
        color="gray",
        marker="o",
        label="real (pool)",
    )
    ax.scatter(
        *Xg[:, :2].T,
        s=8,
        alpha=0.4,
        color=cmap(K if K > 1 else 0),
        marker="x",
        label="gen. (pool)",
    )
    ax.set_xlabel("dim 0")
    ax.set_title("pooled", fontsize=10)
    ax.legend(fontsize=7)

    if K > 1:
        for k in range(K):
            ax = axes[1, k]
            Xr_k = Xr[Yr == k]
            Xg_k = Xg[Yg == k]

            ax.scatter(
                *Xr_k[:, :2].T,
                s=8,
                alpha=0.5,
                color="gray",
                marker="o",
                label="real",
            )
            ax.scatter(
                *Xg_k[:, :2].T,
                s=8,
                alpha=0.5,
                color=cmap(k),
                marker="x",
                label="gen.",
            )
            ax.set_xlabel("dim 0")
            if k == 0:
                ax.set_ylabel(
                    "raw dims 0-1\n(understates sep. by $\\sqrt{dim}$)", fontsize=9
                )
                ax.legend(fontsize=7)

    fig.suptitle(
        rf"$K={K}$, d={dim}, $n={n_show}$" + "\n" + "class shift ",
        fontsize=10,
    )
    plt.tight_layout()

    if out is None:
        plt.show()
    else:
        plt.savefig(out, bbox_inches="tight", dpi=300)
        plt.close()
