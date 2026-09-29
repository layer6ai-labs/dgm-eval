import os
import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

####################################################################################################
# Table of Contents
####################################################################################################
# Filename description utilities
# Dict/List utilities
# Dict/List aggregation utilities
# Data management utilities
# Reading utilities
####################################################################################################


# Filename description utilities
####################################################################################################


def make_description_dic(args, real_dm, gen_dataset_names):

    desc = {}

    desc["train_ds"] = real_dm.dataset_name
    desc["gen_ds"] = "_".join(gen_dataset_names)
    desc["model"] = (
        args.model + "_" + args.arch if args.arch is not None else args.model
    )
    desc["metrics"] = "+".join(sorted(args.metrics)).replace(
        "pr_curve", f"pr_curve_{args.pr_curve_clf}"
    )
    desc["nimgs"] = len(real_dm.dataloader.dataset)

    if args.nruns > 1:
        desc["nruns"] = args.nruns

    if args.label_method is not None:
        desc["lab"] = args.label_method

    if set(args.metrics) & {"fls", "fls_overfit", "prdc", "pr_curve"}:
        desc["reduced"] = args.reduced_n

    if set(args.metrics) & {"prdc", "pr_curve"}:
        desc["k"] = args.nearest_k

    if args.random_labels:
        desc["random_labs"] = ""

    return desc


def make_str(desc):
    out_str = ""

    for k, v in desc.items():
        if k in ["train_ds"]:
            out_str += v.replace("-", "_")
            out_str += "_vs_"
        elif k in ["gen_ds", "model", "metrics"]:
            out_str += v.replace("-", "_")
            out_str += "-"
        elif v is None:
            out_str += f"{k}_None".replace("-", "_")
            out_str += "-"
        elif not v:
            out_str += f"{k}".replace("-", "_")
            out_str += "-"
        else:
            out_str += f"{k}_{v}".replace("-", "_")
            out_str += "-"

    out_str = out_str.removesuffix("-")

    return out_str.replace("_-", "-")


def get_substring(s, prefix=None, innix=None):
    for split in s.split("-"):
        if prefix is not None and not split.startswith(prefix):
            continue
        if innix is not None and innix not in split:
            continue
        if prefix is None and innix is None:
            continue
        return split
    return ""


def remove_subs(s, *prefixes):
    for prefix in prefixes:
        sub = get_substring(s, prefix=prefix)
        if sub:
            s = s.replace("-" + sub, "")
    return s


####################################################################################################


def extend_path(p):
    parts = []
    current = p

    while True:
        base = os.path.basename(current)
        parent = os.path.dirname(current)

        parts.append(base)

        if base.isdigit() or base in ("train", "test"):
            current = parent
        else:
            break

    return "_".join(reversed(parts))


# Dict/List conversion utilities
####################################################################################################


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    """
    Convert a list of (possibly nested) dicts into a single dict of lists.
    Recurses through nested dicts; any non-dict value (int, float, list,
    np.ndarray, etc.) is treated as a leaf and simply collected across
    the list into a list.
    """
    if not list_of_dicts:
        return {}

    # union of keys across all dicts, preserving first-seen order
    all_keys = []
    seen = set()
    for d in list_of_dicts:
        for k in d:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    result = {}
    for k in all_keys:
        values = [d.get(k) for d in list_of_dicts]  # None if missing in some dict
        if all(isinstance(v, dict) for v in values):
            result[k] = list_of_dicts_to_dict_of_lists(values)
        else:
            result[k] = values
    return result


# Curve interpolation utilities
####################################################################################################


def build_grid(n_points=1000, start=0.0, end=1.0):
    return np.linspace(start, end, n_points, dtype=np.float64)


def interpolate_on_grid(x, y, grid):
    """
    Interpolate a single curve (x, y) onto a grid.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # np.interp requires x to be increasing
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    y_interp = np.interp(
        grid,
        x_sorted,
        y_sorted,
        left=y_sorted[0],
        right=0.0,
    )

    return y_interp


def aggregate_curves(curves, n_points=1000, start=0.0, end=1.0):
    """
    Aggregate M curves onto a common grid.

    Args:
        curves: list of (x, y) tuples, one per curve (length M)
        n_points, start, end: passed to build_grid

    Returns:
        grid: the common grid, shape (n_points,)
        curves_interp: interpolated y-values for each curve, shape (M, n_points)
        D: list (per curve) of lists of (x, y) pairs on the common grid
    """
    grid = build_grid(n_points, start, end)

    curves_interp = np.stack([interpolate_on_grid(x, y, grid) for x, y in curves])

    D = [list(zip(grid, curves_interp[m])) for m in range(curves_interp.shape[0])]

    return grid, curves_interp, D


def plot_curves(curves, grid, curves_interp, names=None, band_mode="std"):
    """
    Args:
        curves: list of (x, y) tuples, the raw curves (length M)
        grid: common grid, shape (n_points,)
        curves_interp: interpolated curves, shape (M, n_points)
        names: optional list of labels, length M
        band_mode: "std" or "quant_<level>", passed to compute_band
    """
    M = len(curves)
    if names is None:
        names = [f"{m + 1}" for m in range(M)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: raw vs interpolated curves ---
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, M))
    for m, (x, y) in enumerate(curves):
        ax.plot(
            x,
            y,
            "o--",
            color=colors[m],
            alpha=0.4,
            label=f"{names[m]} (raw)",
        )
        ax.plot(
            grid,
            curves_interp[m],
            "-",
            color=colors[m],
            linewidth=2,
            label=f"{names[m]} (interp)",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Raw vs. interpolated curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- Right: aggregate curve with band ---
    ax = axes[1]
    mean_values, lower_values, upper_values = agg_mean_band(
        curves_interp, mode=band_mode
    )

    ax.plot(grid, mean_values, color="black", linewidth=2, label="Mean curve")
    ax.fill_between(
        grid,
        lower_values,
        upper_values,
        color="gray",
        alpha=0.3,
        label=f"Band ({band_mode})",
    )
    for m in range(M):
        ax.plot(grid, curves_interp[m], color=colors[m], alpha=0.5, linewidth=1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Aggregated curve ({band_mode})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# Dict/List aggregation utilities
####################################################################################################


def aggregate(d, agg_scalar="std", agg_array="std"):
    """
    Aggregate dict leaves into (mean, lower, upper).

    Handles plain scalars, arrays and nested dicts.
    Handles NaN values (via nan-aware reductions in agg_mean_band).

    For arrays, aggregates per-point.
    For scalars, aggregation mode is `agg_scalar`; for arrays, `agg_array`.
    Both accept "std" or "quant_<level>" (e.g. "quant_5", "quant_10").
    """

    if isinstance(d, dict):
        mean, lower, upper = {}, {}, {}
        for k in d:
            mean[k], lower[k], upper[k] = aggregate(d[k], agg_scalar, agg_array)
        return mean, lower, upper

    array = np.array(d, dtype=float)
    agg = agg_array if array.ndim >= 2 else agg_scalar

    return agg_mean_band(array, mode=agg)


def agg_mean_band(arrays, mode="std"):
    """
    Compute mean curve and uncertainty band across arrays.

    Args:
        arrays: array of shape (n_arrays, n_points)
        mode: "std" or "quant_<level>", e.g. "quant_5", "quant_10", "quant_25"

    Returns:
        mean_curve: shape (n_points,)
        lower_curve: shape (n_points,)
        upper_curve: shape (n_points,)
    """

    mean_values = np.nanmean(arrays, axis=0)

    if mode == "std":
        std_values = np.nanstd(arrays, axis=0)
        lower_values = mean_values - std_values
        upper_values = mean_values + std_values

    else:
        match = re.fullmatch(r"quant_(\d+(?:\.\d+)?)", mode)
        if not match:
            raise ValueError(f"Invalid mode '{mode}'.")

        level = float(match.group(1))
        if not (0 <= level <= 50):
            raise ValueError(
                f"Quantile level must be between 0 and 50 (got {level}). "
                f"It represents the lower-tail percentile; the upper bound "
                f"uses 100 - level automatically."
            )

        lower_values = np.percentile(arrays, level, axis=0)
        upper_values = np.percentile(arrays, 100 - level, axis=0)

    return mean_values, lower_values, upper_values


# Data management utilities
####################################################################################################


def strat_tts_optional_labs(X, y, test_size=0.5, random_state=42):
    if y is None:
        X_train, X_test = train_test_split(
            X,
            test_size=test_size,
            random_state=random_state,
        )
        return X_train, X_test, None, None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# Reading utilities
####################################################################################################


def load_run(path, run="run00"):
    data = np.load(path, allow_pickle=True)
    run = data["scores"].item()[run]
    return run


def get_label_keys(run):
    return sorted(
        [k for k in run if k.startswith("label-") and not k.endswith("_std")],
        key=lambda x: int(x.split("-")[1]),
    )


####################################################################################################
# Logging utilities
####################################################################################################


def fmt_duration(seconds):
    """Format a duration in seconds as 'Hh Mm' (or 'Mm' if under an hour)."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"
