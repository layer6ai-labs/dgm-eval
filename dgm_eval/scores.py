import logging
import os
import pathlib
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pprint import pformat

import numpy as np
import pandas as pd

from .metrics import (
    compute_authpct,
    compute_CTscore,
    compute_CTscore_mem,
    compute_CTscore_mode,
    compute_efficient_FD_with_reps,
    compute_FD_infinity,
    compute_FD_with_reps,
    compute_fls,
    compute_fls_overfit,
    compute_mmd,
    compute_per_class_vendi_scores,
    compute_pr_curve,
    compute_prdc,
    sw_approx,
)
from .utils import extend_path, list_of_dicts_to_dict_of_lists, make_str, remove_subs

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s.%(funcName)s: %(message)s",
    force=True,
)

LAMBDAS = np.tan(np.linspace(0, np.pi / 2, 152))[1:-1]


####################################################################################################
# Table of Contents
####################################################################################################
# Aggregate
# Compute
# Labels
# Randomness
# Run
# Save
####################################################################################################


def _fmt_duration(seconds):
    """Format a duration in seconds as 'Hh Mm' (or 'Mm' if under an hour)."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


####################################################################################################
# Aggregate
####################################################################################################


def aggregate(values):
    """Aggregate one metric across runs into (mean, std).

    Handles plain scalars as well as nested dicts (e.g. per-label score dicts),
    and array-valued metrics (e.g. precision/recall curves).
    For array-valued metrics, aggregates per-point across runs.
    Handles NaN values from runs with insufficient samples.
    """
    if isinstance(values[0], dict):
        mean, std = {}, {}
        for k in values[0]:
            mean[k], std[k] = aggregate([v[k] for v in values])
        return mean, std

    arr = np.array(values)
    # If arr is 2D+, it's an array-valued metric: aggregate per-point (along axis 0)
    # If arr is 1D, it's scalar values: aggregate globally
    # Use nanmean/nanstd to ignore NaN values from failed runs (insufficient samples)
    if arr.ndim >= 2:
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)
    else:
        return np.nanmean(arr), np.nanstd(arr)


def aggregate_runwise_scores(all_runs_scores, args):
    """Aggregate scores across multiple runs into mean and std.

    Takes per-run scores (list of dicts with keys: overall, label-i, agg, diff)
    and computes mean and std across runs for each metric and label.

    Args:
        all_runs_scores: list of dicts, one per run, each containing {label_key: {metric: value}}
        args: args object with nruns attribute

    Returns:
        all_scores: dict with aggregated results {label_key: {metric: mean_value, metric_std: std_value}}
    """
    logger.info("Aggregating runwise scores")

    all_scores = {}
    all_keys = all_runs_scores[0].keys()

    for key in all_keys:  # overall, label-i, agg, diff
        all_metrics = all_runs_scores[0][key].keys()
        mean_scores, std_scores = {}, {}

        for metric in all_metrics:
            values = [run[key][metric] for run in all_runs_scores]
            if metric == "realism":
                mean_scores[metric] = values[-1]
                continue
            # values may be scalars or nested per-label dicts
            mean_scores[metric], std_scores[metric] = aggregate(values)

        all_scores[key] = mean_scores
        if args.nruns > 1:
            all_scores[f"{key}_std"] = std_scores

    return all_scores


####################################################################################################
# Compute
####################################################################################################


def compute_scores(
    args,
    rr,
    rg,
    test_reps,
    label_setup=None,
    label_key="overall",
    seed=0,
):
    """
    reps: list of two arrays corresponding to real reps and gen reps, each of shape (Nimage, ndim)
    labels: array of shape (Nimage,) with class labels for generated samples, if available
    """

    if label_setup is not None:
        real_labs, fake_labs, labels = label_setup

        # ----------------------------------------------------------------------------------------------------
        # if real_labs is not None:
        #     unique_real, counts_real = np.unique(real_labs, return_counts=True)
        #     print(
        #         f"[DEBUG compute_scores] real_labs unique values and counts: {dict(zip(unique_real, counts_real))}",
        #         file=sys.stderr,
        #     )
        # ----------------------------------------------------------------------------------------------------

    scores = {}
    vendi_scores = {}

    if "authpct" in args.metrics and label_key == "overall":
        logger.info("Computing authpct \n")
        scores["authpct"] = compute_authpct(rr, rg)

    if "ct" in args.metrics and label_key == "overall":
        logger.info("Computing ct score \n")
        scores["ct"] = compute_CTscore(rr, test_reps, rg)

    if "ct_test" in args.metrics and label_key == "overall":
        logger.info(
            "Computing ct score, modified to identify mode collapse only \n",
            file=sys.stderr,
        )
        scores["ct_test"] = compute_CTscore_mode(rr, test_reps, rg)

    if "ct_modified" in args.metrics and label_key == "overall":
        logger.info(
            "Computing ct score, modified to identify memorization only \n",
            file=sys.stderr,
        )
        scores["ct_modified"] = compute_CTscore_mem(rr, test_reps, rg)

    if "denscov" in args.metrics:
        raise NotImplementedError("denscov metric is currently not implemented")

    if "fd" in args.metrics and label_key == "overall":
        logger.info("Computing FD \n")
        scores["fd"] = compute_FD_with_reps(rr, rg)

    if "fd_eff" in args.metrics and label_key == "overall":
        logger.info("Computing Efficient FD \n")
        scores["fd_eff"] = compute_efficient_FD_with_reps(rr, rg)

    if "fd_infinity" in args.metrics and label_key == "overall":
        logger.info("Computing fd_infinity \n")
        scores["fd_infinity_value"] = compute_FD_infinity(rr, rg)

    if set(args.metrics) & {"fls", "fls_overfit"} and label_key == "overall":
        rng = np.random.default_rng(seed)

        train_reps, gen_reps = rr, rg
        reduced_n = min(
            args.reduced_n,
            train_reps.shape[0] // 2,
            test_reps.shape[0],
            gen_reps.shape[0],
        )

        test_reps = test_reps[rng.choice(test_reps.shape[0], reduced_n, replace=False)]
        gen_reps = gen_reps[rng.choice(gen_reps.shape[0], reduced_n, replace=False)]

        logger.info("Computing fls \n")
        # fls must be after ot, as it changes train_reps
        train_reps = train_reps[
            rng.choice(train_reps.shape[0], 2 * reduced_n, replace=False)
        ]
        train_reps, baseline_reps = train_reps[:reduced_n], train_reps[reduced_n:]

        if "fls" in args.metrics:
            scores["fls"] = compute_fls(
                train_reps,
                baseline_reps,
                test_reps,
                gen_reps,
            )
        if "fls_overfit" in args.metrics:
            scores["fls_overfit"] = compute_fls_overfit(
                train_reps,
                baseline_reps,
                test_reps,
                gen_reps,
            )

    if "kd" in args.metrics and label_key == "overall":
        logger.info("Computing KD \n")
        mmd_values = compute_mmd(rr, rg)
        scores["kd_value"] = mmd_values.mean()
        scores["kd_variance"] = mmd_values.std()

    if "prdc" in args.metrics:  # compute for overall and per label
        reduced_n = min(args.reduced_n, rr.shape[0], rg.shape[0])

        logger.info(
            f"\n Computing PRDC with:\n"
            f"\t samples = {reduced_n}\n"
            f"\t k = {args.nearest_k}\n"
            + (f"\t label_method = {args.label_method}" if args.per_label else "")
        )

        rng = np.random.default_rng(seed)
        inds0 = rng.choice(rr.shape[0], reduced_n, replace=False)
        inds1 = rng.choice(rg.shape[0], reduced_n, replace=False)

        # ----------------------------------------------------------------------------------------------------
        # if args.per_label and args.label_method == "rcf" and real_labs is not None:
        #     print(
        #         f"[DEBUG PRDC] Using labels - real_labs sample: {real_labs[inds0][:5]}, fake_labs sample: {fake_labs[inds1][:5]}",
        #         file=sys.stderr,
        #     )
        # ----------------------------------------------------------------------------------------------------

        scores["prdc"] = compute_prdc(
            rr[inds0],
            rg[inds1],
            nearest_k=args.nearest_k,
            **{
                "derive_labelwise": True,
                "real_labs": real_labs[inds0],
                "fake_labs": fake_labs[inds1],
            }
            if args.label_method == "rcf"
            else {},
        )

        if args.label_method == "rfc":
            for idx, lab in enumerate(labels):
                label_key = f"label-{idx}"
                logger.info(f"--- {label_key} (rfc) ---")

                # ----------------------------------------------------------------------------------------------------
                # n_real = (real_labs[inds0] == lab).sum()
                # n_fake = (fake_labs[inds1] == lab).sum()
                # if args.random_labels:
                #     print(
                #         f"[DEBUG RFC] label={lab}: n_real={n_real}, n_fake={n_fake}",
                #         file=sys.stderr,
                #     )
                # ----------------------------------------------------------------------------------------------------

                scores["prdc"][label_key] = compute_prdc(
                    rr[inds0][real_labs[inds0] == lab],
                    rg[inds1][fake_labs[inds1] == lab],
                    nearest_k=args.nearest_k,
                )

                # ----------------------------------------------------------------------------------------------------
                # if args.random_labels:
                #     print(
                #         f"[DEBUG PRDC values] label={lab}: {scores["prdc"][label_key] }",
                #         file=sys.stderr,
                #     )
                # ----------------------------------------------------------------------------------------------------

        # FIX: move realism to its own metric
        # if "realism" not in args.metrics: # rebuild in another if, keep all generated do not downsample
        # Realism is returned for each sample, so do not shuffle if this metric is desired.
        # Else filenames and realism scores will not align
        # inds1 = rng.choice(
        #     inds1,
        #     min(inds1.shape[0], reduced_n),
        #     replace=False,
        # )

    if set(args.metrics) & {"pr_curve"}:  # compute for overall and per label
        reduced_n = min(args.reduced_n, rr.shape[0], rg.shape[0])

        logger.info(
            f"\n Computing PR curve with:\n"
            f"\t classifier =  {args.pr_curve_clf}\n"
            f"\t samples = {reduced_n}\n"
            f"\t k = {args.nearest_k}\n"
            f"\t split = {args.split}\n"
            + (f"\t label_method = {args.label_method}" if args.per_label else "")
        )

        rng = np.random.default_rng(seed)
        inds0 = rng.choice(rr.shape[0], reduced_n, replace=False)
        inds1 = rng.choice(rg.shape[0], reduced_n, replace=False)

        scores[f"pr_curve_{args.pr_curve_clf}"] = compute_pr_curve(
            rr[inds0],
            rg[inds1],
            clf=args.pr_curve_clf,
            nearest_k=args.nearest_k,
            split=args.split,
            **{
                "derive_labelwise": True,
                "real_labs": real_labs[inds0],
                "fake_labs": fake_labs[inds1],
            }
            if args.label_method == "rcf"
            else {},
        )

        if args.label_method == "rfc":
            for idx, lab in enumerate(labels):
                label_key = f"label-{idx}"
                logger.info(f"--- {label_key} (rfc) ---")

                scores[f"pr_curve_{args.pr_curve_clf}"][label_key] = compute_pr_curve(
                    rr[inds0][real_labs[inds0] == lab],
                    rg[inds1][fake_labs[inds1] == lab],
                    clf=args.pr_curve_clf,
                    nearest_k=args.nearest_k,
                    split=args.split,
                )

    if "sw_approx" in args.metrics and label_key == "overall":
        logger.info("Aprroximating Sliced W2.")
        scores["sw_approx"] = sw_approx(rr, rg)

    if "vendi" in args.metrics and label_key == "overall":
        logger.info("Calculating diversity score")
        # scores['vendi'] = compute_vendi_score(reps[1])
        vendi_scores = compute_per_class_vendi_scores(rg, labels[1])
        scores["mean_vendi_per_class"] = vendi_scores.mean()

    logger.debug(f"{label_key} scores:")
    logger.debug(pformat(scores))

    return scores, vendi_scores


####################################################################################################
# Labels
####################################################################################################


def labelwise_setup(args, labels):
    """Resolve per-label filtering inputs from the overall labels.

    Returns (real_labs, gen_labs, label_values). label_values is empty unless
    --per-label is set and both real and generated samples carry labels, in
    which case only the overall score is computed.
    """
    if not args.per_label or labels is None:
        return None, None, np.array([])  # FIX: problem with vendi score ?

    real_labs = None if labels[0] is None else np.asarray(labels[0])
    gen_labs = None if labels[1] is None else np.asarray(labels[1])

    if (
        real_labs is None
        or gen_labs is None
        or len(real_labs) == 0
        or len(gen_labs) == 0
    ):
        return real_labs, gen_labs, np.array([])

    return real_labs, gen_labs, np.unique(real_labs)


def unpack_nested_labelwise_metric(run_scores):
    """Auto-detect and unpack all metrics with nested labelwise results.

    Scans run_scores["overall"] for metrics that contain nested per-label dicts
    (keyed by "label-i") and unpacks them into the flat run_scores structure.

    Transforms: run_scores["overall"]["prdc"] = {P: ..., label-0: {P: ...}, label-1: {...}}
    Into: run_scores["overall"]["prdc"] = {P: ...}
          run_scores["label-0"]["prdc"] = {P: ...}
          run_scores["label-1"]["prdc"] = {P: ...}

    This is called for rfc/rcf label methods where metrics return nested results.
    """
    if "overall" not in run_scores:
        return

    for metric_name in list(run_scores["overall"].keys()):
        metric_data = run_scores["overall"].get(metric_name)
        # Check if this metric has nested labelwise structure
        if not isinstance(metric_data, dict):
            continue

        label_keys = [k for k in metric_data.keys() if k.startswith("label-")]
        if not label_keys:
            continue

        # Extract overall results (everything that's not label-i)
        overall_data = {
            k: v for k, v in metric_data.items() if not k.startswith("label-")
        }
        run_scores["overall"][metric_name] = overall_data

        # Unpack per-label results
        for label_key in label_keys:
            label_data = metric_data[label_key]
            if label_key not in run_scores:
                run_scores[label_key] = {}
            run_scores[label_key][metric_name] = label_data


def aggregate_labelwise_scores(run_scores):
    """Average label-* scores within a run into run_scores["agg"] (in place).

    Handles scalars, arrays, and nested dicts (e.g., pr_curve results).
    Also computes run_scores["diff"] as the difference between overall and agg.
    """
    logger.info("Aggregating labelwise scores into 'agg' entry")

    label_keys = [k for k in run_scores if k.startswith("label-")]
    if not label_keys:
        return

    sample_keys = [k for k in run_scores[label_keys[0]] if k != "realism"]
    run_scores["agg"] = {}
    run_scores["diff"] = {}

    for metric_name in sample_keys:
        metric_values = [run_scores[lk][metric_name] for lk in label_keys]
        mean, _ = aggregate(metric_values)
        run_scores["agg"][metric_name] = mean

        if "overall" in run_scores:
            # Compute diff as overall - agg (handle dict-valued metrics component-wise)
            overall_value = run_scores["overall"][metric_name]
            if isinstance(overall_value, dict) and isinstance(mean, dict):
                run_scores["diff"][metric_name] = {
                    k: overall_value[k] - mean[k]
                    for k in overall_value
                    if k in mean and not isinstance(overall_value[k], dict)
                }
            else:
                run_scores["diff"][metric_name] = overall_value - mean


####################################################################################################
# Randomness
####################################################################################################


def randomness_manager(args, real_labs, fake_labs):
    """Yield ``(real_labs_r, fake_labs_r, sub_seed, tag)`` for each run.

    Each run advances exactly one source of randomness while pinning the other,
    so the across-run mean/std isolates that source:

    - ``--random-labels`` advances the *random permutation* (the label
      shuffle) each run while pinning the *random reduction* (the reduced_n
      subsampling seed), isolating label-assignment variance. Both reference
      and generative distributions use the same random permutation.
    - otherwise the true labels are kept and the *random reduction* is advanced
      each run, isolating reduction variance.

    ``sub_seed`` is the seed handed to ``compute_scores``; it drives the random
    reduction inside the prdc block.
    """
    for r in range(args.nruns):
        if args.random_labels:
            # advance the random permutation; pin the random reduction
            # Apply the same permutation to BOTH real and fake labels
            rng_perm = np.random.default_rng(args.seed + r)
            if real_labs is None:
                random_real_labs = None
                random_fake_labs = None
            else:
                perm_real = rng_perm.permutation(len(real_labs))
                perm_fake = rng_perm.permutation(len(fake_labs))
                random_real_labs = real_labs[perm_real]
                random_fake_labs = fake_labs[perm_fake]
            yield random_real_labs, random_fake_labs, args.seed, "(random permutation)"
        else:
            # advance the random reduction; keep true labels
            yield real_labs, fake_labs, args.seed + r, ""


####################################################################################################
# Run
####################################################################################################


def run_compute_scores(args, real_reps, fake_reps, test_reps, labels=None):
    """
    Compute scores with per-label breakdown if args.per_label is set, using the
    specified args.label_method.

    Label method strategies:

    frc (filter-reduce-compute):
        For each label:
        1. Filter the dataset to only include samples with the current label.
        2. Reduce the filtered dataset to a smaller subset (if necessary).
        3. Compute the desired metrics on the reduced dataset.

    rfc (reduce-filter-compute):
        1. Reduce the dataset to a smaller subset (if necessary).
        For each label:
        2. Filter the reduced dataset to only include samples with the current label.
        3. Compute the desired metrics on the filtered dataset.

    rcf (reduce-compute-filter):
        1. Reduce the dataset to a smaller subset (if necessary).
        2. Compute the prerequisites for the desired metrics on the reduced dataset.
        For each label:
        3. Filter the computed results to only include samples with the current label.

    When using `rcf` or `rfc` the output scores will be a nested dict with the overall
    score and per-label scores. For example, the output of compute_prdc will be:
    ```
        scores["prdc"] = {
            "P": ..., "R": ..., "C": ..., "D": ...,
            "label-0": {"P": ..., "R": ...},
            "label-1": {"P": ..., "R": ...},
        }
    ```
    The per-label scores will need to be unpacked.
    """

    all_runs_scores = []  # list of per-run dicts: {label_key: scores_dict}
    vendi_scores = {}

    real_labs, fake_labs, labels = labelwise_setup(args, labels)

    start_time = time.time()
    for r, (real_labs_, fake_labs_, seed_, tag) in enumerate(
        randomness_manager(args, real_labs, fake_labs)
    ):
        logger.info(f"=== Run {r + 1}/{args.nruns} {tag} === \n")
        logger.info(f"Samples with shapes {real_reps.shape} and {fake_reps.shape}\n")

        # ----------------------------------------------------------------------------------------------------
        # if args.random_labels and real_labs_ is not None:
        #     print(
        #         f"[DEBUG] real_labs first 5: {real_labs_[:5]}, fake_labs first 5: {fake_labs_[:5]}"
        #     )
        # ----------------------------------------------------------------------------------------------------

        run_scores = defaultdict(dict)

        logger.info("--- overall ---")
        scores, vs = compute_scores(
            args,  # label method, metrics, metrics parameters
            real_reps,
            fake_reps,
            test_reps,
            label_setup=(real_labs_, fake_labs_, labels),
            label_key="overall",
            seed=seed_,
        )
        if vs:
            vendi_scores = vs

        run_scores["overall"] = scores

        if args.label_method == "frc":
            for idx, lab in enumerate(labels):
                label_key = f"label-{idx}"
                logger.info(f"--- {label_key} (frc) ---")

                rr = real_reps[real_labs_ == lab]
                rg = fake_reps[fake_labs_ == lab]

                scores, _ = compute_scores(
                    args,
                    rr,
                    rg,
                    test_reps,
                    label_setup=None,
                    label_key=label_key,
                    seed=seed_,
                )
                run_scores[label_key].update(scores)

        # Unpack nested labelwise metrics from rfc/rcf into flat structure
        unpack_nested_labelwise_metric(run_scores)
        aggregate_labelwise_scores(run_scores)
        all_runs_scores.append(run_scores)

        logger.debug("Run scores:")
        logger.debug(pformat(run_scores))

        # ---- Progress / ETA reporting ----
        n_done = r + 1
        frac_done = n_done / args.nruns
        elapsed = time.time() - start_time
        avg_per_run = elapsed / n_done
        remaining = avg_per_run * (args.nruns - n_done)
        eta_clock = datetime.now() + timedelta(seconds=remaining)
        logger.info(
            f"\n[progress] {n_done}/{args.nruns} runs done ({frac_done:.0%}) | "
            f"elapsed: {_fmt_duration(elapsed)} | "
            f"remaining: {_fmt_duration(remaining)} | "
            f"ETA: {eta_clock.strftime('%H:%M')}"
            f"\n"
        )

    all_scores = list_of_dicts_to_dict_of_lists(all_runs_scores)

    logger.debug("Final scores:")
    logger.debug(pformat(all_scores))

    return all_scores, vendi_scores


####################################################################################################
# Save
####################################################################################################


def save_score(
    scores,
    output_dir,
    model,
    path,
    ckpt,
    nsample,
    is_only=False,
    metrics=None,
):
    logger.info("\nSaving scores...")

    ckpt_str = ""
    if ckpt is not None:
        ckpt_str = f"_ckpt-{os.path.splitext(os.path.basename(ckpt))[0]}"

    metrics_str = "" if metrics is None else "-" + "_".join(metrics)
    path_str = (
        "_".join([extend_path(p) for p in path]) if path is not None else "overall"
    )

    if is_only:
        out_str = f"inception_score-{path_str}{ckpt_str}_nimage-{nsample}.txt"
    else:
        out_str = f"{model}-{metrics_str}-{path_str}{ckpt_str}-nimage_{nsample}.txt"

    out_path = os.path.join(output_dir, out_str)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for key, value in scores.items():
            if key == "realism":
                continue
            if isinstance(value, dict):
                f.write(f"[{key}]\n")
                for k, v in value.items():
                    if k == "realism":
                        continue
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {value} \n")

    logger.info(f"Wrote scores to {out_path}\n")


def save_scores(description, scores, args, is_only=False, vendi_scores={}):
    logger.info("\nSaving scores from all generated paths...")

    run_params = vars(args)
    run_params["reference_dataset"] = run_params["train"]
    run_params["generated_dataset"] = run_params["gen"]

    ckpt_str = ""
    logger.debug(pformat(scores))

    if is_only:
        description["scores"] = "is"

    out_str = make_str(description)

    results_dir = args.output_dir
    if args.xp in ["sweep_prdc_k"]:
        subdir = remove_subs(out_str, "k", "random_labs")
        results_dir = os.path.join(results_dir, subdir)
    if args.xp in ["sweep_reduced_n"]:
        subdir = remove_subs(out_str, "reduced", "random_labs")
        results_dir = os.path.join(results_dir, subdir)
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    out_path = os.path.join(results_dir, out_str)
    fname = f"{out_path}.npz"

    np.savez(fname, scores=scores, run_params=run_params)
    logger.info(f"Saved scores to {fname}\n")

    if vendi_scores is not None and len(vendi_scores) > 0:
        description["scores"] = "vendi"
        df = pd.DataFrame.from_dict(data=vendi_scores)

        out_str = make_str(description)
        out_path = os.path.join(results_dir, out_str)
        logger.info(f"Saving vendi score to {out_path}.csv")
        df.to_csv(f"{out_path}.csv")
