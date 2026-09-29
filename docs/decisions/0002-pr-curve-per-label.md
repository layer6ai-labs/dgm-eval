# Context 

PR curves per label has initially been implemented by combining calls to `compute_pr_curve` on filtered datasets per label or using the flag argument `derive_labelwise`, depending on the `label_method`.

The `compute_pr_curve_cond` function introduces new options for evaluation per label through the `handle_labels` argument:
- `filter_dataset` (FD): Filter samples per label, then fit an unconditional classifier on label-`y` samples only.
- `filter_probes` (FP): Fit the classifier on pooled samples, then mask the probes per label.
- `filter_centers` (FC): Keep pooled radii/bandwidths/neighbourhoods, mask the centers per label, then evaluate per label.

Methods compare as:

| `label_method` + `compute_pr_curve` |	`compute_pr_curve_cond` with `handle_labels` |
| - | - |
| No labels | `None` |
| `FRC`, filter then reduce | `FD` |
| `RFC`, reduce then filter | `FD` |
| `RCF` (`derive_labelwise=True`) | `FP` (previous behaviour) or `FC` (new) |

The issue is that `label_method` is a general setting for all metrics while `handle_labels` only applies to PR curves.

Replacing the `label_method` loop structure with `handle_labels` would not leverage the structure of the code. 

# Decision

We keep the `label_method` philosophy and combine it with a `handle_labels` value check (cf table above).

The functions are merged: `compute_pr_curve_cond` is renamed `compute_pr_curve` and the old `compute_pr_curve` (with `derive_labelwise`) is removed. The default stays `split=False`, as in the old function.

In `compute_scores`:
- `frc` / `rfc` call `compute_pr_curve(..., handle_labels=None)` on the filtered data (FD by construction).
- `rcf` passes `Yr`, `Yg` and `handle_labels` (`filter_probes` or `filter_centers`).

The FD branch of `compute_pr_curve` is not used by the CLI, but is kept for standalone use.

CLI:
- `--pr-curve-hl` takes full names (`HANDLE_LABELS`) and is checked against `PR_CURVE_HL_PER_METHOD`:

| `label_method` | allowed `--pr-curve-hl` (first is default) |
| - | - |
| `frc` | `filter_dataset` |
| `rfc` | `filter_dataset` |
| `rcf` | `filter_probes`, `filter_centers` |

  A mismatch raises a `ValueError`. It is `None` unless `--per-label` is set and `pr_curve` is in the metrics.
- `--pr-curve-split` (default `False`) controls the train/test split of the PR curve classifier.
- The output filename gets `hl_FD` / `hl_FP` / `hl_FC` (via `HANDLE_LABELS_ABBR`), and `split` when the flag is set.

# Consequences

Equivalences, checked on synthetic data:
- `rfc` gives the same result as FD when `split=False`.
- `frc` gives the same result as FD only with no subsampling, `split=False`, and equal real/generated counts per label. With `split=True` they differ, because `frc` permutes samples before splitting.
- `rcf` + `filter_probes` gives the same result as the old `derive_labelwise` path.

Behaviour change: FP/FC now return a NaN curve for a label only when it has no probes, since radii are pooled. The old `rcf` returned NaN whenever a label had fewer than `k+1` samples, so small labels can differ from older results.

Known edge case: if the pooled check fails (`Nr` or `Ng` < `k+1`), the returned NaN curve has no `label-i` keys. Aggregation across runs then breaks if other runs do have them.

ImageNet: with the default `reduced_n=10000`, each class has about 10 samples (about 5 probes with a split). Per-label curves are noisy and FD yields many NaN curves; raise `--reduced-n` accordingly.
