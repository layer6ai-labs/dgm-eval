# Context

For some metrics, evaluation sometimes necessitate to reduce the dataset to fewer samples (e.g., 50k samples to 10k samples). Then, some structure might need to be computed from the dataset (e.g., kNN balls). Finally, samples are evaluated with the help of the structure to yield a metric or score for the dataset.

For a labeled dataset, computing labelwise metrics needs the practitioner to make several choices. First, samples might need to be filtered per label. Reduction might need to be applied somewhere along the evaluation process. And structure might need to be computed.

Several questions arise: 
- Should they start by filtering the dataset ? 
- Should they start by reducing the dataset ? 
- Should they compute structure on pooled samples or filtered samples ?  

# Decision

We introduce a parameter `label_method` in our project. It can take the following values:
- `frc`: Filter-Reduce-Compute. The naive choice, the dataset is first filtered per label, then the evaluation process applies as usual: reduction if needed, computation of structure if needed, score.
- `rfc`: Reduce-Filter-Compute. Here we start by reducing the pooled samples, then we filter per label. This setting accounts for difficulty of evaluation due to the rarity of samples for some labels. The number of relevant samples available for structure computation and evaluation is always smaller than for the `frc` method since we start by reducing the pooled dataset before filtering per label.
- `rcf`: Reduce-Compute-Filter. In this setting, we first apply reduction on the pooled dataset. Structure is computed on the reduced pooled dataset and leverages information from other-label samples. Finally, points are filtered per label. Evaluation is carried out per label on structure computed on pooled labels. 

Note that we do not implement situation where "compute" happens before "reduce" since reduction is carried out for computational purposes.

# Consequences

Each method modifies where the loop over labels is created.  
The pipeline is `run_compute_scores` > `compute_scores` > `compute_{name}`

When `label_method="frc"`,  
- `run_compute_scores`: Filter per label and LOOP CALLS to next function.
- `compute_scores`: Reduce, then call next function.
- `compute_{name}`: Compute metric.

When `label_method="rfc"`,  
- `run_compute_scores`: Call next function. 
- `compute_scores`: Reduce, then filter and LOOP CALLS to next function. 
- `compute_{name}`: Compute metric.

When `label_method="rcf"`,  
- `run_compute_scores`: Call next function. 
- `compute_scores`: Reduce, then call next function.
- `compute_{name}`: Compute structure then, LOOP COMPUTES of the metric.


**Remarks**  
- The `overall` pass always goes first before per-label passes.
- `label_method` is `None` unless `--per-label` is set.
- Per-label scores are only implemented for `prdc` and `pr_curve`. Other metrics are guarded by `label_key == "overall"` and only computed in the `overall` pass, whatever the `label_method`.
- With `frc`, reduction is applied per label: each label is reduced to `min(reduced_n, n_real_y, n_gen_y)`. A label can therefore use up to `reduced_n` samples, and the side (real or generated) with more label-`y` samples is subsampled to the other's count.
- The same seed drives every reduction. With `frc`, `rng.choice(n, n)` permutes the samples even when nothing is dropped, which matters for order-dependent steps (e.g. a train/test split).
- Output structure: `frc` writes `run_scores["label-i"]` directly. `rfc` and `rcf` return `label-i` keys nested inside the metric dict, which `unpack_nested_labelwise_metric` moves up to `run_scores["label-i"]`. In all cases `aggregate_labelwise_scores` then builds `agg` and `diff`.
- Labels are assumed to be contiguous integers in `[0, n_labels-1]`: keys are `label-{idx}` with `idx` indexing `np.unique(real_labs)`.