# Additional Comparison Benchmarks

These modules provide KNN-50 and four-way multi-scenario transfer comparisons.
Run commands from the repository root with Python 3.9 or newer. Paths in JSON
configurations are relative to the configuration file. Inputs use the repository's
datasets and checkpoints; results are written to `fig_data/`.

## KNN-50

`docs/comparison_configs/knn50.json` includes case118, case300, case1354pegase and
low/medium/strong case1888rte coupling settings. Keep only the desired case entries
in a copy of the config. Provide the original dataset, storage, network and trusted
checkpoints. Check each DNN metadata file: `dnn_layout` is `per_period` when
`cutted_by_time` is true, otherwise `flattened`. Optional `models` entries can be
omitted to run KNN alone. Supplied neural checkpoints are used for inference.

```bash
python -m uc_ml.knn50 --config docs/comparison_configs/knn50.json --smoke
python -m uc_ml.knn50 --config docs/comparison_configs/knn50.json --resume --resolve-numeric
```

The smoke pass solves one full 50-candidate test and one reference per case. These
tests count toward the final 100. Solving requires a Gurobi license.
One global pool has eight workers; each solve
uses one Gurobi thread, Seed=0 and MIPGap=1e-4 without a time limit.

1. Shuffle IDs 0..1099 with Python `random.Random(42)`, then take 900 training,
   100 validation and 100 test IDs, matching the archived loader.
2. Retrieve the 50 training-only nearest neighbors by squared Euclidean distance
   on the full raw, normalized 36-period load trajectory. Break ties by source ID.
3. Fix each neighbor's complete commitment `u`; derive startup/shutdown `y/z`
   from `u` and the network initial state. Reoptimize continuous variables.
4. Start constraint generation with no enabled lines. Retain the original model
   construction, top-10/gate-0.7 addition rule, slack rule and final line checks.
5. After **all 50** candidates terminate, choose the minimum objective among
   `legacy_accepted` candidates. All-50 failure is included in coverage, with null
   accuracy and cost.
6. Solve one unrestricted same-model reference separately. Cost deviation is
   `100*(candidate-reference)/abs(reference)` in percent, including negative values.

`summary.json` reports failure counts, pooled-entry (micro) and per-test mean
(macro) KNN accuracies, and DNN/BiLSTM/Transformer on both all tests and the exact
KNN-success subset. Changed-generator accuracy includes **all 36 periods** of
generators that change state. An empty changed-generator denominator is null.
KNN training accuracy and high-confidence accuracy are marked N/A because the
method retrieves complete schedules directly from the training set.

`actual8_seconds` includes neighbor lookup, the entire candidate batch (including
failed candidates) and selection. `ideal50_seconds` is lookup + maximum measured
candidate duration + selection, giving the theoretical 50-worker estimate.
Reference time is recorded separately. Outputs include CPU/worker seconds for
every attempt and the Gurobi runtime version.

`--resume` uses an OS lock and immutable input/settings bindings, skips completed
keys and preserves partial attempts. A partially executed candidate batch has
null clean latency; it is still included in final accuracy/coverage/cost. With
`--resolve-numeric`, only solver status NUMERIC (12) with zero solutions is
deferred. After the first pass, one separate same-model attempt uses Method=2,
BarHomogeneous=1, NumericFocus=3, Aggregate=0, ScaleFlag=2, Presolve=0, Crossover=0.
Model matrix/RHS/objective/bounds/types are checked before and after parameter
application and against common original rounds. Original attempts are preserved.
Other errors stop submission and drain in-flight work for inspection.

## Multi-Scenario Transfer Comparison

This entry point takes **frozen, labeled cohorts** from the existing UC labeling
pipeline and performs model training, evaluation and plotting.
The pure functions in `uc_ml.transfer_fields` expose the bounded AR(1), spatial
fields, weighted box projection and perturbation formula for input generation.
Use the frozen source groups, weights, shared random fields and split IDs across cells.

For factor `f`, the system mean-shift magnitude is `delta=0.05*f` (2.5% at 0.5,
20% at 4). Cell signs `a,b` belong to {-1,0,+1}. With source total load `L`,
zero-node weights `w`, bounded fields `z,s` and load-weighted zero-mean `eta`:

```text
c_zero = w * f * (0.025*b + 0.05*z)
c_active = 0.05*f*a + 0.01*f*s - sum(c_zero)
new_zero_load = L * c_zero
new_active_load = original_active_load * (1 + c_active + f*eta)
```

Zero amplitude has one condition and is exactly the original input. Factors above
five use the signed-load domain. The factor scales the mean shift and random-field
amplitudes together according to the formula above.

### Input Contract

Copy `docs/comparison_configs/transfer.json` and set the input paths. `source_data`
provides the original `average_load` scalar. `preprocessing.npz` contains the source
`matrix` (buses x selected lines), `mean` and `std` (selected lines + residual).
Use the mapping and source statistics associated with the frozen checkpoints.
`legacy_projection` applies their original axis-reshape convention to target inputs.

Each `cohort_root/amplitude_{f}/{cell}/` contains:

| File | Contents |
| --- | --- |
| `cohort.npz` | For each prefix `train`, `val`, `test`: `{prefix}_input_MW` (N,36,buses), `{prefix}_labels` (N,36,generators), `{prefix}_source_ids` (N, integer) |
| `attempts.json` | List of `{split, slot, attempt, source_id, status}` for **every** attempt, including failures; attempt numbering starts at zero |

N is exactly 900/100/100. Source IDs must remain within the original corresponding
split. Repeated source IDs from accepted replacements keep their separate slots.
A slot's contiguous sequence ends at its first `legacy_accepted`;
previous statuses are `no_answer` or `infeasible`. Convert historical
`legacy_no_answer` to `no_answer` when exporting the manifest. Include all failed
attempts and export cohorts with passed upstream label checks. The comparison
validates the bundle dimensions, source splits and first-acceptance sequences.

The regular layout uses `a_minus_b_minus` through `a_plus_b_plus`. `factors` expands
to all nine conditions; factor zero expands only to `a_zero_b_zero`. Alternatively,
use an explicit `cohorts` list with `factor`, `cell`, `data`, `attempts` and optional
`retrained_raw` / `retrained_aggregated` checkpoint paths to compare already trained
models without training again.

```bash
python -m uc_ml.transfer_scenarios train-evaluate --config docs/comparison_configs/transfer.json
# Reuse completed models and cells after an orderly interruption:
python -m uc_ml.transfer_scenarios train-evaluate --config docs/comparison_configs/transfer.json --resume
# With supplied target checkpoints, or after training above:
python -m uc_ml.transfer_scenarios evaluate --config docs/comparison_configs/transfer.json --resume
python -m uc_ml.transfer_scenarios plot --config docs/comparison_configs/transfer.json
```

Each missing target model starts from seed42 with the repository Transformer,
hidden128, full-batch900 FP32, Adam lr=0.0002/wd=0.001, BCE, clip1. Maximum 50,000
steps; warmup5,000, cosine denominator45,000, final lr ratio0.01. Validation every50
steps, patience50 checks; absolute min_delta1e-5 affects patience only. The actual
minimum validation-loss weights are restored. Training uses CUDA, source-frozen
preprocessing and validation-only model selection. This accelerated recipe is
specific to the new entry point. Completed models are reused; incomplete training
is retained and requires an explicit new run.

Saved probabilities generate both overall and changed-generator accuracy, with
history/uniform subsets. Curves use equal weights across the nine conditions;
heatmaps show **frozen aggregated minus frozen raw** in percentage points. Outputs
include all available amplitudes and an additional through-factor4 view, with
mean-shift-percentage axes. Plotting requires complete conditions at each amplitude.
JSON summaries, PNGs and PDFs are written to the output directory. Audit records
contain the input bindings, per-cell checks and saved-probability checks.

## Lightweight Tests

```bash
python -m unittest discover -s tests -p "test_comparison*.py"
```

The default tests use synthetic arrays. Set `UC_ML_TEST_GUROBI=1` to include small
Gurobi integration tests with synthetic networks.
