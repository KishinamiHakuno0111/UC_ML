# UC_ML: ML-Assisted Acceleration for Unit Commitment

This repository provides an ML-assisted acceleration framework for Unit Commitment (UC).

## What This Code Does

The framework has three main parts:

1. **Dataset generation**: solve historical and augmented UC instances to generate unit-status labels, active-constraint labels, and line-flow records.
2. **Input aggregation**: project high-dimensional bus-level net-load inputs into a compact virtual-bus representation using PTDF rows of active transmission lines.
3. **ML-assisted optimization**: train predictors for active constraints and unit commitment status, then use predictions for constraint filtering and warm-started Gurobi solves.

Recommended entrypoints:

```bash
python -m uc_ml --help
python -m uc_ml workflow
python -m uc_ml check
```

Legacy root scripts are retained for reproducibility because existing experiments and saved model names depend on these paths. New users should prefer the `uc_ml` CLI.

## Repository Layout

```text
UC_ML/
  uc_ml/                         # Public CLI and workflow metadata
  docs/                          # Architecture notes and script map
  case_data/                     # External MATPOWER/UnitCommitment.jl case data, not tracked by git
  training_data/                 # Generated datasets and storage parameters, not tracked by git
  module_save/                   # Trained models and metadata, not tracked by git
  figures/, fig_data/            # Generated figures and training curves, not tracked by git
  *_parallel.py                  # Parallel data-generation and experiment scripts
  unit_commitment_gurobi.py      # Main Gurobi UC model
  uc_gurobi_constraint_filter.py # Iterative transmission-constraint filtering
  ml_*.py                        # ML model definitions, loading, training, and testing utilities
```

See [docs/PROJECT_MAP.md](docs/PROJECT_MAP.md) for the full file map.

These runtime directories are kept in the repository with `.gitkeep` placeholders. `python -m uc_ml check` and `python -m uc_ml run ...` also create missing directories before execution.

## Installation

The maintained local runtime is the Conda environment named `python39`. Gurobi is required for UC solves and must have a valid license.

```bash
conda activate python39
pip install -r requirements.txt
```

For development:

```bash
pip install -e .
```

If multiple Python installations exist, call the `python39` interpreter explicitly:

```bash
C:\Users\TangHaolin\anaconda3\envs\python39\python.exe -m uc_ml workflow
```

## Quick Checks

Check directories, Python dependencies, and the Gurobi license. The check performs a minimal Gurobi solve so it can detect expired licenses:

```bash
python -m uc_ml check
```

Print the workflow:

```bash
python -m uc_ml workflow
```

Dry-run a legacy workflow step:

```bash
python -m uc_ml run --dry-run generate-base-data -- --case-name case1888rte --storage-data-name storage_data_case1888rte
```

## Recommended Workflow

Base training path:

```text
generate_storage_data.py
  -> generate_data_parallel.py          # 365 historical samples
  -> generate_all_data_parallel.py      # append augmented samples
  -> add_new_input.py                   # PTDF aggregation for aggregated-input models
  -> ml_training_model.py               # or python -m uc_ml train
```

Transferability branch:

```text
generate_new_input_data.py              # or generate_new_input_data_parallel.py
  -> add_new_input.py                   # aggregate new-input data with the base active-line source
  -> do_model_test.py                   # raw/aggregated and old/retrained comparisons
```

Parallel scripts are preferred for expensive UC solves.

## Important Case Mapping

| Experiment name | Program name or file | Notes |
| --- | --- | --- |
| `case1888rte` | `case1888rte_pro` | Main 1888-bus case used in the code |
| `case1888rte-low coupling` | `case1888rte_storage_data_case1888rte_0%_minud0.3` | 0% storage, minimum up/down time scaled by 0.3 |
| `case1888rte-medium coupling` | `case1888rte_storage_data_case1888rte_50%_minud0.7` | 50% storage, minimum up/down time scaled by 0.7 |
| `case300` | `storage_data_case300_60%` | case300 uses 60% storage |
| `case118` | `storage_data_case118_30%` | case118 uses 30% storage |

## Running the Workflow

### 1. Prepare Case and Storage Data

Case JSON files should be placed under:

```text
case_data/matpower/<case_name>/<YYYY-MM-DD>.json.gz
```

Use `download_data.py` to download the 365 daily files for year 2017 from the public UnitCommitment.jl/MATPOWER instance repository. The script creates `case_data/matpower/<case_name>/` automatically and skips existing files by default:

```bash
python -m uc_ml run download-case-data -- --case-name case118 --year 2017
python -m uc_ml run download-case-data -- --case-name case300 --year 2017
```

If the local case name used by this code differs from the upstream case name, pass `--output-case-name`. For example, the local 1888-bus directory is `case1888rte_pro`:

```bash
python -m uc_ml run download-case-data -- ^
  --case-name case1888rte ^
  --output-case-name case1888rte_pro ^
  --year 2017
```

You can also call the downloader directly:

```bash
python download_data.py --case-name case118 --year 2017
```

If automatic download is unavailable, manually download files that follow this URL pattern and keep the same directory layout:

```text
https://axavier.org/UnitCommitment.jl/0.4/instances/matpower/<case_name>/<YYYY-MM-DD>.json.gz
```

Generate storage parameters:

```bash
python -m uc_ml run make-storage
```

`generate_storage_data.py` is a profile-based legacy script. Check the active profile before publishing a new experiment.

### 2. Generate Base UC Data

```bash
python -m uc_ml run generate-base-data -- ^
  --case-name case1888rte ^
  --storage-data-name storage_data_case1888rte ^
  --num-workers 4 ^
  --gurobi-output-flag 0
```

Typical outputs:

```text
training_data/training_data_case1888rte_storage_data_case1888rte.npz
training_data/all_training_data_case1888rte_storage_data_case1888rte.npz
```

### 3. Generate Strong-Coupling Variants

```bash
python -m uc_ml run generate-coupling-variants -- --profiles both --stage all --num-workers 4
```

Analyze min up/down distributions without solving UC:

```bash
python -m uc_ml run generate-coupling-variants -- --analyze-only
```

### 4. Generate Augmented Training Data

```bash
python -m uc_ml run augment-history-data -- --new-data-num 1000 --num-workers 4
```

This step samples synthetic net-load profiles from historical min/max ranges and re-solves UC to obtain consistent labels.

### 5. Apply PTDF Input Aggregation

For a base aggregated model, aggregate the base augmented dataset before training:

```bash
python -m uc_ml aggregate ^
  --data training_data/all_training_data_case1888rte_pro.npz ^
  --ptdf case_data/matpower/case1888rte_pro/2017-01-01.json.gz ^
  --data-num 1100 ^
  --test-num 100
```

For no-retraining transferability tests, new-input data should reuse the base active-line source and base aggregation normalization statistics:

```bash
python -m uc_ml run generate-transfer-data -- ^
  --case-name case1888rte_pro ^
  --output-suffix newinput ^
  --num-workers 4
```

```bash
python -m uc_ml aggregate ^
  --data training_data/all_training_data_case1888rte_pro_newinput.npz ^
  --ptdf case_data/matpower/case1888rte_pro/2017-01-01.json.gz ^
  --active-source training_data/all_training_data_case1888rte_pro.npz ^
  --normalization-source training_data/all_training_data_case1888rte_pro.npz ^
  --data-num 1100 ^
  --test-num 100
```

### 6. Train Models

Train an adapted Transformer for unit-status prediction:

```bash
python -m uc_ml train ^
  --target units ^
  --model-type transformer ^
  --data all_training_data_case1888rte_pro_decreased_base.npz ^
  --training-num 1000 ^
  --test-num 100 ^
  --epochs 50000 ^
  --hidden-sizes 128 ^
  --learning-rate 0.0002
```

Train a DNN for active-constraint prediction:

```bash
python -m uc_ml train ^
  --target constraints ^
  --model-type DNN ^
  --data all_training_data_case1888rte_pro.npz ^
  --training-num 1000 ^
  --test-num 100 ^
  --hidden-sizes 256,256,256
```

### 7. Run Reproducibility Experiments

Transferability comparison:

```bash
python -m uc_ml run table-ii-transferability
```

Prediction-model comparison:

```bash
python -m uc_ml run table-iii-model-comparison
```

Optimization-efficiency comparison:

```bash
python -m uc_ml run table-iv-optimization -- ^
  --sample-count 20 ^
  --output table_IV_optimization_efficiency_20samples.xlsx ^
  --raw-output table_IV_optimization_efficiency_20samples_raw.csv ^
  --gurobi-output-flag 0 ^
  --gurobi-threads 1
```

Feasibility comparison for fixed predictions:

```bash
python -m uc_ml run table-vi-feasibility -- ^
  --max-samples 100 ^
  --gurobi-output-flag 0 ^
  --gurobi-threads 1
```

### Additional KNN-50 and Multi-Scenario Comparisons

Two optional entry points extend the comparisons without replacing the scripts above:

- `python -m uc_ml.knn50 --config docs/comparison_configs/knn50.json --smoke`: KNN-50 fixed-schedule candidates for 118/300/1354pegase and low/medium/strong coupling, with eight workers, paired accuracy/cost comparisons and measured-eight versus ideal-fifty timing.
- `python -m uc_ml.transfer_scenarios train-evaluate --config docs/comparison_configs/transfer.json`: compare frozen/retrained raw and aggregated models on frozen 900/100/100 scenario cohorts; save four accuracy curves and frozen-aggregation-gain heatmaps.

Supply the datasets and checkpoints and adjust the example paths first. See
[Comparison Benchmarks](docs/COMPARISON_BENCHMARKS.md) for input formats, the opt-in
accelerated training recipe, numerical-error handling, resume commands and tests.
The new entry points use the repository's datasets and checkpoints and save results under `fig_data/`.

## Data Format

New code prefers `.npz` files with PyTorch-friendly axes:

| Field | Shape | Meaning |
| --- | --- | --- |
| `input_data` | `(sample, time, feature)` | Normalized bus-level or aggregated load input |
| `output_data` | `(sample, time, generator)` | Binary unit commitment labels |
| `enable_data` | `(sample, line)` | Active transmission constraint labels |
| `l_data` | `(sample, time, line)` | Transmission line flow values |
| `average_load` | scalar | Normalization base used to recover MW load |
| `active_lines` | vector, optional | Active line indices used by aggregation |
| `aggregation_mean`, `aggregation_std` | vector, optional | Per-channel z-score statistics used by aggregated inputs |

Legacy `.mat` files usually use `(sample, feature, time)` axes. Convert them with:

```bash
python convert_mat_to_npz.py
python convert_storage_data.py
```

## Generated Artifacts and Runtime Notes

The repository keeps source code, configuration, and lightweight placeholder directories only. Case files, generated datasets, trained weights, evaluation outputs, figures, and solver logs are produced by the workflow scripts as needed and are written to `case_data/`, `training_data/`, `module_save/`, `fig_data/`, `figures/`, or the current working directory. These generated artifacts are excluded by `.gitignore`.

Gurobi solves can be expensive. Use `--sample-count`, `--max-samples`, `--gurobi-output-flag 0`, and `--gurobi-threads 1` for smoke tests before launching full experiments.
