# Project Map

This document maps the Python files in the repository to the reproducibility workflow. It is based on a full static read of all `.py` files in the project root.

## Workflow

The workflow has two related paths. The base training path is:

```text
generate_storage_data.py
  -> generate_data_parallel.py
  -> generate_all_data_parallel.py
  -> add_new_input.py                      # only for aggregated-input models
  -> ml_training_model.py / python -m uc_ml train
```

The transferability branch is:

```text
generate_new_input_data.py / generate_new_input_data_parallel.py
  -> add_new_input.py                      # reuse base active-line source for no-retraining tests
  -> do_model_test.py
```

`generate_new_input_data.py` is therefore not a prerequisite for training the base model. It creates the modified-input dataset used to test transferability and retraining comparisons.

The local maintained runtime is the Conda environment named `python39`.

1. **UC formulation with storage and transmission constraints**
   - Main solver: `unit_commitment_gurobi.py`
   - Iterative active-constraint filtering: `uc_gurobi_constraint_filter.py`
   - Legacy CPLEX/Pyomo solver: `unit_commitment_with_line_limit_save_cplex.py`

2. **Dataset generation**
   - Historical 365-day samples: `generate_data.py`, `generate_data_parallel.py`
   - Augmented load samples: `generate_all_data.py`, `generate_all_data_parallel.py`
   - New-input transferability samples: `generate_new_input_data.py`, `generate_new_input_data_parallel.py`
   - Strong-coupling case1888rte variants: `generate_storage_min_time_data_parallel.py`
   - Storage parameters: `generate_storage_data.py`, `convert_storage_data.py`

3. **Input aggregation**
   - PTDF active-line aggregation: `add_new_input.py`
   - Public command: `python -m uc_ml aggregate ...`

4. **ML training and testing**
   - Model definitions: `ml_module_define.py`
   - Data loading, training loop, metrics, model metadata: `ml_basic_functions.py`
   - Legacy training script: `ml_training_model.py`
   - Public training command: `python -m uc_ml train ...`
   - Transferability evaluation: `do_model_test.py`
   - Model comparison: `compare_models.py`

5. **Optimization and feasibility experiments**
   - Table IV optimization efficiency: `generate_table_iv_optimization_efficiency.py`
   - Table VI feasibility of fixed predictions: `count_infeasible_binary_predictions.py`
   - High-confidence fixing vs warm start optimality: `compare_high_accuracy_fix_vs_warmstart_optimality.py`
   - Early warm-start/reliability scripts: `warm_start_accelerate.py`, `reliable_test.py`, `ml_model_test_functions.py`

6. **Utilities and legacy checks**
   - Download external case data: `download_data.py`
   - Convert `.mat` training data and metadata: `convert_mat_to_npz.py`
   - Inspect old `.mat` data: `read_training_data.py`

## File-by-File Responsibilities

| File | Role | Workflow connection | Public status |
| --- | --- | --- | --- |
| `add_new_input.py` | Builds active-line PTDF projection and residual channel; saves aggregated `.npz`/`.mat` datasets with optional active-line and normalization metadata. | Input aggregation and transferability tests. | Core, wrapped by `python -m uc_ml aggregate`. |
| `compare_high_accuracy_fix_vs_warmstart_optimality.py` | Compares baseline, high-confidence warm start, and high-confidence variable fixing on the same UC samples. | Supports warm-start-only vs aggressive fixing optimality checks. | Experiment script. |
| `compare_models.py` | Evaluates DNN, BiLSTM, Transformer, and aggregated Transformer models; writes Excel and plot outputs. | Model prediction comparison. | Experiment script. |
| `convert_mat_to_npz.py` | Converts legacy `.mat` training data to `.npz` and old model metadata `.mat` to JSON. | Data format cleanup for open-source release. | Utility. |
| `convert_storage_data.py` | Converts `storage_data*.mat` files to `.npz` while keeping storage array shapes compatible. | Storage parameter compatibility. | Utility. |
| `count_infeasible_binary_predictions.py` | Tests full prediction fixing vs high-confidence subset fixing and writes raw/summary CSV/XLSX outputs. | Feasibility comparison. | Core experiment script. |
| `download_data.py` | Downloads daily UnitCommitment.jl/MATPOWER JSON.gz files. | Raw case-data preparation. | Legacy utility; case name is hard-coded. |
| `do_model_test.py` | Runs four transferability comparisons: old/new, raw/aggregated model-data pairings. | Transferability comparison. | Core experiment script. |
| `generate_all_data.py` | Serial version for augmenting base datasets with randomized load samples and UC re-solves. | Data augmentation. | Legacy serial script. |
| `generate_all_data_parallel.py` | Parallel augmented-data generator with optional predicted initial constraints. | Data augmentation at scale. | Preferred data-generation script. |
| `generate_data.py` | Serial 365-day historical dataset generator; computes PTDF and solves daily UC instances. | Base dataset generation. | Legacy serial script. |
| `generate_data_parallel.py` | Parallel 365-day dataset generator with argparse and optional active-constraint model initialization. | Base dataset generation at scale. | Preferred data-generation script. |
| `generate_new_input_data.py` | Serial generator for transferability data with zero-bus renewable injection and nonzero-bus load growth compensation. | New-input transferability scenario. | Legacy serial script. |
| `generate_new_input_data_parallel.py` | Parallel new-input transferability dataset generator. | New-input transferability experiments. | Preferred data-generation script. |
| `generate_storage_data.py` | Creates manually configured storage parameter `.mat` and `.npz` files. | ESS parameters. | Utility, profile currently selected in file. |
| `generate_storage_min_time_data_parallel.py` | Generates case1888rte low/medium coupling variants, trains constraint predictors, and augments samples. | Strong intertemporal coupling variants. | Core experiment generator. |
| `generate_table_iv_optimization_efficiency.py` | Runs enable-all, no-initial, predicted-constraints, full warm start, high-confidence warm start, and high-confidence fixing strategies. | Optimization efficiency comparison. | Core experiment script. |
| `ml_basic_functions.py` | Loads `.npz` data, builds train/val/test splits, trains models, evaluates accuracy/confidence, loads metadata. | Shared ML infrastructure for all tables. | Core library module. |
| `ml_model_test_functions.py` | Uses constraint predictions to initialize constraint filtering over test samples. | Early active-constraint prediction evaluation. | Legacy helper. |
| `ml_module_define.py` | Defines DNN, KAN, BiLSTM, BiLSTM_new, and TransformerModel. | Model architecture. | Core library module. |
| `ml_training_model.py` | Legacy top-level training script controlled by global variables. | Training units and constraints models. | Kept for compatibility; prefer `python -m uc_ml train`. |
| `read_training_data.py` | Reads old `.mat` training files and compares output/input differences. | Data sanity checking. | Legacy utility. |
| `reliable_test.py` | Plots prediction confidence distribution and builds reliable warm-start dictionaries. | Early confidence filtering analysis. | Legacy exploratory script. |
| `uc_gurobi_constraint_filter.py` | Iteratively solves UC with a subset of line constraints, computes flows, adds overloaded lines by overload rate. | Active-constraint filtering. | Core solver wrapper. |
| `unit_commitment_gurobi.py` | Main Gurobi UC MILP with storage, reserves, ramping, min up/down times, line flows, warm starts, and fixing. | UC model and optimization solves. | Core solver. |
| `unit_commitment_with_line_limit_save_cplex.py` | Older Pyomo/CPLEX UC solver. | Historical implementation before the Gurobi workflow. | Legacy reference. |
| `warm_start_accelerate.py` | Early monolithic warm-start/fixing comparison script. | Prototype for Table IV-style acceleration. | Legacy exploratory script. |

## Recommended Public Entry Points

Use the `uc_ml` CLI for new runs:

```bash
python -m uc_ml workflow
python -m uc_ml check
python -m uc_ml aggregate --help
python -m uc_ml train --help
python -m uc_ml run --help
```

Use direct legacy scripts only when reproducing an existing experiment exactly, especially if the saved model name in `module_save/` depends on the old naming convention.

## Important Conventions

- `.npz` training arrays use `(sample, time, feature)` order.
- Legacy `.mat` training arrays use `(sample, feature, time)` order.
- `average_load` converts normalized inputs back to MW for UC solving.
- Aggregated input dimensions must match the active-constraint source used during model training.
- Transferability tests for old aggregated models should aggregate new-input data using the original base active-line set, not the new-input data's own active-line set.
- If z-score normalization is enabled for aggregated inputs, no-retraining transferability tests should also reuse the base aggregation normalization statistics via `--normalization-source`.
- `case_data/`, `training_data/`, `module_save/`, generated tables, and figures are not meant to be tracked in git.
