# UC_ML: ML-Assisted Acceleration for Unit Commitment

中文 | [English](#english)

本仓库提供一个面向机组组合（Unit Commitment, UC）问题的机器学习辅助加速框架。当前版本用于受限访问和内部复现实验，因此 README 不包含未发表稿件题目、正文、回复材料或表格原文。

## 项目功能

本项目主要包含三部分：

1. **数据生成**：求解历史日和增强样本的 UC 问题，生成机组启停标签、活跃输电约束标签和线路潮流。
2. **输入聚合**：使用活跃线路 PTDF 将高维母线净负荷映射到低维虚拟母线输入。
3. **机器学习辅助优化**：训练活跃约束预测模型和机组状态预测模型，并将预测结果用于约束过滤和 warm start，加速 Gurobi 求解。

推荐入口：

```bash
python -m uc_ml --help
python -m uc_ml workflow
python -m uc_ml check
```

根目录中的旧脚本仍然保留，用于兼容已有实验和模型命名。新用户优先使用 `uc_ml` CLI。

## 目录结构

```text
UC_ML/
  uc_ml/                         # 公共 CLI 和流程元数据
  docs/                          # 架构说明和脚本映射
  case_data/                     # 外部 MATPOWER/UnitCommitment.jl 算例数据，不纳入 git
  training_data/                 # 生成的数据集和储能参数，不纳入 git
  module_save/                   # 训练好的模型权重和元数据，不纳入 git
  figures/, fig_data/            # 生成的图和训练曲线，不纳入 git
  *_parallel.py                  # 并行数据生成和实验脚本
  unit_commitment_gurobi.py      # 主 Gurobi UC 模型
  uc_gurobi_constraint_filter.py # 迭代式输电约束过滤
  ml_*.py                        # 模型定义、数据加载、训练和测试工具
```

完整文件说明见 [docs/PROJECT_MAP.md](docs/PROJECT_MAP.md)。

这些运行目录通过 `.gitkeep` 在仓库中保留空目录；同时 `python -m uc_ml check` 和 `python -m uc_ml run ...` 会在运行前自动创建缺失目录。

## 环境安装

当前维护的本地运行环境是 Conda 环境 `python39`。UC 求解需要安装 Gurobi 并配置有效许可证。

```bash
conda activate python39
pip install -r requirements.txt
```

开发模式安装：

```bash
pip install -e .
```

如果本机存在多个 Python 版本，可以直接调用 `python39` 环境中的解释器：

```bash
C:\Users\TangHaolin\anaconda3\envs\python39\python.exe -m uc_ml workflow
```

## 快速检查

检查目录、Python 依赖和 Gurobi 许可证。该命令会执行一个最小 Gurobi solve，因此可以发现许可证过期等问题：

```bash
python -m uc_ml check
```

打印流程：

```bash
python -m uc_ml workflow
```

仅打印旧脚本调用命令，不真正执行：

```bash
python -m uc_ml run --dry-run generate-base-data -- --case-name case1888rte --storage-data-name storage_data_case1888rte
```

## 推荐流程

基础训练主流程：

```text
generate_storage_data.py
  -> generate_data_parallel.py          # 365 天历史样本
  -> generate_all_data_parallel.py      # 增补增强样本
  -> add_new_input.py                   # 如训练聚合输入模型，则执行 PTDF 聚合
  -> ml_training_model.py               # 或 python -m uc_ml train
```

迁移性测试分支：

```text
generate_new_input_data.py              # 或 generate_new_input_data_parallel.py
  -> add_new_input.py                   # new-input 数据使用 base active-line source 聚合
  -> do_model_test.py                   # raw/aggregated、old/retrained 对比
```

昂贵的 UC 求解建议优先使用并行脚本。

## 实验映射

| 实验名称 | 程序中的名称或文件 | 说明 |
| --- | --- | --- |
| `case1888rte` | `case1888rte_pro` | 1888 节点主算例在代码中使用 `case1888rte_pro` |
| `case1888rte-low coupling` | `case1888rte_storage_data_case1888rte_0%_minud0.3` | 0% 储能，最小启停时间缩放 0.3 |
| `case1888rte-medium coupling` | `case1888rte_storage_data_case1888rte_50%_minud0.7` | 50% 储能，最小启停时间缩放 0.7 |
| `case300` | `storage_data_case300_60%` | case300 使用 60% 储能 |
| `case118` | `storage_data_case118_30%` | case118 使用 30% 储能 |

## 运行步骤

### 1. 准备算例和储能数据

算例 JSON 文件应放在：

```text
case_data/matpower/<case_name>/<YYYY-MM-DD>.json.gz
```

可使用 `download_data.py` 从 UnitCommitment.jl/MATPOWER 公开算例库下载 2017 年 365 个日算例文件。脚本会自动创建 `case_data/matpower/<case_name>/` 目录，已有文件默认跳过：

```bash
python -m uc_ml run download-case-data -- --case-name case118 --year 2017
python -m uc_ml run download-case-data -- --case-name case300 --year 2017
```

若程序中使用的本地算例名和上游算例名不同，可用 `--output-case-name` 指定本地目录。例如本项目中 1888 节点算例的本地目录名为 `case1888rte_pro`：

```bash
python -m uc_ml run download-case-data -- ^
  --case-name case1888rte ^
  --output-case-name case1888rte_pro ^
  --year 2017
```

也可以直接运行下载脚本：

```bash
python download_data.py --case-name case118 --year 2017
```

若自动下载不可用，请手动从以下 URL 模式下载对应日期文件，并保持相同目录结构：

```text
https://axavier.org/UnitCommitment.jl/0.4/instances/matpower/<case_name>/<YYYY-MM-DD>.json.gz
```

生成储能参数：

```bash
python -m uc_ml run make-storage
```

`generate_storage_data.py` 是 profile-based 旧脚本。发布新实验前应检查文件内当前启用的 profile。

### 2. 生成基础 UC 数据

```bash
python -m uc_ml run generate-base-data -- ^
  --case-name case1888rte ^
  --storage-data-name storage_data_case1888rte ^
  --num-workers 4 ^
  --gurobi-output-flag 0
```

典型输出：

```text
training_data/training_data_case1888rte_storage_data_case1888rte.npz
training_data/all_training_data_case1888rte_storage_data_case1888rte.npz
```

### 3. 生成强时间耦合变体

```bash
python -m uc_ml run generate-coupling-variants -- --profiles both --stage all --num-workers 4
```

只分析最小启停时间分布，不求解 UC：

```bash
python -m uc_ml run generate-coupling-variants -- --analyze-only
```

### 4. 生成增强训练数据

```bash
python -m uc_ml run augment-history-data -- --new-data-num 1000 --num-workers 4
```

该步骤从历史净负荷 max/min 构造随机样本，并重新求解 UC，得到一致的输入、机组启停、活跃约束和潮流标签。

### 5. 执行 PTDF 输入聚合

训练 base aggregated model 时，先聚合基础增强数据：

```bash
python -m uc_ml aggregate ^
  --data training_data/all_training_data_case1888rte_pro.npz ^
  --ptdf case_data/matpower/case1888rte_pro/2017-01-01.json.gz ^
  --data-num 1100 ^
  --test-num 100
```

无重训练迁移测试时，new-input 数据应复用 base 数据的 active-line source 和聚合标准化统计量：

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

### 6. 训练模型

训练 adapted Transformer 机组启停预测模型：

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

训练 DNN 活跃输电约束预测模型：

```bash
python -m uc_ml train ^
  --target constraints ^
  --model-type DNN ^
  --data all_training_data_case1888rte_pro.npz ^
  --training-num 1000 ^
  --test-num 100 ^
  --hidden-sizes 256,256,256
```

### 7. 运行复现实验

迁移性对比：

```bash
python -m uc_ml run table-ii-transferability
```

预测模型对比：

```bash
python -m uc_ml run table-iii-model-comparison
```

优化效率对比：

```bash
python -m uc_ml run table-iv-optimization -- ^
  --sample-count 20 ^
  --output table_IV_optimization_efficiency_20samples.xlsx ^
  --raw-output table_IV_optimization_efficiency_20samples_raw.csv ^
  --gurobi-output-flag 0 ^
  --gurobi-threads 1
```

固定预测可行性对比：

```bash
python -m uc_ml run table-vi-feasibility -- ^
  --max-samples 100 ^
  --gurobi-output-flag 0 ^
  --gurobi-threads 1
```

## 数据格式

新代码优先使用 `.npz`，并采用 PyTorch 友好的轴顺序：

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `input_data` | `(sample, time, feature)` | 归一化后的母线级或聚合负荷输入 |
| `output_data` | `(sample, time, generator)` | 机组启停二元标签 |
| `enable_data` | `(sample, line)` | 活跃输电约束标签 |
| `l_data` | `(sample, time, line)` | 输电线路潮流 |
| `average_load` | scalar | 将归一化输入还原为 MW 的基准 |
| `active_lines` | vector, optional | 输入聚合使用的活跃线路索引 |
| `aggregation_mean`, `aggregation_std` | vector, optional | 聚合输入每个通道的 z-score 统计量 |

旧 `.mat` 文件通常使用 `(sample, feature, time)` 轴顺序。可使用以下工具转换：

```bash
python convert_mat_to_npz.py
python convert_storage_data.py
```

## 开源和受限访问说明

大规模数据、模型权重、Excel 表格、图像、Word 文件和求解器输出不应直接提交到 git。可通过脚本重新生成，或在需要时通过私有渠道提供。

如果仓库用于受限评阅，建议保持 private，只邀请编辑或评阅人账户访问。不要在 public 仓库中提交稿件正文、回复材料、作者信息或未发表题目。

Gurobi 求解耗时较高。正式运行前建议使用 `--sample-count`、`--max-samples`、`--gurobi-output-flag 0` 和 `--gurobi-threads 1` 做小样本 smoke test。

---

## English

This repository provides an ML-assisted acceleration framework for Unit Commitment (UC). The current version is intended for anonymous review and internal reproducibility checks, so the README does not include any unpublished manuscript title, manuscript text, response letter, or table text.

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

## Open-Source and Anonymous-Review Notes

Large datasets, trained weights, Excel tables, figures, Word files, and solver outputs are not meant to be tracked in git. Regenerate them with the scripts above or provide them through private channels when needed.

For anonymous review, keep the repository private and invite only editor/reviewer accounts. Do not commit manuscript text, response letters, author-identifying information, or unpublished manuscript titles to a public repository.

Gurobi solves can be expensive. Use `--sample-count`, `--max-samples`, `--gurobi-output-flag 0`, and `--gurobi-threads 1` for smoke tests before launching full experiments.
