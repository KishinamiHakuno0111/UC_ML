"""Workflow metadata for the UC_ML project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class WorkflowStep:
    """A script-level workflow step."""

    id: str
    paper_section: str
    purpose: str
    script: str | None
    default_args: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    notes: str = ""
    expensive: bool = True


WORKFLOW_STEPS: tuple[WorkflowStep, ...] = (
    WorkflowStep(
        id="download-case-data",
        paper_section="Data source",
        purpose="Download UnitCommitment.jl/MATPOWER daily case JSON files.",
        script="download_data.py",
        outputs=("case_data/matpower/<case>/*.json.gz",),
        notes="Use --case-name for the upstream case and --output-case-name when a local alias is needed.",
    ),
    WorkflowStep(
        id="make-storage",
        paper_section="UC model with ESS",
        purpose="Create storage parameter files used by the UC formulation.",
        script="generate_storage_data.py",
        outputs=("training_data/storage_data_*.mat", "training_data/storage_data_*.npz"),
        notes="Legacy script is profile-based and currently writes the active hard-coded profile.",
        expensive=False,
    ),
    WorkflowStep(
        id="generate-base-data",
        paper_section="Dataset generation",
        purpose="Solve 365 daily UC instances and save base training labels.",
        script="generate_data_parallel.py",
        outputs=("training_data/training_data_<case>_<storage>.npz", "training_data/all_training_data_<case>_<storage>.npz"),
    ),
    WorkflowStep(
        id="augment-history-data",
        paper_section="Data augmentation",
        purpose="Append randomized load samples derived from historical min/max ranges.",
        script="generate_all_data_parallel.py",
        outputs=("training_data/all_training_data_*.npz",),
    ),
    WorkflowStep(
        id="aggregate-input",
        paper_section="Input aggregation",
        purpose="Apply active-line PTDF aggregation and residual channel construction before aggregated-model training.",
        script=None,
        outputs=("training_data/*_decreased_base*.npz",),
        notes="For no-retraining transfer tests, reuse both the base active-line source and base normalization source.",
        expensive=False,
    ),
    WorkflowStep(
        id="train-model",
        paper_section="ML models",
        purpose="Train units or active-constraint prediction models.",
        script=None,
        outputs=("module_save/*.pth", "module_save/*_metadata.json", "module_save/*_loss.csv"),
        notes="Use `python -m uc_ml train ...` instead of editing ml_training_model.py globals.",
    ),
    WorkflowStep(
        id="generate-coupling-variants",
        paper_section="Strong intertemporal coupling cases",
        purpose="Generate case1888rte low/medium coupling variants by scaling min up/down times.",
        script="generate_storage_min_time_data_parallel.py",
        default_args=("--profiles", "both", "--stage", "all"),
        outputs=("training_data/*_minud*.npz",),
        notes="Case-study branch; not required before training generic base-case models.",
    ),
    WorkflowStep(
        id="generate-transfer-data",
        paper_section="Transferability test scenario",
        purpose="Generate new-input data with changed spatial load/renewable injections and re-solved UC labels.",
        script="generate_new_input_data_parallel.py",
        outputs=("training_data/all_training_data_*_newinput.npz",),
        notes="Transferability branch after the base dataset/model exists; it is not a prerequisite for base training.",
    ),
    WorkflowStep(
        id="table-ii-transferability",
        paper_section="Transferability",
        purpose="Run the four transferability cases used by the comparison script.",
        script="do_model_test.py",
        outputs=("do_model_test_summary.xlsx", "test_result.xlsx"),
    ),
    WorkflowStep(
        id="table-iii-model-comparison",
        paper_section="Prediction performance",
        purpose="Compare DNN, BiLSTM, Transformer, and aggregated Transformer.",
        script="compare_models.py",
        outputs=("model_comparison_result.xlsx", "model_comparison.png"),
    ),
    WorkflowStep(
        id="table-iv-optimization",
        paper_section="Optimization efficiency",
        purpose="Generate optimization-efficiency results for constraint filtering and warm starts.",
        script="generate_table_iv_optimization_efficiency.py",
        outputs=("table_IV_optimization_efficiency.xlsx", "table_IV_optimization_efficiency_raw.csv"),
    ),
    WorkflowStep(
        id="table-vi-feasibility",
        paper_section="Confidence filtering",
        purpose="Compare full prediction fixing with high-confidence subset fixing.",
        script="count_infeasible_binary_predictions.py",
        default_args=("--all-default-cases",),
        outputs=("binary_feasibility_transformer_cases.xlsx", "binary_feasibility_transformer_cases_summary.csv"),
    ),
)


def get_step(step_id: str) -> WorkflowStep:
    """Return a workflow step by id."""
    for step in WORKFLOW_STEPS:
        if step.id == step_id:
            return step
    valid = ", ".join(step.id for step in WORKFLOW_STEPS)
    raise KeyError(f"Unknown workflow step {step_id!r}. Valid steps: {valid}")


def workflow_sections() -> Iterable[str]:
    """Yield paper sections in workflow order without duplicates."""
    seen: set[str] = set()
    for step in WORKFLOW_STEPS:
        if step.paper_section not in seen:
            seen.add(step.paper_section)
            yield step.paper_section
