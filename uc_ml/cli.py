"""Command line interface for the paper codebase."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

from .paper_workflow import WORKFLOW_STEPS, get_step, workflow_sections
from .paths import project_root
from .training import TrainConfig, train_from_config


def _print_workflow() -> None:
    by_section = {section: [] for section in workflow_sections()}
    for step in WORKFLOW_STEPS:
        by_section[step.paper_section].append(step)

    for section, steps in by_section.items():
        print(f"\n{section}")
        print("-" * len(section))
        for step in steps:
            script = step.script or "uc_ml CLI"
            cost = "expensive" if step.expensive else "light"
            print(f"{step.id:28} {script:46} {cost}")
            print(f"  {step.purpose}")
            if step.outputs:
                print(f"  outputs: {', '.join(step.outputs)}")
            if step.notes:
                print(f"  note: {step.notes}")


def _run_step(args: argparse.Namespace) -> int:
    step = get_step(args.step)
    if step.script is None:
        raise SystemExit(
            f"Step {step.id!r} uses a dedicated CLI command. "
            "Run `python -m uc_ml workflow` for details."
        )

    root = project_root()
    script_path = root / step.script
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    extra_args = list(args.args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    cmd = [sys.executable, str(script_path), *step.default_args, *extra_args]
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=root, check=False)
    return int(completed.returncode)


def _check_environment() -> int:
    root = project_root()
    print(f"project_root: {root}")
    required_dirs = ["case_data", "training_data", "module_save"]
    for dirname in required_dirs:
        path = root / dirname
        status = "OK" if path.exists() else "MISSING"
        print(f"{status:8} {dirname}/")

    packages = [
        ("numpy", True),
        ("scipy", True),
        ("torch", True),
        ("pandas", True),
        ("matplotlib", True),
        ("openpyxl", True),
        ("requests", False),
        ("pyomo", False),
        ("gurobipy", False),
    ]
    missing_required = []
    print("\nPython packages")
    for package, required in packages:
        found = importlib.util.find_spec(package) is not None
        label = "OK" if found else ("MISSING" if required else "optional missing")
        print(f"{label:16} {package}")
        if required and not found:
            missing_required.append(package)

    if missing_required:
        print("\nMissing required packages: " + ", ".join(missing_required))
        return 1

    gurobi_spec = importlib.util.find_spec("gurobipy")
    if gurobi_spec is not None:
        print("\nGurobi license")
        try:
            import gurobipy as gp

            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            model = gp.Model(env=env)
            x = model.addVar(lb=0.0, name="x")
            model.setObjective(x, gp.GRB.MINIMIZE)
            model.optimize()
            model.dispose()
            env.dispose()
            print("OK               minimal solve")
        except Exception as exc:
            print(f"ERROR            minimal solve failed: {exc}")
            return 1

    return 0


def _aggregate(args: argparse.Namespace) -> int:
    from add_new_input import aggregate_from_file

    aggregate_from_file(
        data_path=args.data,
        ptdf_data_path=args.ptdf,
        active_constraints_source=args.active_source,
        data_num=args.data_num,
        test_num=args.test_num,
        use_npz=not args.mat,
        use_normalization=not args.no_normalization,
        normalization_source=args.normalization_source,
        save_path=args.output,
    )
    return 0


def _train(args: argparse.Namespace) -> int:
    config = TrainConfig(
        data=args.data,
        data_dir=args.data_dir,
        model_type=args.model_type,
        target=args.target,
        training_num=args.training_num,
        test_num=args.test_num,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed,
        data_is_random=not args.no_shuffle,
        epochs=args.epochs,
        stop_gap=args.stop_gap,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_sizes=args.hidden_sizes,
        dropout_prob=args.dropout_prob,
        use_scheduler=not args.no_scheduler,
        warmup_ratio=args.warmup_ratio,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_steps=args.accumulate_steps,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        reliability_low_gate=args.reliability_low_gate,
        reliability_high_gate=args.reliability_high_gate,
        output_dir=args.output_dir,
        fig_dir=args.fig_dir,
        result_xlsx=args.result_xlsx,
        cutted_by_time=args.cutted_by_time,
    )
    train_from_config(config)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uc_ml",
        description="Workflow CLI for the UC_ML project.",
    )
    sub = parser.add_subparsers(dest="command")

    workflow = sub.add_parser("workflow", help="Print the paper-aligned workflow map.")
    workflow.set_defaults(func=lambda _args: (_print_workflow() or 0))

    check = sub.add_parser("check", help="Check local directories and Python dependencies.")
    check.set_defaults(func=lambda _args: _check_environment())

    run = sub.add_parser("run", help="Run a legacy workflow step by id.")
    run.add_argument("step", choices=[step.id for step in WORKFLOW_STEPS if step.script])
    run.add_argument("args", nargs=argparse.REMAINDER, help="Extra args passed to the legacy script. Use -- before them.")
    run.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    run.set_defaults(func=_run_step)

    aggregate = sub.add_parser("aggregate", help="Run PTDF input aggregation with explicit paths.")
    aggregate.add_argument("--data", required=True, help="Input training .npz/.mat file.")
    aggregate.add_argument("--ptdf", required=True, help="Case JSON.gz used to compute PTDF.")
    aggregate.add_argument("--active-source", default=None, help="Optional data file used to detect active lines.")
    aggregate.add_argument(
        "--normalization-source",
        default=None,
        help="Optional reference data file used for aggregation z-score statistics.",
    )
    aggregate.add_argument("--data-num", type=int, default=None, help="Samples considered when detecting active lines.")
    aggregate.add_argument("--test-num", type=int, default=100)
    aggregate.add_argument("--output", default=None, help="Optional output .npz/.mat path.")
    aggregate.add_argument("--mat", action="store_true", help="Write MATLAB .mat instead of .npz.")
    aggregate.add_argument("--no-normalization", action="store_true", help="Disable per-channel z-score normalization.")
    aggregate.set_defaults(func=_aggregate)

    train = sub.add_parser("train", help="Train a units or constraints model without editing globals.")
    train.add_argument("--data", required=True, help="Dataset file name or path, with or without .npz.")
    train.add_argument("--data-dir", default="training_data")
    train.add_argument("--model-type", choices=["DNN", "KAN", "transformer", "LSTM_new", "BiLSTM"], default="transformer")
    train.add_argument("--target", choices=["units", "constraints"], default="units")
    train.add_argument("--training-num", type=int, default=1000)
    train.add_argument("--test-num", type=int, default=100)
    train.add_argument("--val-ratio", type=float, default=0.1)
    train.add_argument("--random-seed", type=int, default=42)
    train.add_argument("--no-shuffle", action="store_true")
    train.add_argument("--epochs", type=int, default=50000)
    train.add_argument("--stop-gap", type=int, default=50)
    train.add_argument("--learning-rate", type=float, default=0.0002)
    train.add_argument("--weight-decay", type=float, default=0.001)
    train.add_argument("--hidden-sizes", default=None, help="Integer for Transformer/BiLSTM or comma/list for DNN/KAN.")
    train.add_argument("--dropout-prob", type=float, default=0.3)
    train.add_argument("--no-scheduler", action="store_true")
    train.add_argument("--warmup-ratio", type=float, default=0.1)
    train.add_argument("--gradient-clip-val", type=float, default=1.0)
    train.add_argument("--accumulate-steps", type=int, default=1)
    train.add_argument("--early-stopping-patience", type=int, default=50)
    train.add_argument("--early-stopping-metric", choices=["val_loss", "train_loss"], default="val_loss")
    train.add_argument("--reliability-low-gate", type=float, default=0.1)
    train.add_argument("--reliability-high-gate", type=float, default=0.9)
    train.add_argument("--output-dir", default="module_save")
    train.add_argument("--fig-dir", default="fig_data")
    train.add_argument("--result-xlsx", default="test_result.xlsx")
    train.add_argument("--cutted-by-time", action=argparse.BooleanOptionalAction, default=None)
    train.set_defaults(func=_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
