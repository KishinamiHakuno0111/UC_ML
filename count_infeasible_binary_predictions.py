"""
count_infeasible_binary_predictions.py

Purpose
-------
Evaluate feasibility of NN-predicted binary commitment decisions on a UC test split,
with two fixing strategies on the same samples:

1) full_fix:
   Fix all predicted binary variables (u, y, z).
2) high_conf_fix:
   Only fix high-confidence variables where:
     prob >= reliability_high_gate OR prob <= reliability_low_gate

The script uses the same acceleration style as generate_new_input.py:
- predict initial active constraints by a constraints model
- call run_constraint_filter(...) to iteratively add violated constraints

No-solution rule
----------------
- If run_constraint_filter returns no_answer=True, the case is counted as no-solution.
- If solve error contains "Unable to retrieve attribute 'x'", it is also counted as
  no-solution (as requested).
"""

import argparse
import csv
import gzip
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from ml_basic_functions import ml_load_model
from uc_gurobi_constraint_filter import run_constraint_filter


MODEL_DIR = "module_save"
CASE_DIR = "case_data"
DATA_DIR = "training_data"
DEFAULT_RELIABILITY_GATES = (0.1, 0.9)
DEFAULT_CASE_RELIABILITY_GATES = {
    "case1888rte": (0.05, 0.95),
}


@dataclass(frozen=True)
class DefaultCaseSpec:
    case_label: str
    case_json: str
    units_model_name: str
    constraints_model_name: str
    storage_data: Optional[str]
    use_storage: bool = True
    line_limit_rate: float = 1.0


DEFAULT_CASE_SPECS = [
    DefaultCaseSpec(
        case_label="case118",
        case_json=os.path.join(CASE_DIR, "matpower", "case118", "2017-02-01.json.gz"),
        units_model_name="units_transformer_all_training_data_case118_storage_data_case118_30%_random_1000_50000_128_0.0002",
        constraints_model_name="constraints_DNN_all_training_data_case118_storage_data_case118_30%.npz_random_1000_50000_[256, 256, 256]_0.0002",
        storage_data=os.path.join(DATA_DIR, "storage_data_case118_30%"),
    ),
    DefaultCaseSpec(
        case_label="case300",
        case_json=os.path.join(CASE_DIR, "matpower", "case300", "2017-01-01.json.gz"),
        units_model_name="units_transformer_all_training_data_case300_storage_data_case300_60%_random_1000_50000_128_0.0002",
        constraints_model_name="constraints_DNN_all_training_data_case300_storage_data_case300_60%.npz_random_1000_50000_[256, 256, 256]_0.0002",
        storage_data=os.path.join(DATA_DIR, "storage_data_case300_60%"),
    ),
    DefaultCaseSpec(
        case_label="case1354pegase",
        case_json=os.path.join(CASE_DIR, "matpower", "case1354pegase", "2017-01-01.json.gz"),
        units_model_name="units_transformer_all_training_data_case1354pegase_storage_data_case1354pegase_random_1000_50000_128_0.0002",
        constraints_model_name="constraints_DNN_all_training_data_case1354pegase_storage_data_case1354pegase.npz_random_1000_50000_[256, 256, 256]_0.0002",
        storage_data=os.path.join(DATA_DIR, "storage_data_case1354pegase"),
    ),
    DefaultCaseSpec(
        case_label="case1888rte",
        case_json=os.path.join(CASE_DIR, "matpower", "case1888rte_pro", "2017-01-01.json.gz"),
        units_model_name="units_transformer_all_training_data_case1888rte_pro_random_1000_50000_128_0.0002",
        constraints_model_name="constraints_DNN_all_training_data_case1888rte_pro_random_1000_50000_[256, 256, 256]_0.0002",
        storage_data=os.path.join(DATA_DIR, "storage_data_case1888rte"),
    ),
]


def _replace_ext(path: str, new_ext: str) -> str:
    base, _ = os.path.splitext(path)
    return base + new_ext


def _load_array_dict(data_path: str) -> Dict[str, np.ndarray]:
    """
    Load .npz (preferred) or fallback to .mat with same stem.
    """
    npz_path = _replace_ext(data_path, ".npz")
    if os.path.exists(npz_path):
        return dict(np.load(npz_path, allow_pickle=True))
    if os.path.exists(data_path):
        import scipy.io
        return scipy.io.loadmat(data_path)
    raise FileNotFoundError(f"Neither {npz_path} nor {data_path} exists.")


def _to_stb_input(d: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Return input_data in (S, T, B)-style shape.
    """
    x = d["input_data"]
    if x.ndim != 3:
        raise ValueError(f"input_data ndim must be 3, got {x.ndim}.")

    if "output_data" in d and d["output_data"].ndim == 3:
        y = d["output_data"]
        t_candidate = max(y.shape[1], y.shape[2])
        if x.shape[1] == t_candidate:
            return x

    if x.shape[1] > x.shape[2]:
        return np.transpose(x, (0, 2, 1))
    return x


def _read_average_load(d: Dict[str, np.ndarray]) -> float:
    return float(np.asarray(d["average_load"]).item())


def _get_case_info(data: Dict) -> Tuple[int, int, int, np.ndarray]:
    """
    Return NT, NG, NL, init_u.
    """
    _, units = zip(*data["Generators"].items())
    _, lines = zip(*data["Transmission lines"].items()) if "Transmission lines" in data else ([], [])

    time_horizon = data["Parameters"].get("Time horizon (h)") or data["Parameters"].get("Time (h)")
    time_step = data["Parameters"].get("Time step (min)", 60)
    if 60 % time_step != 0:
        raise ValueError(f"Unsupported time_step={time_step}.")
    time_multiplier = int(60 / time_step)

    nt = time_horizon * time_multiplier
    ng = len(units)
    nl = len(lines)

    initial_status = np.array([units[i]["Initial status (h)"] * time_multiplier for i in range(ng)])
    init_u = (initial_status > 0).astype(np.int32)
    return nt, ng, nl, init_u


def _load_storage(storage_path: str) -> Dict[str, np.ndarray]:
    if not storage_path:
        return {}
    return _load_array_dict(storage_path)


def _build_split_indices(
    total_samples: int,
    training_num: int,
    test_num: int,
    val_ratio: float,
    data_is_random: bool,
    seed: int,
    split: str,
) -> List[int]:
    """
    Reproduce project train/val/test split style.
    """
    usable = training_num + test_num
    if total_samples < usable:
        raise ValueError(
            f"Dataset has {total_samples} samples, but training_num+test_num={usable}."
        )

    indices = list(range(usable))
    if data_is_random:
        rng = random.Random(seed)
        rng.shuffle(indices)

    val_size = int(training_num * val_ratio)
    train_size = training_num - val_size

    train_idx = indices[:train_size]
    val_idx = indices[train_size:training_num]
    test_idx = indices[training_num:training_num + test_num]

    if split == "train":
        return train_idx
    if split == "val":
        return val_idx
    if split == "test":
        return test_idx
    if split == "all":
        return indices
    raise ValueError(f"Unknown split: {split}")


def _predict_probabilities(
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.tensor(x[start:end], dtype=torch.float32, device=device)
            yb = model(xb).detach().cpu().numpy()
            outs.append(yb)
    return np.concatenate(outs, axis=0)


def _ensure_ntg_shape(pred: np.ndarray, nt: int, ng: int) -> np.ndarray:
    if pred.ndim == 3:
        if pred.shape[1] == nt and pred.shape[2] == ng:
            return pred
        raise ValueError(f"Prediction shape {pred.shape} is 3D but not (N,{nt},{ng}).")
    if pred.ndim == 2 and pred.shape[1] == nt * ng:
        return pred.reshape(pred.shape[0], nt, ng)
    raise ValueError(
        f"Unsupported prediction shape {pred.shape}; expected (N,T,G) or (N,T*G)."
    )


def _prepare_constraints_input(x_stb: np.ndarray, model_type: str) -> np.ndarray:
    if model_type in ("DNN", "KAN"):
        return x_stb.reshape(x_stb.shape[0], -1)
    return x_stb


def _ensure_constraints_shape(pred: np.ndarray, nl: int) -> np.ndarray:
    if pred.ndim == 2 and pred.shape[1] == nl:
        return pred
    raise ValueError(
        f"Constraints prediction shape {pred.shape} is invalid, expected (N,{nl})."
    )


def _to_g_t_shape(x_2d: np.ndarray, ng: int) -> np.ndarray:
    """
    Convert a 2D matrix to (G, T) according to NG.
    Input can be (T, G) or (G, T).
    """
    if x_2d.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got ndim={x_2d.ndim}.")
    if x_2d.shape[0] == ng:
        return x_2d
    if x_2d.shape[1] == ng:
        return x_2d.T
    raise ValueError(f"Cannot map shape {x_2d.shape} to (G,T) with NG={ng}.")


def _build_dicts_from_u_and_mask(
    u_pred_2d: np.ndarray,
    fix_mask_2d: np.ndarray,
    init_u: np.ndarray,
) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
    """
    Build partial or full fixing dicts.
    - u is fixed where mask is True.
    - y/z at t=0 can be fixed if u(0) is fixed.
    - y/z at t>0 can be fixed only if both u(t-1) and u(t) are fixed.
    """
    ng = len(init_u)
    u_gt = _to_g_t_shape(u_pred_2d, ng=ng).astype(np.int32)
    mask_gt = _to_g_t_shape(fix_mask_2d.astype(np.int32), ng=ng).astype(bool)

    if u_gt.shape != mask_gt.shape:
        raise ValueError(
            f"u/mask shape mismatch: u={u_gt.shape}, mask={mask_gt.shape}."
        )

    _, nt = u_gt.shape
    u_start: Dict[Tuple[int, int], int] = {}
    y_start: Dict[Tuple[int, int], int] = {}
    z_start: Dict[Tuple[int, int], int] = {}

    for g in range(ng):
        for t in range(nt):
            if mask_gt[g, t]:
                u_start[(g, t)] = int(u_gt[g, t])

        if mask_gt[g, 0]:
            delta0 = int(u_gt[g, 0] - init_u[g])
            if delta0 == 1:
                y_start[(g, 0)] = 1
                z_start[(g, 0)] = 0
            elif delta0 == -1:
                y_start[(g, 0)] = 0
                z_start[(g, 0)] = 1
            else:
                y_start[(g, 0)] = 0
                z_start[(g, 0)] = 0

        for t in range(1, nt):
            if mask_gt[g, t] and mask_gt[g, t - 1]:
                delta = int(u_gt[g, t] - u_gt[g, t - 1])
                if delta == 1:
                    y_start[(g, t)] = 1
                    z_start[(g, t)] = 0
                elif delta == -1:
                    y_start[(g, t)] = 0
                    z_start[(g, t)] = 1
                else:
                    y_start[(g, t)] = 0
                    z_start[(g, t)] = 0

    return u_start, y_start, z_start


def _build_full_fix_dicts(
    u_pred_2d: np.ndarray,
    init_u: np.ndarray,
) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
    ng = len(init_u)
    u_gt = _to_g_t_shape(u_pred_2d, ng=ng)
    full_mask = np.ones_like(u_gt, dtype=np.int32)
    return _build_dicts_from_u_and_mask(u_gt, full_mask, init_u)


def _build_high_conf_fix_dicts(
    u_pred_2d: np.ndarray,
    prob_2d: np.ndarray,
    init_u: np.ndarray,
    reliability_low_gate: float,
    reliability_high_gate: float,
) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], int, int]:
    ng = len(init_u)
    u_gt = _to_g_t_shape(u_pred_2d, ng=ng)
    p_gt = _to_g_t_shape(prob_2d, ng=ng)

    high_mask = np.logical_or(p_gt <= reliability_low_gate, p_gt >= reliability_high_gate)
    u_start, y_start, z_start = _build_dicts_from_u_and_mask(u_gt, high_mask, init_u)
    high_count = int(np.sum(high_mask))
    total_count = int(high_mask.size)
    return u_start, y_start, z_start, high_count, total_count


def _is_solver_x_error(error_text: str) -> bool:
    msg = error_text.lower()
    return (
        "unable to retrieve attribute 'x'" in msg
        or "unable to retrieve attribute \"x\"" in msg
    )


def _solve_one_strategy(
    strategy_name: str,
    case_data: Dict[str, Any],
    load_bt: np.ndarray,
    storage_data: Dict[str, np.ndarray],
    initial_constraints: np.ndarray,
    u_start: Dict[Tuple[int, int], int],
    y_start: Dict[Tuple[int, int], int],
    z_start: Dict[Tuple[int, int], int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    try:
        (
            _model, _result, _u, _y, _z, _p, _r, _c, _l,
            _enable_constraints,
            model_time, solve_t, load_data_t, round_num, no_answer
        ) = run_constraint_filter(
            case_data,
            is_warm_start=False,
            use_new_input=True,
            use_storage=args.use_storage,
            use_initial_enable=True,
            add_more_constr=False,
            use_new_min_up_down_time=args.use_new_min_up_down_time,
            use_solar=False,
            u_start=u_start,
            y_start=y_start,
            z_start=z_start,
            new_input=load_bt,
            storage_data=storage_data if args.use_storage else 0,
            line_limit_rate=args.line_limit_rate,
            new_add_num=args.new_add_num,
            initial_constraints=initial_constraints,
            add_constr_gate=args.add_constr_gate,
            fix_int=True,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
        )

        if bool(no_answer):
            status = "infeasible_no_answer"
            feasible = 0
            no_solution = 1
        else:
            status = "feasible"
            feasible = 1
            no_solution = 0

        return {
            "strategy": strategy_name,
            "status": status,
            "feasible": feasible,
            "no_solution": no_solution,
            "no_answer": int(bool(no_answer)),
            "model_time_s": float(model_time),
            "solve_time_s": float(solve_t),
            "load_data_time_s": float(load_data_t),
            "iteration_num": int(round_num),
            "solver_error": "",
        }

    except Exception as exc:  # pylint: disable=broad-except
        err = str(exc)
        if _is_solver_x_error(err):
            status = "infeasible_solver_x"
        else:
            status = "error_other"
        return {
            "strategy": strategy_name,
            "status": status,
            "feasible": 0,
            "no_solution": 1,
            "no_answer": "",
            "model_time_s": "",
            "solve_time_s": "",
            "load_data_time_s": "",
            "iteration_num": "",
            "solver_error": err,
        }


def _summarize_strategy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    feasible_count = sum(int(r["feasible"]) for r in results)
    no_solution_count = total - feasible_count

    infeasible_no_answer = sum(1 for r in results if r["status"] == "infeasible_no_answer")
    infeasible_solver_x = sum(1 for r in results if r["status"] == "infeasible_solver_x")
    error_other = sum(1 for r in results if r["status"] == "error_other")

    iter_vals = [int(r["iteration_num"]) for r in results if r["iteration_num"] != ""]
    avg_iter = (sum(iter_vals) / len(iter_vals)) if iter_vals else float("nan")

    return {
        "total": total,
        "feasible_count": feasible_count,
        "no_solution_count": no_solution_count,
        "feasible_ratio": (feasible_count / total) if total > 0 else float("nan"),
        "no_solution_ratio": (no_solution_count / total) if total > 0 else float("nan"),
        "infeasible_no_answer": infeasible_no_answer,
        "infeasible_solver_x": infeasible_solver_x,
        "error_other": error_other,
        "avg_iteration_num": avg_iter,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare UC feasibility under full-fix and high-confidence-fix binary strategies."
    )
    parser.add_argument("--all-default-cases", action="store_true", help="Run the four default transformer cases used in the paper.")
    parser.add_argument("--case", dest="cases", action="append", default=[], help="Default case label filter; may be repeated.")
    parser.add_argument("--summary-output", default="binary_feasibility_transformer_cases.xlsx", help="Excel summary output for --all-default-cases.")
    parser.add_argument("--summary-csv", default="binary_feasibility_transformer_cases_summary.csv", help="CSV summary output for --all-default-cases.")
    parser.add_argument("--per-case-output-dir", default=".", help="Directory for per-case raw CSV outputs in --all-default-cases.")
    parser.add_argument("--model-name", default="", help="Units model stem without .pth")
    parser.add_argument("--model-dir", default="module_save", help="Directory of saved units models")
    parser.add_argument("--constraints-model-name", default="", help="Constraints model stem without .pth")
    parser.add_argument("--constraints-model-dir", default="module_save", help="Directory of saved constraints models")
    parser.add_argument("--model-data", default="", help="Optional override for model input data path")
    parser.add_argument("--load-data", default="", help="Optional bus-level load data path for UC feasibility check")
    parser.add_argument("--case-json", default="", help="Path to case json.gz")
    parser.add_argument("--storage-data", default="", help="Path to storage .mat/.npz (required if --use-storage)")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold for units model outputs")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random split when data_is_random=True")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to evaluate (0 means all in split)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--line-limit-rate", type=float, default=1.0)
    parser.add_argument("--use-storage", action="store_true", default=False)
    parser.add_argument("--use-new-min-up-down-time", action="store_true", default=True)
    parser.add_argument("--new-add-num", type=int, default=10, help="Lines added per filter iteration")
    parser.add_argument("--add-constr-gate", type=float, default=0.7, help="Optional extra constraint gate")
    parser.add_argument("--time-limit", type=float, default=20.0, help="Per-iteration MILP time limit (seconds)")
    parser.add_argument("--mip-gap", type=float, default=1e-3, help="MIPGap for UC solver")
    parser.add_argument("--reliability-low-gate", type=float, default=0.1, help="Low reliability gate")
    parser.add_argument("--reliability-high-gate", type=float, default=0.9, help="High reliability gate")
    parser.add_argument("--gurobi-output-flag", type=int, default=0)
    parser.add_argument("--gurobi-threads", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output-csv", default="", help="Optional output csv path")
    return parser


def _format_float(value: Any, digits: int = 4) -> str:
    if value == "" or value is None:
        return ""
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(value_f):
        return ""
    return f"{value_f:.{digits}f}"


def _format_pct(value: Any, digits: int = 2) -> str:
    if value == "" or value is None:
        return ""
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(value_f):
        return ""
    return f"{100.0 * value_f:.{digits}f}%"


def _write_summary_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    if not output_path:
        return
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "case",
        "split",
        "sample_count",
        "unit_status_variables_per_sample",
        "reliability_gate",
        "avg_high_conf_ratio",
        "avg_high_conf_u_count",
        "full_fix_feasible_count",
        "full_fix_infeasible_count",
        "full_fix_infeasible_ratio",
        "full_fix_infeasible_no_answer",
        "full_fix_infeasible_solver_x",
        "full_fix_error_other",
        "high_conf_fix_feasible_count",
        "high_conf_fix_infeasible_count",
        "high_conf_fix_infeasible_ratio",
        "high_conf_fix_infeasible_no_answer",
        "high_conf_fix_infeasible_solver_x",
        "high_conf_fix_error_other",
        "both_feasible",
        "only_full_fix_feasible",
        "only_high_conf_fix_feasible",
        "both_infeasible",
        "elapsed_s",
        "raw_csv",
        "units_model",
        "constraints_model",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_summary_xlsx(rows: List[Dict[str, Any]], output_path: str) -> None:
    if not output_path:
        return
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Binary feasibility"

    headers = [
        ("Case", "case"),
        ("Samples", "sample_count"),
        ("Gate", "reliability_gate"),
        ("# status vars/sample", "unit_status_variables_per_sample"),
        ("High-conf. proportion", "avg_high_conf_ratio_pct"),
        ("Full-fix feasible", "full_fix_feasible"),
        ("Full-fix infeasible", "full_fix_infeasible"),
        ("Full-fix infeasible ratio", "full_fix_infeasible_ratio_pct"),
        ("High-conf. fix feasible", "high_conf_fix_feasible"),
        ("High-conf. fix infeasible", "high_conf_fix_infeasible"),
        ("High-conf. fix infeasible ratio", "high_conf_fix_infeasible_ratio_pct"),
    ]

    header_fill = PatternFill("solid", fgColor="D9EAF7")
    thin = Side(style="thin", color="B7B7B7")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    ws.append([h for h, _ in headers])
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border

    for row in rows:
        sample_count = int(row.get("sample_count", 0))
        full_feasible = int(row.get("full_fix_feasible_count", 0))
        full_infeasible = int(row.get("full_fix_infeasible_count", 0))
        high_feasible = int(row.get("high_conf_fix_feasible_count", 0))
        high_infeasible = int(row.get("high_conf_fix_infeasible_count", 0))
        display_row = {
            "case": row.get("case", ""),
            "sample_count": sample_count,
            "reliability_gate": row.get("reliability_gate", ""),
            "unit_status_variables_per_sample": row.get("unit_status_variables_per_sample", ""),
            "avg_high_conf_ratio_pct": _format_pct(row.get("avg_high_conf_ratio")),
            "full_fix_feasible": f"{full_feasible}/{sample_count}" if sample_count else "",
            "full_fix_infeasible": f"{full_infeasible}/{sample_count}" if sample_count else "",
            "full_fix_infeasible_ratio_pct": _format_pct(row.get("full_fix_infeasible_ratio")),
            "high_conf_fix_feasible": f"{high_feasible}/{sample_count}" if sample_count else "",
            "high_conf_fix_infeasible": f"{high_infeasible}/{sample_count}" if sample_count else "",
            "high_conf_fix_infeasible_ratio_pct": _format_pct(row.get("high_conf_fix_infeasible_ratio")),
        }
        ws.append([display_row[key] for _, key in headers])

    for row_cells in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row_cells:
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border

    widths = {
        1: 18,
        2: 10,
        3: 24,
        4: 18,
        5: 18,
        6: 16,
        7: 17,
        8: 20,
        9: 20,
        10: 22,
        11: 26,
    }
    for col_idx, width in widths.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    ws.freeze_panes = "A2"

    detail = wb.create_sheet("Details")
    detail_headers = [
        "case",
        "units_model",
        "constraints_model",
        "split",
        "sample_count",
        "reliability_gate",
        "avg_high_conf_ratio",
        "avg_high_conf_u_count",
        "full_fix_infeasible_no_answer",
        "full_fix_infeasible_solver_x",
        "full_fix_error_other",
        "high_conf_fix_infeasible_no_answer",
        "high_conf_fix_infeasible_solver_x",
        "high_conf_fix_error_other",
        "both_feasible",
        "only_full_fix_feasible",
        "only_high_conf_fix_feasible",
        "both_infeasible",
        "elapsed_s",
        "raw_csv",
    ]
    detail.append(detail_headers)
    for cell in detail[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border
    for row in rows:
        detail.append([row.get(name, "") for name in detail_headers])
    for row_cells in detail.iter_rows(min_row=2, max_row=detail.max_row):
        for cell in row_cells:
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border
    for col_idx, header in enumerate(detail_headers, start=1):
        detail.column_dimensions[get_column_letter(col_idx)].width = min(max(len(header) + 2, 14), 36)

    wb.save(output_path)


def _config_from_default_case(spec: DefaultCaseSpec, args: argparse.Namespace) -> Dict[str, Any]:
    low_gate, high_gate = DEFAULT_CASE_RELIABILITY_GATES.get(
        spec.case_label,
        DEFAULT_RELIABILITY_GATES,
    )
    output_dir = args.per_case_output_dir or "."
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"binary_feasibility_compare_{spec.case_label}_{args.split}.csv")

    return {
        "case_label": spec.case_label,
        "model_name": spec.units_model_name,
        "model_dir": args.model_dir,
        "constraints_model_name": spec.constraints_model_name,
        "constraints_model_dir": args.constraints_model_dir,
        "model_data": args.model_data,
        "load_data": args.load_data,
        "case_json": spec.case_json,
        "storage_data": spec.storage_data or "",
        "split": args.split,
        "threshold": args.threshold,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "line_limit_rate": spec.line_limit_rate,
        "use_storage": spec.use_storage,
        "use_new_min_up_down_time": args.use_new_min_up_down_time,
        "new_add_num": args.new_add_num,
        "add_constr_gate": args.add_constr_gate,
        "time_limit": args.time_limit,
        "mip_gap": args.mip_gap,
        "reliability_low_gate": low_gate,
        "reliability_high_gate": high_gate,
        "gurobi_output_flag": args.gurobi_output_flag,
        "gurobi_threads": args.gurobi_threads,
        "torch_threads": args.torch_threads,
        "output_csv": output_csv,
    }


def run_default_cases(args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected = set(args.cases or [])
    unknown = selected - {spec.case_label for spec in DEFAULT_CASE_SPECS}
    if unknown:
        raise ValueError(f"Unknown default case label(s): {sorted(unknown)}")

    specs = [spec for spec in DEFAULT_CASE_SPECS if not selected or spec.case_label in selected]
    summaries: List[Dict[str, Any]] = []
    for spec in specs:
        print(f"\n\n################ {spec.case_label} ################")
        summaries.append(run_with_config(_config_from_default_case(spec, args)))

    _write_summary_csv(summaries, args.summary_csv)
    _write_summary_xlsx(summaries, args.summary_output)
    print(f"\nsummary_csv_saved: {args.summary_csv}")
    print(f"summary_xlsx_saved: {args.summary_output}")
    return summaries


def run_with_config(config: Dict[str, Any]):
    """
    Function-style entrypoint for PyCharm.
    """
    args = argparse.Namespace(**config)
    args.case_label = getattr(args, "case_label", "")
    args.model_name = getattr(args, "model_name", "")
    args.model_dir = getattr(args, "model_dir", MODEL_DIR)
    args.constraints_model_name = getattr(args, "constraints_model_name", "")
    args.constraints_model_dir = getattr(args, "constraints_model_dir", MODEL_DIR)
    args.new_add_num = int(getattr(args, "new_add_num", 10))
    args.add_constr_gate = float(getattr(args, "add_constr_gate", 0.7))
    args.time_limit = float(getattr(args, "time_limit", 20.0))
    args.mip_gap = float(getattr(args, "mip_gap", 1e-3))
    args.reliability_low_gate = float(getattr(args, "reliability_low_gate", 0.1))
    args.reliability_high_gate = float(getattr(args, "reliability_high_gate", 0.9))
    args.gurobi_output_flag = int(getattr(args, "gurobi_output_flag", 0))
    args.gurobi_threads = int(getattr(args, "gurobi_threads", 1))
    args.torch_threads = int(getattr(args, "torch_threads", 1))

    if not args.model_name:
        raise ValueError("model_name is empty.")
    if not getattr(args, "case_json", ""):
        raise ValueError("case_json is empty.")

    if args.use_storage and not args.storage_data:
        raise ValueError("--use-storage is set, but --storage-data is empty.")

    if not (0.0 <= args.reliability_low_gate <= 1.0 and 0.0 <= args.reliability_high_gate <= 1.0):
        raise ValueError("reliability gates must be in [0, 1].")
    if args.reliability_low_gate >= args.reliability_high_gate:
        raise ValueError("reliability_low_gate must be smaller than reliability_high_gate.")

    os.environ["UC_GUROBI_OUTPUTFLAG"] = str(args.gurobi_output_flag)
    if args.gurobi_threads > 0:
        os.environ["UC_GUROBI_THREADS"] = str(args.gurobi_threads)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # 1) Load units model metadata.
    (
        model,
        model_type,
        training_num,
        test_num,
        data_is_random,
        cutted_by_time,
        data_name,
        data_type,
    ) = ml_load_model(args.model_name, args.model_dir)

    if data_type != "units":
        raise ValueError(f"Model data_type is '{data_type}', expected 'units'.")
    if model_type not in ("transformer", "BiLSTM", "LSTM_new"):
        raise ValueError(
            f"This script supports sequential units models (transformer/BiLSTM/LSTM_new), got '{model_type}'."
        )
    if cutted_by_time:
        raise ValueError("cutted_by_time=True is not supported in this script.")
    if not args.constraints_model_name:
        raise ValueError("constraints_model_name is empty.")

    # Read val_ratio from metadata if available.
    meta_json = os.path.join(args.model_dir, args.model_name + "_metadata.json")
    val_ratio = 0.1
    if os.path.exists(meta_json):
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
            val_ratio = float(meta.get("val_ratio", 0.1))

    model_data_path = args.model_data if args.model_data else os.path.join("training_data", data_name)

    # 2) Load input arrays.
    d_model = _load_array_dict(model_data_path)
    x_model = _to_stb_input(d_model)  # (S,T,F_model)

    if args.load_data:
        d_load = _load_array_dict(args.load_data)
        x_load = _to_stb_input(d_load)  # (S,T,B)
        avg_load = _read_average_load(d_load)
        load_data_path = args.load_data
    else:
        x_load = x_model
        avg_load = _read_average_load(d_model)
        load_data_path = model_data_path

    # 3) Load case and validate dimensions.
    with gzip.open(args.case_json, "rt", encoding="utf-8") as f:
        case_data = json.load(f)

    nt, ng, nl, init_u = _get_case_info(case_data)
    nb = len(case_data["Buses"])

    if x_model.shape[1] != nt:
        raise ValueError(f"Model input T={x_model.shape[1]} but case NT={nt}.")
    if x_load.shape[1] != nt:
        raise ValueError(f"Load input T={x_load.shape[1]} but case NT={nt}.")
    if x_load.shape[2] != nb:
        raise ValueError(
            f"Load feature dim is {x_load.shape[2]}, but case bus number is {nb}. "
            f"If model input is aggregated, provide bus-level --load-data."
        )

    # 4) Build split indices.
    idx = _build_split_indices(
        total_samples=x_model.shape[0],
        training_num=training_num,
        test_num=test_num,
        val_ratio=val_ratio,
        data_is_random=bool(data_is_random),
        seed=args.seed,
        split=args.split,
    )
    if args.max_samples and args.max_samples > 0:
        idx = idx[:args.max_samples]
    if not idx:
        raise ValueError("Selected split has zero samples.")

    x_model_split = x_model[idx]
    x_load_split = x_load[idx]

    # 5) Predict unit commitment probabilities.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    probs = _predict_probabilities(model, x_model_split, device, args.batch_size)
    probs_ntg = _ensure_ntg_shape(probs, nt=nt, ng=ng)
    u_bin = (probs_ntg >= args.threshold).astype(np.int32)  # (N,T,G)

    # 6) Predict initial active constraints.
    (
        constraints_model,
        constraints_model_type,
        _c_training_num,
        _c_test_num,
        _c_data_is_random,
        _c_cutted_by_time,
        _c_data_name,
        constraints_data_type,
    ) = ml_load_model(args.constraints_model_name, args.constraints_model_dir)

    if constraints_data_type != "constraints":
        raise ValueError(
            f"constraints model data_type is '{constraints_data_type}', expected 'constraints'."
        )

    constraints_model = constraints_model.to(device)
    constraints_input = _prepare_constraints_input(x_load_split, constraints_model_type)
    constraints_probs = _predict_probabilities(
        constraints_model, constraints_input, device, args.batch_size
    )
    constraints_init = _ensure_constraints_shape(constraints_probs, nl=nl)

    # 7) Feasibility checks with run_constraint_filter for two strategies.
    storage_data = _load_storage(args.storage_data) if args.use_storage else {}

    print("\n================ Run Settings ================")
    print(f"units_model_name: {args.model_name}")
    if args.case_label:
        print(f"case_label: {args.case_label}")
    print(f"constraints_model_name: {args.constraints_model_name}")
    print(f"units_model_data_path: {model_data_path}")
    print(f"load_data_path: {load_data_path}")
    print(f"split: {args.split}")
    print(f"selected_samples: {len(idx)}")
    print(f"threshold: {args.threshold}")
    print(f"reliability_low_gate: {args.reliability_low_gate}")
    print(f"reliability_high_gate: {args.reliability_high_gate}")
    print("==============================================\n")

    rows: List[Dict[str, Any]] = []
    full_results: List[Dict[str, Any]] = []
    high_results: List[Dict[str, Any]] = []

    t0 = time.time()
    total = len(idx)

    for k in range(total):
        sample_id = idx[k]
        load_bt = x_load_split[k].T * avg_load  # (B,T)
        initial_constraints = constraints_init[k]

        # Strategy A: fix all predicted binary vars.
        u_full, y_full, z_full = _build_full_fix_dicts(u_bin[k], init_u=init_u)
        full_res = _solve_one_strategy(
            strategy_name="full_fix",
            case_data=case_data,
            load_bt=load_bt,
            storage_data=storage_data,
            initial_constraints=initial_constraints,
            u_start=u_full,
            y_start=y_full,
            z_start=z_full,
            args=args,
        )

        # Strategy B: fix high-confidence subset only.
        u_high, y_high, z_high, high_count, total_count = _build_high_conf_fix_dicts(
            u_pred_2d=u_bin[k],
            prob_2d=probs_ntg[k],
            init_u=init_u,
            reliability_low_gate=args.reliability_low_gate,
            reliability_high_gate=args.reliability_high_gate,
        )
        high_res = _solve_one_strategy(
            strategy_name="high_conf_fix",
            case_data=case_data,
            load_bt=load_bt,
            storage_data=storage_data,
            initial_constraints=initial_constraints,
            u_start=u_high,
            y_start=y_high,
            z_start=z_high,
            args=args,
        )

        full_results.append(full_res)
        high_results.append(high_res)

        high_ratio = (high_count / total_count) if total_count > 0 else float("nan")
        row = {
            "sample_id": sample_id,
            "all_u_count": total_count,
            "high_conf_u_count": high_count,
            "high_conf_u_ratio": high_ratio,
            "full_fixed_u": len(u_full),
            "full_fixed_y": len(y_full),
            "full_fixed_z": len(z_full),
            "full_status": full_res["status"],
            "full_feasible": full_res["feasible"],
            "full_no_solution": full_res["no_solution"],
            "full_no_answer": full_res["no_answer"],
            "full_iteration_num": full_res["iteration_num"],
            "full_model_time_s": full_res["model_time_s"],
            "full_solve_time_s": full_res["solve_time_s"],
            "full_load_data_time_s": full_res["load_data_time_s"],
            "full_solver_error": full_res["solver_error"],
            "high_fixed_u": len(u_high),
            "high_fixed_y": len(y_high),
            "high_fixed_z": len(z_high),
            "high_status": high_res["status"],
            "high_feasible": high_res["feasible"],
            "high_no_solution": high_res["no_solution"],
            "high_no_answer": high_res["no_answer"],
            "high_iteration_num": high_res["iteration_num"],
            "high_model_time_s": high_res["model_time_s"],
            "high_solve_time_s": high_res["solve_time_s"],
            "high_load_data_time_s": high_res["load_data_time_s"],
            "high_solver_error": high_res["solver_error"],
        }
        rows.append(row)

        print(
            f"[{k+1}/{total}] sample={sample_id}, "
            f"full={full_res['status']}, "
            f"high={high_res['status']}, "
            f"high_conf_u={high_count}/{total_count} ({high_ratio:.3f})"
        )

    elapsed = time.time() - t0

    # 8) Summaries.
    full_summary = _summarize_strategy(full_results)
    high_summary = _summarize_strategy(high_results)

    both_feasible = sum(
        1 for fr, hr in zip(full_results, high_results) if fr["feasible"] == 1 and hr["feasible"] == 1
    )
    full_only_feasible = sum(
        1 for fr, hr in zip(full_results, high_results) if fr["feasible"] == 1 and hr["feasible"] == 0
    )
    high_only_feasible = sum(
        1 for fr, hr in zip(full_results, high_results) if fr["feasible"] == 0 and hr["feasible"] == 1
    )
    both_no_solution = sum(
        1 for fr, hr in zip(full_results, high_results) if fr["feasible"] == 0 and hr["feasible"] == 0
    )

    # 9) Save csv.
    model_tag = args.model_name.replace("\\", "_").replace("/", "_")
    default_csv = f"binary_feasibility_compare_{model_tag}_{args.split}.csv"
    out_csv = args.output_csv if args.output_csv else default_csv

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id",
            "all_u_count",
            "high_conf_u_count",
            "high_conf_u_ratio",
            "full_fixed_u",
            "full_fixed_y",
            "full_fixed_z",
            "full_status",
            "full_feasible",
            "full_no_solution",
            "full_no_answer",
            "full_iteration_num",
            "full_model_time_s",
            "full_solve_time_s",
            "full_load_data_time_s",
            "full_solver_error",
            "high_fixed_u",
            "high_fixed_y",
            "high_fixed_z",
            "high_status",
            "high_feasible",
            "high_no_solution",
            "high_no_answer",
            "high_iteration_num",
            "high_model_time_s",
            "high_solve_time_s",
            "high_load_data_time_s",
            "high_solver_error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n================ Summary (Full Fix) ================")
    print(f"total_cases: {full_summary['total']}")
    print(f"solvable_count: {full_summary['feasible_count']}")
    print(f"unsolvable_count: {full_summary['no_solution_count']}")
    print(f"solvable_ratio: {full_summary['feasible_ratio']:.6f}")
    print(f"unsolvable_ratio: {full_summary['no_solution_ratio']:.6f}")
    print(f"infeasible_no_answer_count: {full_summary['infeasible_no_answer']}")
    print(f"infeasible_solver_x_count: {full_summary['infeasible_solver_x']}")
    print(f"other_error_count: {full_summary['error_other']}")
    if np.isnan(full_summary["avg_iteration_num"]):
        print("avg_iteration_num: NaN")
    else:
        print(f"avg_iteration_num: {full_summary['avg_iteration_num']:.3f}")

    print("\n========== Summary (High-Confidence Fix) ==========")
    print(f"total_cases: {high_summary['total']}")
    print(f"solvable_count: {high_summary['feasible_count']}")
    print(f"unsolvable_count: {high_summary['no_solution_count']}")
    print(f"solvable_ratio: {high_summary['feasible_ratio']:.6f}")
    print(f"unsolvable_ratio: {high_summary['no_solution_ratio']:.6f}")
    print(f"infeasible_no_answer_count: {high_summary['infeasible_no_answer']}")
    print(f"infeasible_solver_x_count: {high_summary['infeasible_solver_x']}")
    print(f"other_error_count: {high_summary['error_other']}")
    if np.isnan(high_summary["avg_iteration_num"]):
        print("avg_iteration_num: NaN")
    else:
        print(f"avg_iteration_num: {high_summary['avg_iteration_num']:.3f}")

    print("\n============== Cross Comparison ==============")
    print(f"both_solvable: {both_feasible}")
    print(f"only_full_fix_solvable: {full_only_feasible}")
    print(f"only_high_conf_fix_solvable: {high_only_feasible}")
    print(f"both_unsolvable: {both_no_solution}")
    print(f"elapsed_s: {elapsed:.2f}")
    print(f"csv_saved: {out_csv}")
    print("=============================================")

    summary = {
        "case": args.case_label if args.case_label else args.model_name,
        "units_model": args.model_name,
        "constraints_model": args.constraints_model_name,
        "split": args.split,
        "sample_count": full_summary["total"],
        "unit_status_variables_per_sample": rows[0]["all_u_count"] if rows else "",
        "reliability_gate": f"[0,{args.reliability_low_gate:g}] U [{args.reliability_high_gate:g},1]",
        "avg_high_conf_ratio": float(np.mean([float(r["high_conf_u_ratio"]) for r in rows])) if rows else float("nan"),
        "avg_high_conf_u_count": float(np.mean([float(r["high_conf_u_count"]) for r in rows])) if rows else float("nan"),
        "full_fix_feasible_count": full_summary["feasible_count"],
        "full_fix_infeasible_count": full_summary["no_solution_count"],
        "full_fix_infeasible_ratio": full_summary["no_solution_ratio"],
        "full_fix_infeasible_no_answer": full_summary["infeasible_no_answer"],
        "full_fix_infeasible_solver_x": full_summary["infeasible_solver_x"],
        "full_fix_error_other": full_summary["error_other"],
        "high_conf_fix_feasible_count": high_summary["feasible_count"],
        "high_conf_fix_infeasible_count": high_summary["no_solution_count"],
        "high_conf_fix_infeasible_ratio": high_summary["no_solution_ratio"],
        "high_conf_fix_infeasible_no_answer": high_summary["infeasible_no_answer"],
        "high_conf_fix_infeasible_solver_x": high_summary["infeasible_solver_x"],
        "high_conf_fix_error_other": high_summary["error_other"],
        "both_feasible": both_feasible,
        "only_full_fix_feasible": full_only_feasible,
        "only_high_conf_fix_feasible": high_only_feasible,
        "both_infeasible": both_no_solution,
        "raw_csv": out_csv,
        "elapsed_s": elapsed,
    }
    return summary


def get_default_config() -> Dict[str, Any]:
    """
    PyCharm-friendly config function. Edit this function directly.
    """
    return {
        "case_label": "case1888rte",
        "model_name": "units_transformer_all_training_data_case1888rte_pro_random_1000_50000_128_0.0002",
        "model_dir": "module_save",
        "constraints_model_name": "constraints_DNN_all_training_data_case1888rte_pro_random_1000_50000_[256, 256, 256]_0.0002",
        "constraints_model_dir": "module_save",
        "model_data": "",  # empty -> use data_name from units model metadata
        "load_data": "",   # set bus-level load data if model input is aggregated
        "case_json": "case_data/matpower/case1888rte_pro/2017-01-01.json.gz",
        "storage_data": "training_data/storage_data_case1888rte",
        "split": "test",   # train / val / test / all
        "threshold": 0.5,
        "seed": 42,
        "max_samples": 100,  # 0 means all samples in split
        "batch_size": 32,
        "line_limit_rate": 1.0,
        "use_storage": True,
        "use_new_min_up_down_time": True,
        "new_add_num": 10,
        "add_constr_gate": 0.7,
        "time_limit": 200.0,
        "mip_gap": 1e-3,
        "reliability_low_gate": 0.05,
        "reliability_high_gate": 0.95,
        "gurobi_output_flag": 0,
        "gurobi_threads": 1,
        "torch_threads": 1,
        "output_csv": "",  # empty -> auto name
    }


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.all_default_cases:
        run_default_cases(args)
        return

    if not args.model_name:
        parser.error("--model-name is required unless --all-default-cases is used.")
    if not args.case_json:
        parser.error("--case-json is required unless --all-default-cases is used.")
    if not args.constraints_model_name:
        parser.error("--constraints-model-name is required unless --all-default-cases is used.")
    run_with_config(vars(args))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        run_with_config(get_default_config())
