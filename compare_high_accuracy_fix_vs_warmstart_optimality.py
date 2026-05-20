"""
compare_high_accuracy_fix_vs_warmstart_optimality.py

Compare optimality for three strategies on the same UC samples:
1) baseline: no warm-start, no fixed NN binaries
2) high_conf_warmstart: use high-confidence NN binaries only as warm-start
3) high_conf_fix: fix high-confidence NN binaries

Acceleration:
- constraints NN predicts initial active constraints
- run_constraint_filter iteratively adds violated constraints

Default mip_gap for this comparison: 1e-5
"""

import argparse
import csv
import gzip
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ml_basic_functions import ml_load_model
from uc_gurobi_constraint_filter import run_constraint_filter


def _replace_ext(path: str, new_ext: str) -> str:
    base, _ = os.path.splitext(path)
    return base + new_ext


def _load_array_dict(data_path: str) -> Dict[str, np.ndarray]:
    npz_path = _replace_ext(data_path, ".npz")
    if os.path.exists(npz_path):
        return dict(np.load(npz_path, allow_pickle=True))
    if os.path.exists(data_path):
        import scipy.io
        return scipy.io.loadmat(data_path)
    raise FileNotFoundError(f"Neither {npz_path} nor {data_path} exists.")


def _to_stb_input(d: Dict[str, np.ndarray]) -> np.ndarray:
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
    usable = training_num + test_num
    if total_samples < usable:
        raise ValueError(f"Dataset has {total_samples} samples, but training_num+test_num={usable}.")
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


def _predict_probabilities(model: torch.nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.tensor(x[start:end], dtype=torch.float32, device=device)
            outs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def _ensure_ntg_shape(pred: np.ndarray, nt: int, ng: int) -> np.ndarray:
    if pred.ndim == 3:
        if pred.shape[1] == nt and pred.shape[2] == ng:
            return pred
        raise ValueError(f"Prediction shape {pred.shape} is not (N,{nt},{ng}).")
    if pred.ndim == 2 and pred.shape[1] == nt * ng:
        return pred.reshape(pred.shape[0], nt, ng)
    raise ValueError(f"Unsupported prediction shape {pred.shape}.")


def _prepare_constraints_input(x_stb: np.ndarray, model_type: str) -> np.ndarray:
    if model_type in ("DNN", "KAN"):
        return x_stb.reshape(x_stb.shape[0], -1)
    return x_stb


def _ensure_constraints_shape(pred: np.ndarray, nl: int) -> np.ndarray:
    if pred.ndim == 2 and pred.shape[1] == nl:
        return pred
    raise ValueError(f"Constraints prediction shape {pred.shape} is invalid, expected (N,{nl}).")


def _to_g_t_shape(x_2d: np.ndarray, ng: int) -> np.ndarray:
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
    ng = len(init_u)
    u_gt = _to_g_t_shape(u_pred_2d, ng=ng).astype(np.int32)
    mask_gt = _to_g_t_shape(fix_mask_2d.astype(np.int32), ng=ng).astype(bool)
    if u_gt.shape != mask_gt.shape:
        raise ValueError(f"u/mask shape mismatch: u={u_gt.shape}, mask={mask_gt.shape}.")

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

def _build_high_conf_dicts(
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


def _safe_gap_pct(ref_obj: float, other_obj: float) -> float:
    denom = abs(ref_obj)
    if denom < 1e-12:
        return float("nan")
    return 100.0 * (other_obj - ref_obj) / denom


def _degradation_pct(ref_obj: float, other_obj: float) -> float:
    raw = _safe_gap_pct(ref_obj, other_obj)
    if np.isnan(raw):
        return float("nan")
    return max(raw, 0.0)


def _solve_strategy(
    strategy_name: str,
    case_data: Dict[str, Any],
    load_bt: np.ndarray,
    storage_data: Dict[str, np.ndarray],
    initial_constraints: np.ndarray,
    is_warm_start: bool,
    fix_int: bool,
    u_start: Dict[Tuple[int, int], int],
    y_start: Dict[Tuple[int, int], int],
    z_start: Dict[Tuple[int, int], int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    try:
        (
            _model, result, _u, _y, _z, _p, _r, _c, _l,
            _enable_constraints,
            model_time, solve_t, load_data_t, round_num, no_answer
        ) = run_constraint_filter(
            case_data,
            is_warm_start=is_warm_start,
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
            fix_int=fix_int,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
        )

        if bool(no_answer):
            status = "infeasible_no_answer"
            feasible = 0
            no_solution = 1
            objective = ""
        else:
            status = "feasible"
            feasible = 1
            no_solution = 0
            objective = float(result)

        return {
            "strategy": strategy_name,
            "status": status,
            "feasible": feasible,
            "no_solution": no_solution,
            "objective": objective,
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
            "objective": "",
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
    return {
        "total": total,
        "feasible_count": feasible_count,
        "no_solution_count": no_solution_count,
        "infeasible_no_answer": infeasible_no_answer,
        "infeasible_solver_x": infeasible_solver_x,
        "error_other": error_other,
        "avg_iteration_num": (sum(iter_vals) / len(iter_vals)) if iter_vals else float("nan"),
    }


def _summarize_gap(gaps: List[float]) -> Dict[str, Any]:
    vals = [float(x) for x in gaps if not np.isnan(x)]
    if not vals:
        return {"count": 0, "avg": float("nan"), "max": float("nan")}
    return {"count": len(vals), "avg": float(np.mean(vals)), "max": float(np.max(vals))}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare optimality: high-confidence fix vs high-confidence warm-start")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-dir", default="module_save")
    parser.add_argument("--constraints-model-name", required=True)
    parser.add_argument("--constraints-model-dir", default="module_save")
    parser.add_argument("--model-data", default="")
    parser.add_argument("--load-data", default="")
    parser.add_argument("--case-json", required=True)
    parser.add_argument("--storage-data", default="")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--reliability-low-gate", type=float, default=0.1)
    parser.add_argument("--reliability-high-gate", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--line-limit-rate", type=float, default=1.0)
    parser.add_argument("--use-storage", action="store_true", default=False)
    parser.add_argument("--use-new-min-up-down-time", action="store_true", default=True)
    parser.add_argument("--new-add-num", type=int, default=10)
    parser.add_argument("--add-constr-gate", type=float, default=0.7)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--mip-gap", type=float, default=1e-5)
    parser.add_argument("--output-csv", default="")
    return parser

def run_with_config(config: Dict[str, Any]):
    args = argparse.Namespace(**config)
    args.constraints_model_name = getattr(args, "constraints_model_name", "")
    args.constraints_model_dir = getattr(args, "constraints_model_dir", "module_save")
    args.new_add_num = int(getattr(args, "new_add_num", 10))
    args.add_constr_gate = float(getattr(args, "add_constr_gate", 0.7))
    args.time_limit = float(getattr(args, "time_limit", 120.0))
    args.mip_gap = float(getattr(args, "mip_gap", 1e-5))
    args.reliability_low_gate = float(getattr(args, "reliability_low_gate", 0.1))
    args.reliability_high_gate = float(getattr(args, "reliability_high_gate", 0.9))

    if args.use_storage and not args.storage_data:
        raise ValueError("--use-storage is set, but --storage-data is empty.")
    if args.reliability_low_gate >= args.reliability_high_gate:
        raise ValueError("reliability_low_gate must be smaller than reliability_high_gate.")

    model, model_type, training_num, test_num, data_is_random, cutted_by_time, data_name, data_type = \
        ml_load_model(args.model_name, args.model_dir)
    if data_type != "units":
        raise ValueError(f"Units model data_type is '{data_type}', expected 'units'.")
    if model_type not in ("transformer", "BiLSTM", "LSTM_new"):
        raise ValueError(f"Unsupported units model_type: {model_type}")
    if cutted_by_time:
        raise ValueError("cutted_by_time=True is not supported.")

    meta_json = os.path.join(args.model_dir, args.model_name + "_metadata.json")
    val_ratio = 0.1
    if os.path.exists(meta_json):
        with open(meta_json, "r", encoding="utf-8") as f:
            val_ratio = float(json.load(f).get("val_ratio", 0.1))

    model_data_path = args.model_data if args.model_data else os.path.join("training_data", data_name)
    d_model = _load_array_dict(model_data_path)
    x_model = _to_stb_input(d_model)

    if args.load_data:
        d_load = _load_array_dict(args.load_data)
        x_load = _to_stb_input(d_load)
        avg_load = _read_average_load(d_load)
        load_data_path = args.load_data
    else:
        x_load = x_model
        avg_load = _read_average_load(d_model)
        load_data_path = model_data_path

    with gzip.open(args.case_json, "rt", encoding="utf-8") as f:
        case_data = json.load(f)

    nt, ng, nl, init_u = _get_case_info(case_data)
    nb = len(case_data["Buses"])
    if x_model.shape[1] != nt:
        raise ValueError(f"Model input T={x_model.shape[1]} but case NT={nt}.")
    if x_load.shape[1] != nt:
        raise ValueError(f"Load input T={x_load.shape[1]} but case NT={nt}.")
    if x_load.shape[2] != nb:
        raise ValueError(f"Load feature dim is {x_load.shape[2]}, but case bus number is {nb}.")

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    probs = _predict_probabilities(model, x_model_split, device, args.batch_size)
    probs_ntg = _ensure_ntg_shape(probs, nt=nt, ng=ng)
    u_bin = (probs_ntg >= args.threshold).astype(np.int32)

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
        raise ValueError(f"constraints model data_type is '{constraints_data_type}', expected 'constraints'.")

    constraints_model = constraints_model.to(device)
    constraints_input = _prepare_constraints_input(x_load_split, constraints_model_type)
    constraints_probs = _predict_probabilities(constraints_model, constraints_input, device, args.batch_size)
    constraints_init = _ensure_constraints_shape(constraints_probs, nl=nl)

    storage_data = _load_storage(args.storage_data) if args.use_storage else {}

    print("\n================ Run Settings ================")
    print(f"units_model_name: {args.model_name}")
    print(f"constraints_model_name: {args.constraints_model_name}")
    print(f"units_model_data_path: {model_data_path}")
    print(f"load_data_path: {load_data_path}")
    print(f"split: {args.split}")
    print(f"selected_samples: {len(idx)}")
    print(f"threshold: {args.threshold}")
    print(f"reliability_low_gate: {args.reliability_low_gate}")
    print(f"reliability_high_gate: {args.reliability_high_gate}")
    print(f"mip_gap: {args.mip_gap}")
    print("==============================================\n")

    rows: List[Dict[str, Any]] = []
    baseline_results: List[Dict[str, Any]] = []
    warm_results: List[Dict[str, Any]] = []
    fix_results: List[Dict[str, Any]] = []

    gap_warm_vs_base_raw: List[float] = []
    gap_warm_vs_base_deg: List[float] = []
    gap_fix_vs_base_raw: List[float] = []
    gap_fix_vs_base_deg: List[float] = []
    gap_fix_vs_warm_raw: List[float] = []
    gap_fix_vs_warm_deg: List[float] = []

    t0 = time.time()
    total = len(idx)

    for k in range(total):
        sample_id = idx[k]
        load_bt = x_load_split[k].T * avg_load
        initial_constraints = constraints_init[k]

        u_high, y_high, z_high, high_count, total_count = _build_high_conf_dicts(
            u_pred_2d=u_bin[k],
            prob_2d=probs_ntg[k],
            init_u=init_u,
            reliability_low_gate=args.reliability_low_gate,
            reliability_high_gate=args.reliability_high_gate,
        )

        baseline = _solve_strategy("baseline", case_data, load_bt, storage_data, initial_constraints,
                                   is_warm_start=False, fix_int=False,
                                   u_start={}, y_start={}, z_start={}, args=args)
        warm = _solve_strategy("high_conf_warmstart", case_data, load_bt, storage_data, initial_constraints,
                               is_warm_start=True, fix_int=False,
                               u_start=u_high, y_start=y_high, z_start=z_high, args=args)
        fixed = _solve_strategy("high_conf_fix", case_data, load_bt, storage_data, initial_constraints,
                                is_warm_start=False, fix_int=True,
                                u_start=u_high, y_start=y_high, z_start=z_high, args=args)

        baseline_results.append(baseline)
        warm_results.append(warm)
        fix_results.append(fixed)

        warm_gap_raw = float("nan")
        warm_gap_deg = float("nan")
        fix_gap_raw = float("nan")
        fix_gap_deg = float("nan")
        fix_warm_gap_raw = float("nan")
        fix_warm_gap_deg = float("nan")

        if baseline["feasible"] == 1 and warm["feasible"] == 1:
            warm_gap_raw = _safe_gap_pct(float(baseline["objective"]), float(warm["objective"]))
            warm_gap_deg = _degradation_pct(float(baseline["objective"]), float(warm["objective"]))
        if baseline["feasible"] == 1 and fixed["feasible"] == 1:
            fix_gap_raw = _safe_gap_pct(float(baseline["objective"]), float(fixed["objective"]))
            fix_gap_deg = _degradation_pct(float(baseline["objective"]), float(fixed["objective"]))
        if warm["feasible"] == 1 and fixed["feasible"] == 1:
            fix_warm_gap_raw = _safe_gap_pct(float(warm["objective"]), float(fixed["objective"]))
            fix_warm_gap_deg = _degradation_pct(float(warm["objective"]), float(fixed["objective"]))

        gap_warm_vs_base_raw.append(warm_gap_raw)
        gap_warm_vs_base_deg.append(warm_gap_deg)
        gap_fix_vs_base_raw.append(fix_gap_raw)
        gap_fix_vs_base_deg.append(fix_gap_deg)
        gap_fix_vs_warm_raw.append(fix_warm_gap_raw)
        gap_fix_vs_warm_deg.append(fix_warm_gap_deg)

        high_ratio = (high_count / total_count) if total_count > 0 else float("nan")
        rows.append({
            "sample_id": sample_id,
            "all_u_count": total_count,
            "high_conf_u_count": high_count,
            "high_conf_u_ratio": high_ratio,
            "high_fixed_u": len(u_high),
            "high_fixed_y": len(y_high),
            "high_fixed_z": len(z_high),
            "baseline_status": baseline["status"],
            "baseline_feasible": baseline["feasible"],
            "baseline_objective": baseline["objective"],
            "warm_status": warm["status"],
            "warm_feasible": warm["feasible"],
            "warm_objective": warm["objective"],
            "fix_status": fixed["status"],
            "fix_feasible": fixed["feasible"],
            "fix_objective": fixed["objective"],
            "gap_warm_vs_baseline_raw_pct": warm_gap_raw,
            "gap_warm_vs_baseline_degradation_pct": warm_gap_deg,
            "gap_fix_vs_baseline_raw_pct": fix_gap_raw,
            "gap_fix_vs_baseline_degradation_pct": fix_gap_deg,
            "gap_fix_vs_warm_raw_pct": fix_warm_gap_raw,
            "gap_fix_vs_warm_degradation_pct": fix_warm_gap_deg,
            "baseline_solver_error": baseline["solver_error"],
            "warm_solver_error": warm["solver_error"],
            "fix_solver_error": fixed["solver_error"],
        })

        print(
            f"[{k+1}/{total}] sample={sample_id}, baseline={baseline['status']}, "
            f"warm={warm['status']}, fix={fixed['status']}, high_conf_u={high_count}/{total_count} ({high_ratio:.3f})"
        )

    elapsed = time.time() - t0

    baseline_summary = _summarize_strategy(baseline_results)
    warm_summary = _summarize_strategy(warm_results)
    fix_summary = _summarize_strategy(fix_results)

    warm_base_raw_summary = _summarize_gap(gap_warm_vs_base_raw)
    warm_base_deg_summary = _summarize_gap(gap_warm_vs_base_deg)
    fix_base_raw_summary = _summarize_gap(gap_fix_vs_base_raw)
    fix_base_deg_summary = _summarize_gap(gap_fix_vs_base_deg)
    fix_warm_raw_summary = _summarize_gap(gap_fix_vs_warm_raw)
    fix_warm_deg_summary = _summarize_gap(gap_fix_vs_warm_deg)

    model_tag = args.model_name.replace("\\", "_").replace("/", "_")
    default_csv = f"optimality_compare_highacc_fix_vs_warmstart_{model_tag}_{args.split}.csv"
    out_csv = args.output_csv if args.output_csv else default_csv

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print("\n================ Strategy Summary ================")
    print(f"total_cases: {baseline_summary['total']}")
    print(f"baseline_solvable: {baseline_summary['feasible_count']}, baseline_unsolvable: {baseline_summary['no_solution_count']}")
    print(f"warm_solvable: {warm_summary['feasible_count']}, warm_unsolvable: {warm_summary['no_solution_count']}")
    print(f"fix_solvable: {fix_summary['feasible_count']}, fix_unsolvable: {fix_summary['no_solution_count']}")

    print("\n================ Optimality Gaps (%) ================")
    print(f"warm_vs_baseline_raw_gap_count: {warm_base_raw_summary['count']}")
    print(f"warm_vs_baseline_raw_gap_avg: {warm_base_raw_summary['avg']}")
    print(f"warm_vs_baseline_raw_gap_max: {warm_base_raw_summary['max']}")
    print(f"warm_vs_baseline_degradation_count: {warm_base_deg_summary['count']}")
    print(f"warm_vs_baseline_degradation_avg: {warm_base_deg_summary['avg']}")
    print(f"warm_vs_baseline_degradation_max: {warm_base_deg_summary['max']}")
    print(f"fix_vs_baseline_raw_gap_count: {fix_base_raw_summary['count']}")
    print(f"fix_vs_baseline_raw_gap_avg: {fix_base_raw_summary['avg']}")
    print(f"fix_vs_baseline_raw_gap_max: {fix_base_raw_summary['max']}")
    print(f"fix_vs_baseline_degradation_count: {fix_base_deg_summary['count']}")
    print(f"fix_vs_baseline_degradation_avg: {fix_base_deg_summary['avg']}")
    print(f"fix_vs_baseline_degradation_max: {fix_base_deg_summary['max']}")
    print(f"fix_vs_warm_raw_gap_count: {fix_warm_raw_summary['count']}")
    print(f"fix_vs_warm_raw_gap_avg: {fix_warm_raw_summary['avg']}")
    print(f"fix_vs_warm_raw_gap_max: {fix_warm_raw_summary['max']}")
    print(f"fix_vs_warm_degradation_count: {fix_warm_deg_summary['count']}")
    print(f"fix_vs_warm_degradation_avg: {fix_warm_deg_summary['avg']}")
    print(f"fix_vs_warm_degradation_max: {fix_warm_deg_summary['max']}")

    print("\n================ Solver Status Breakdown ================")
    print(f"baseline_infeasible_no_answer: {baseline_summary['infeasible_no_answer']}")
    print(f"baseline_infeasible_solver_x: {baseline_summary['infeasible_solver_x']}")
    print(f"baseline_error_other: {baseline_summary['error_other']}")
    print(f"warm_infeasible_no_answer: {warm_summary['infeasible_no_answer']}")
    print(f"warm_infeasible_solver_x: {warm_summary['infeasible_solver_x']}")
    print(f"warm_error_other: {warm_summary['error_other']}")
    print(f"fix_infeasible_no_answer: {fix_summary['infeasible_no_answer']}")
    print(f"fix_infeasible_solver_x: {fix_summary['infeasible_solver_x']}")
    print(f"fix_error_other: {fix_summary['error_other']}")
    print(f"elapsed_s: {elapsed:.2f}")
    print(f"csv_saved: {out_csv}")
    print("========================================================")


def get_default_config() -> Dict[str, Any]:
    return {
        "model_name": "units_transformer_all_training_data_case1888rte_pro.npz_random_1000_50000_128_0.0002",
        "model_dir": "module_save",
        "constraints_model_name": "constraints_DNN_all_training_data_case1888rte_pro.mat_random_1000_10000_[256, 256, 256]_0.0002",
        "constraints_model_dir": "module_save",
        "model_data": "",
        "load_data": "",
        "case_json": "case_data/matpower/case1888rte_pro/2017-01-01.json.gz",
        "storage_data": "training_data/storage_data_1888rte.mat",
        "split": "test",
        "threshold": 0.5,
        "reliability_low_gate": 0.1,
        "reliability_high_gate": 0.9,
        "seed": 42,
        "max_samples": 100,
        "batch_size": 32,
        "line_limit_rate": 1.0,
        "use_storage": True,
        "use_new_min_up_down_time": True,
        "new_add_num": 10,
        "add_constr_gate": 0.7,
        "time_limit": 1200.0,
        "mip_gap": 1e-5,
        "output_csv": "",
    }


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_with_config(vars(args))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        run_with_config(get_default_config())
