import argparse
import csv
import gc
import gzip
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from ml_basic_functions import get_data, ml_load_model
from uc_gurobi_constraint_filter import run_constraint_filter


DATA_DIR = "training_data"
MODEL_DIR = "module_save"
CASE_DIR = "case_data"
DEFAULT_BATCH_SIZE = 32
DEFAULT_RELIABILITY_GATES = (0.1, 0.9)
DEFAULT_CASE_RELIABILITY_GATES: Dict[str, Tuple[float, float]] = {
    "case1888rte": (0.05, 0.95),
}
DEFAULT_CASE_EXTRA_GATE_PAIRS: Dict[str, List[Tuple[float, float]]] = {
    "case1888rte": [(0.1, 0.9), (0.15, 0.85)],
}


@dataclass(frozen=True)
class CaseSpec:
    case_label: str
    case_json: str
    units_model_name: str
    constraints_model_name: str
    storage_data_name: Optional[str]
    use_storage: bool
    line_limit_rate: float = 1.0


CASE_SPECS: List[CaseSpec] = [
    CaseSpec(
        case_label="case118",
        case_json=os.path.join(CASE_DIR, "matpower", "case118", "2017-02-01.json.gz"),
        units_model_name="units_transformer_all_training_data_case118_storage_data_case118_30%_random_1000_50000_128_0.0002",
        constraints_model_name="constraints_DNN_all_training_data_case118_storage_data_case118_30%.npz_random_1000_50000_[256, 256, 256]_0.0002",
        storage_data_name="storage_data_case118_30%",
        use_storage=True,
    ),
    CaseSpec(
        case_label="case300",
        case_json=os.path.join(CASE_DIR, "matpower", "case300", "2017-01-01.json.gz"),
        units_model_name="units_transformer_all_training_data_case300_storage_data_case300_60%_random_1000_50000_128_0.0002",
        constraints_model_name="constraints_DNN_all_training_data_case300_storage_data_case300_60%.npz_random_1000_50000_[256, 256, 256]_0.0002",
        storage_data_name="storage_data_case300_60%",
        use_storage=True,
    ),
    CaseSpec(
        case_label="case1354pegase",
        case_json=os.path.join(CASE_DIR, "matpower", "case1354pegase", "2017-01-01.json.gz"),
        units_model_name="units_transformer_all_training_data_case1354pegase_storage_data_case1354pegase_random_1000_50000_128_0.0002",
        constraints_model_name="constraints_DNN_all_training_data_case1354pegase_storage_data_case1354pegase.npz_random_1000_50000_[256, 256, 256]_0.0002",
        storage_data_name="storage_data_case1354pegase",
        use_storage=True,
    ),
    CaseSpec(
        case_label="case1888rte",
        case_json=os.path.join(CASE_DIR, "matpower", "case1888rte_pro", "2017-01-01.json.gz"),
        units_model_name="units_transformer_all_training_data_case1888rte_pro_random_1000_50000_128_0.0002",
        constraints_model_name="constraints_DNN_all_training_data_case1888rte_pro_random_1000_50000_[256, 256, 256]_0.0002",
        storage_data_name="storage_data_case1888rte",
        use_storage=True,
    ),
]


METHOD_ENABLE_ALL = "no warm start + enable all constraints"
METHOD_NO_INITIAL = "no warm start + no initial constraints"
METHOD_PREDICT_CONSTRAINTS = "no warm start + predict constraints"
METHOD_FULL_WARM = "full warm start prediction + predict constraints"
METHOD_HIGH_WARM = "high-confidence warm start prediction + predict constraints"
METHOD_FIX_HIGH = "fix high-confidence prediction+ predict constraints"
METHOD_STRICT_BASELINE = "strict high-confidence warm-start baseline for fixed-gap calculation"

METHOD_ROW_ORDER = [
    METHOD_ENABLE_ALL,
    METHOD_NO_INITIAL,
    METHOD_PREDICT_CONSTRAINTS,
    METHOD_FULL_WARM,
    METHOD_HIGH_WARM,
    METHOD_FIX_HIGH,
]

TABLE_HEADERS = [
    "Case",
    "Method type",
    "Modeling time (s)",
    "Solution time (s)",
    "Total time (s)",
    "Iteration number",
    "Speedup ratio",
    "Gap",
]


@dataclass
class ModelBundle:
    model: torch.nn.Module
    model_type: str
    training_num: int
    test_num: int
    data_is_random: bool
    cutted_by_time: bool
    data_name: str
    data_type: str


@dataclass
class CaseArtifacts:
    spec: CaseSpec
    case_data: Dict[str, Any]
    storage_data: Dict[str, np.ndarray]
    units: ModelBundle
    constraints: ModelBundle
    x_units_test: np.ndarray
    x_constraints_test: np.ndarray
    average_load: float
    nt: int
    ng: int
    nl: int
    init_u: np.ndarray


def _resolve_data_path(data_name: str) -> str:
    candidates = []
    if os.path.isabs(data_name):
        candidates.append(data_name)
    else:
        candidates.append(os.path.join(DATA_DIR, data_name))
    lower_path = candidates[0].lower()
    if lower_path.endswith(".mat"):
        candidates.append(candidates[0][:-4] + ".npz")
    elif not lower_path.endswith(".npz"):
        candidates.append(candidates[0] + ".npz")
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Cannot find data file for '{data_name}'. Tried: {candidates}")


def _load_npz_or_mat(path: str) -> Dict[str, np.ndarray]:
    if path.lower().endswith(".npz"):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return dict(np.load(path, allow_pickle=True))
    npz_path = os.path.splitext(path)[0] + ".npz"
    if os.path.exists(npz_path):
        return dict(np.load(npz_path, allow_pickle=True))
    import scipy.io

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return scipy.io.loadmat(path)


def _load_storage(storage_data_name: Optional[str], use_storage: bool) -> Dict[str, np.ndarray]:
    if not use_storage:
        return {}
    if not storage_data_name:
        raise ValueError("use_storage=True but storage_data_name is empty.")
    return _load_npz_or_mat(_resolve_data_path(storage_data_name))


def _load_case_json(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _load_model_bundle(model_name: str, expected_data_type: str) -> ModelBundle:
    model, model_type, training_num, test_num, data_is_random, cutted_by_time, data_name, data_type = ml_load_model(
        model_name, MODEL_DIR
    )
    if data_type != expected_data_type:
        raise ValueError(f"Model {model_name} has data_type={data_type}, expected {expected_data_type}.")
    return ModelBundle(
        model=model,
        model_type=str(model_type),
        training_num=int(training_num),
        test_num=int(test_num),
        data_is_random=bool(data_is_random),
        cutted_by_time=bool(cutted_by_time),
        data_name=str(data_name),
        data_type=str(data_type),
    )


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _get_case_info(case_data: Dict[str, Any]) -> Tuple[int, int, int, np.ndarray]:
    _, units = zip(*case_data["Generators"].items())
    _, lines = zip(*case_data["Transmission lines"].items()) if "Transmission lines" in case_data else ([], [])
    time_horizon = case_data["Parameters"].get("Time horizon (h)") or case_data["Parameters"].get("Time (h)")
    time_step = case_data["Parameters"].get("Time step (min)", 60)
    if 60 % time_step != 0:
        raise ValueError(f"Unsupported time step: {time_step} min.")
    time_multiplier = int(60 / time_step)
    nt = int(time_horizon * time_multiplier)
    ng = len(units)
    nl = len(lines)
    initial_status = np.array([units[i]["Initial status (h)"] * time_multiplier for i in range(ng)])
    init_u = (initial_status > 0).astype(np.int32)
    return nt, ng, nl, init_u


def _load_test_inputs(bundle: ModelBundle, recognize_constr: bool) -> Tuple[np.ndarray, float, int]:
    data_path = _resolve_data_path(bundle.data_name)
    (
        _train_inputs,
        _train_targets,
        _val_inputs,
        _val_targets,
        test_inputs,
        _test_targets,
        time_num,
        _input_size,
        _output_size,
        average_load,
        _train_size,
    ) = get_data(
        bundle.model_type,
        data_path,
        test_num=bundle.test_num,
        training_num=bundle.training_num,
        data_is_random=bundle.data_is_random,
        cutted_by_time=bundle.cutted_by_time,
        recognize_constr=recognize_constr,
        random_seed=42,
        val_ratio=0.1,
    )
    return _as_numpy(test_inputs).astype(np.float32, copy=False), float(average_load), int(time_num)


def _load_case_artifacts(spec: CaseSpec) -> CaseArtifacts:
    units = _load_model_bundle(spec.units_model_name, expected_data_type="units")
    constraints = _load_model_bundle(spec.constraints_model_name, expected_data_type="constraints")
    case_data = _load_case_json(spec.case_json)
    storage_data = _load_storage(spec.storage_data_name, spec.use_storage)
    nt, ng, nl, init_u = _get_case_info(case_data)

    x_units_test, average_load_u, time_num_u = _load_test_inputs(units, recognize_constr=False)
    x_constraints_test, average_load_c, time_num_c = _load_test_inputs(constraints, recognize_constr=True)

    if time_num_u != nt:
        raise ValueError(f"{spec.case_label}: units data time_num={time_num_u}, case NT={nt}.")
    if time_num_c != nt:
        raise ValueError(f"{spec.case_label}: constraints data time_num={time_num_c}, case NT={nt}.")
    if len(x_units_test) != len(x_constraints_test):
        raise ValueError(
            f"{spec.case_label}: units test samples={len(x_units_test)}, "
            f"constraints test samples={len(x_constraints_test)}."
        )
    if abs(average_load_u - average_load_c) > max(1e-6, abs(average_load_u) * 1e-6):
        print(
            f"[warning] {spec.case_label}: units average_load={average_load_u}, "
            f"constraints average_load={average_load_c}. The units value is used."
        )

    return CaseArtifacts(
        spec=spec,
        case_data=case_data,
        storage_data=storage_data,
        units=units,
        constraints=constraints,
        x_units_test=x_units_test,
        x_constraints_test=x_constraints_test,
        average_load=average_load_u,
        nt=nt,
        ng=ng,
        nl=nl,
        init_u=init_u,
    )


def _predict_probabilities(
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.as_tensor(x[start:end], dtype=torch.float32, device=device)
            outputs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _ensure_ntg_shape(pred: np.ndarray, nt: int, ng: int) -> np.ndarray:
    if pred.ndim == 3 and pred.shape[1] == nt and pred.shape[2] == ng:
        return pred
    if pred.ndim == 2 and pred.shape[1] == nt * ng:
        return pred.reshape(pred.shape[0], nt, ng)
    raise ValueError(f"Cannot map units prediction shape {pred.shape} to (N,{nt},{ng}).")


def _prepare_constraints_input(x: np.ndarray, model_type: str) -> np.ndarray:
    if model_type in ("DNN", "KAN"):
        return x.reshape(x.shape[0], -1)
    return x


def _ensure_constraints_shape(pred: np.ndarray, nl: int) -> np.ndarray:
    if pred.ndim == 2 and pred.shape[1] == nl:
        return pred
    raise ValueError(f"Cannot map constraints prediction shape {pred.shape} to (N,{nl}).")


def _to_g_t_shape(x_2d: np.ndarray, ng: int) -> np.ndarray:
    if x_2d.ndim != 2:
        raise ValueError(f"Expected 2-D unit matrix, got shape {x_2d.shape}.")
    if x_2d.shape[0] == ng:
        return x_2d
    if x_2d.shape[1] == ng:
        return x_2d.T
    raise ValueError(f"Cannot map shape {x_2d.shape} to (G,T) with NG={ng}.")


def _build_dicts_from_u_and_mask(
    u_pred_2d: np.ndarray,
    mask_2d: np.ndarray,
    init_u: np.ndarray,
) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
    ng = len(init_u)
    u_gt = _to_g_t_shape(u_pred_2d, ng).astype(np.int32)
    mask_gt = _to_g_t_shape(mask_2d.astype(bool), ng)
    if u_gt.shape != mask_gt.shape:
        raise ValueError(f"u/mask shape mismatch: {u_gt.shape} vs {mask_gt.shape}.")

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
            y_start[(g, 0)] = 1 if delta0 == 1 else 0
            z_start[(g, 0)] = 1 if delta0 == -1 else 0

        for t in range(1, nt):
            if mask_gt[g, t] and mask_gt[g, t - 1]:
                delta = int(u_gt[g, t] - u_gt[g, t - 1])
                y_start[(g, t)] = 1 if delta == 1 else 0
                z_start[(g, t)] = 1 if delta == -1 else 0

    return u_start, y_start, z_start


def _build_high_conf_dicts(
    u_pred_2d: np.ndarray,
    prob_2d: np.ndarray,
    init_u: np.ndarray,
    low_gate: float,
    high_gate: float,
) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], int, int]:
    ng = len(init_u)
    prob_gt = _to_g_t_shape(prob_2d, ng)
    mask = np.logical_or(prob_gt <= low_gate, prob_gt >= high_gate)
    u_start, y_start, z_start = _build_dicts_from_u_and_mask(u_pred_2d, mask, init_u)
    return u_start, y_start, z_start, int(mask.sum()), int(mask.size)


def _safe_gap_pct(ref_obj: float, other_obj: float) -> float:
    denom = abs(ref_obj)
    if denom < 1e-12:
        return float("nan")
    return 100.0 * (other_obj - ref_obj) / denom


def _degradation_pct(ref_obj: float, other_obj: float) -> float:
    gap = _safe_gap_pct(ref_obj, other_obj)
    if not np.isfinite(gap):
        return float("nan")
    return max(0.0, gap)


def _combine_fixed_gap_pct(degradation_pct: float, reference_gap_pct: float) -> float:
    if not np.isfinite(degradation_pct) or not np.isfinite(reference_gap_pct):
        return float("nan")
    return max(0.0, float(degradation_pct)) + max(0.0, float(reference_gap_pct))


def _get_model_mip_gap_pct(model: Any) -> float:
    try:
        gap = float(model.MIPGap)
    except Exception:  # pylint: disable=broad-except
        return float("nan")
    if not np.isfinite(gap):
        return float("nan")
    return max(0.0, 100.0 * gap)


def _solve_strategy(
    *,
    case_data: Dict[str, Any],
    load_bt: np.ndarray,
    storage_data: Dict[str, np.ndarray],
    initial_constraints: np.ndarray,
    is_warm_start: bool,
    use_initial_enable: bool,
    fix_int: bool,
    u_start: Dict[Tuple[int, int], int],
    y_start: Dict[Tuple[int, int], int],
    z_start: Dict[Tuple[int, int], int],
    use_storage: bool,
    line_limit_rate: float,
    new_add_num: int,
    add_constr_gate: float,
    time_limit: float,
    mip_gap: float,
) -> Dict[str, Any]:
    try:
        (
            model,
            result,
            _u,
            _y,
            _z,
            _p,
            _r,
            _c,
            _l,
            _enable_constraints,
            model_time,
            solve_t,
            load_data_t,
            round_num,
            no_answer,
        ) = run_constraint_filter(
            case_data,
            is_warm_start=is_warm_start,
            use_new_input=True,
            use_storage=use_storage,
            use_initial_enable=use_initial_enable,
            add_more_constr=False,
            use_new_min_up_down_time=True,
            use_solar=False,
            u_start=u_start,
            y_start=y_start,
            z_start=z_start,
            new_input=load_bt,
            storage_data=storage_data if use_storage else 0,
            line_limit_rate=line_limit_rate,
            solar_gen=0,
            new_add_num=new_add_num,
            initial_constraints=initial_constraints,
            add_constr_gate=add_constr_gate,
            fix_int=fix_int,
            time_limit=time_limit,
            mip_gap=mip_gap,
        )
        feasible = 0 if bool(no_answer) else 1
        return {
            "status": "feasible" if feasible else "no_answer",
            "feasible": feasible,
            "objective": float(result) if feasible else float("nan"),
            "actual_mip_gap_pct": _get_model_mip_gap_pct(model) if feasible else float("nan"),
            "model_time_s": float(model_time),
            "solve_time_s": float(solve_t),
            "load_data_time_s": float(load_data_t),
            "iteration_num": float(round_num),
            "solver_error": "",
        }
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "status": "error",
            "feasible": 0,
            "objective": float("nan"),
            "actual_mip_gap_pct": float("nan"),
            "model_time_s": float("nan"),
            "solve_time_s": float("nan"),
            "load_data_time_s": float("nan"),
            "iteration_num": float("nan"),
            "solver_error": str(exc),
        }


def _finite_mean(values: Iterable[Any]) -> float:
    arr = np.array([float(v) for v in values if v is not None and v != ""], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _format_float(value: float, digits: int = 4) -> Any:
    if not np.isfinite(value):
        return ""
    return round(float(value), digits)


def _format_solver_gap_limit(mip_gap: float) -> str:
    pct = 100.0 * float(mip_gap)
    if pct >= 0.01:
        text = f"{pct:.3f}".rstrip("0").rstrip(".")
    else:
        text = f"{pct:.6f}".rstrip("0").rstrip(".")
    return f"<{text}%"


def _format_actual_gap(gap_pct: float) -> str:
    if not np.isfinite(gap_pct):
        return "N/A"
    return f"{gap_pct:.3g}%"


def _build_raw_record(
    spec: CaseSpec,
    sample_id: int,
    method: str,
    result: Dict[str, Any],
    mip_gap: float,
    gap_pct: float,
    high_conf_ratio: float,
    degradation_pct: Optional[float] = None,
    reference_gap_pct: Optional[float] = None,
) -> Dict[str, Any]:
    model_time = float(result["model_time_s"]) if np.isfinite(result["model_time_s"]) else float("nan")
    solve_time = float(result["solve_time_s"]) if np.isfinite(result["solve_time_s"]) else float("nan")
    total_time = model_time + solve_time if np.isfinite(model_time) and np.isfinite(solve_time) else float("nan")
    if degradation_pct is None:
        degradation_pct = gap_pct
    if reference_gap_pct is None:
        reference_gap_pct = float("nan")
    return {
        "Case": spec.case_label,
        "Sample id": sample_id,
        "Method type": method,
        "status": result["status"],
        "feasible": result["feasible"],
        "objective": result["objective"],
        "mip_gap": mip_gap,
        "gap_pct_vs_strict_baseline": degradation_pct,
        "degradation_pct_vs_strict_baseline": degradation_pct,
        "reference_gap_pct_for_fixed_gap": reference_gap_pct,
        "reported_gap_pct": gap_pct,
        "actual_mip_gap_pct": result.get("actual_mip_gap_pct", float("nan")),
        "high_conf_ratio": high_conf_ratio,
        "model_time_s": model_time,
        "solve_time_s": solve_time,
        "total_time_s": total_time,
        "load_data_time_s": result["load_data_time_s"],
        "iteration_num": result["iteration_num"],
        "solver_error": result["solver_error"],
    }


def _aggregate_case_rows(
    spec: CaseSpec,
    raw_rows: List[Dict[str, Any]],
    default_mip_gap: float,
) -> List[Dict[str, Any]]:
    baseline_rows = [r for r in raw_rows if r["Method type"] == METHOD_ENABLE_ALL]
    baseline_solve = _finite_mean(r["solve_time_s"] for r in baseline_rows)
    aggregated: List[Dict[str, Any]] = []

    present_methods = {r["Method type"] for r in raw_rows}
    extra_high_warm_methods: List[str] = []
    extra_fix_high_methods: List[str] = []
    other_extra_methods: List[str] = []
    for row in raw_rows:
        method = row["Method type"]
        if method in METHOD_ROW_ORDER:
            continue
        if method.startswith(f"{METHOD_HIGH_WARM} ("):
            if method not in extra_high_warm_methods:
                extra_high_warm_methods.append(method)
        elif method.startswith(f"{METHOD_FIX_HIGH} ("):
            if method not in extra_fix_high_methods:
                extra_fix_high_methods.append(method)
        elif method not in other_extra_methods:
            other_extra_methods.append(method)

    methods_in_order: List[str] = []
    for method in METHOD_ROW_ORDER:
        if method in present_methods:
            methods_in_order.append(method)
        if method == METHOD_HIGH_WARM:
            methods_in_order.extend(extra_high_warm_methods)
        if method == METHOD_FIX_HIGH:
            methods_in_order.extend(extra_fix_high_methods)
    methods_in_order.extend(other_extra_methods)

    for method in methods_in_order:
        method_rows = [r for r in raw_rows if r["Method type"] == method]
        model_time = _finite_mean(r["model_time_s"] for r in method_rows)
        solve_time = _finite_mean(r["solve_time_s"] for r in method_rows)
        total_time = _finite_mean(r["total_time_s"] for r in method_rows)
        iteration_num = _finite_mean(r["iteration_num"] for r in method_rows)
        speedup = baseline_solve / solve_time if np.isfinite(baseline_solve) and np.isfinite(solve_time) and solve_time > 0 else float("nan")
        if method == METHOD_FIX_HIGH or method.startswith(f"{METHOD_FIX_HIGH} ("):
            gap_text = _format_actual_gap(
                _finite_mean(
                    r.get("reported_gap_pct", r.get("gap_pct_vs_strict_baseline", float("nan")))
                    for r in method_rows
                )
            )
        else:
            gap_text = _format_solver_gap_limit(default_mip_gap)
        aggregated.append(
            {
                "Case": spec.case_label,
                "Method type": method,
                "Modeling time (s)": _format_float(model_time),
                "Solution time (s)": _format_float(solve_time),
                "Total time (s)": _format_float(total_time),
                "Iteration number": _format_float(iteration_num, digits=2),
                "Speedup ratio": _format_float(speedup, digits=2),
                "Gap": gap_text,
            }
        )
    return aggregated


def _parse_gate_pair(text: str) -> Tuple[float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Gate pair must be formatted as low,high, for example 0.05,0.95.")
    low, high = float(parts[0]), float(parts[1])
    if low < 0 or high > 1 or low >= high:
        raise argparse.ArgumentTypeError("Gate pair must satisfy 0 <= low < high <= 1.")
    return low, high


def _merge_gate_pairs(*groups: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    merged: List[Tuple[float, float]] = []
    seen = set()
    for group in groups:
        for low, high in group:
            key = (round(float(low), 12), round(float(high), 12))
            if key in seen:
                continue
            seen.add(key)
            merged.append((float(low), float(high)))
    return merged


def _resolve_reliability_gates(spec: CaseSpec, args: argparse.Namespace) -> Tuple[float, float]:
    low_override = args.reliability_low_gate
    high_override = args.reliability_high_gate
    if (low_override is None) != (high_override is None):
        raise ValueError("--reliability-low-gate and --reliability-high-gate must be set together.")
    if low_override is not None and high_override is not None:
        return float(low_override), float(high_override)
    return DEFAULT_CASE_RELIABILITY_GATES.get(spec.case_label, DEFAULT_RELIABILITY_GATES)


def _validate_gate_pair(low: float, high: float, name: str) -> None:
    if low < 0 or high > 1 or low >= high:
        raise ValueError(f"{name} must satisfy 0 <= low < high <= 1, got low={low}, high={high}.")


def _selected_sample_indices(total_available: int, sample_start: int, sample_count: int) -> List[int]:
    if sample_start < 0:
        raise ValueError("--sample-start must be >= 0.")
    if sample_start >= total_available:
        raise ValueError(f"--sample-start={sample_start} exceeds available test samples={total_available}.")
    if sample_count <= 0:
        return list(range(sample_start, total_available))
    end = min(total_available, sample_start + sample_count)
    return list(range(sample_start, end))


def _build_result_rows_for_case(
    spec: CaseSpec,
    sample_count: int,
    sample_start: int,
    threshold: float,
    reliability_low_gate: float,
    reliability_high_gate: float,
    extra_gate_pairs: List[Tuple[float, float]],
    default_mip_gap: float,
    strict_mip_gap: float,
    time_limit: float,
    new_add_num: int,
    add_constr_gate: float,
    batch_size: int,
    dry_run: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    artifacts = _load_case_artifacts(spec)
    indices = _selected_sample_indices(len(artifacts.x_units_test), sample_start, sample_count)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    units_model = artifacts.units.model.to(device)
    constraints_model = artifacts.constraints.model.to(device)

    print(f"[{spec.case_label}] device={device}, selected base-test samples={len(indices)}, test_available={len(artifacts.x_units_test)}")
    print(f"[{spec.case_label}] units_data={artifacts.units.data_name}, constraints_data={artifacts.constraints.data_name}")
    print(f"[{spec.case_label}] use_storage={spec.use_storage}, storage={spec.storage_data_name}")

    x_units_selected = artifacts.x_units_test[indices]
    x_constraints_selected = artifacts.x_constraints_test[indices]
    unit_probs = _predict_probabilities(units_model, x_units_selected, device, batch_size)
    unit_probs_ntg = _ensure_ntg_shape(unit_probs, artifacts.nt, artifacts.ng)
    unit_binary = (unit_probs_ntg >= threshold).astype(np.int32)
    constraints_input = _prepare_constraints_input(x_constraints_selected, artifacts.constraints.model_type)
    constraints_probs = _predict_probabilities(constraints_model, constraints_input, device, batch_size)
    constraints_init = _ensure_constraints_shape(constraints_probs, artifacts.nl)

    if dry_run:
        high_mask = np.logical_or(unit_probs_ntg <= reliability_low_gate, unit_probs_ntg >= reliability_high_gate)
        return [], [], {
            "case_label": spec.case_label,
            "selected_samples": len(indices),
            "nt": artifacts.nt,
            "ng": artifacts.ng,
            "nl": artifacts.nl,
            "avg_high_conf_ratio": float(np.mean(high_mask)),
            "dry_run": True,
        }

    raw_rows: List[Dict[str, Any]] = []
    all_constraints = np.ones(artifacts.nl, dtype=np.float32)
    no_constraints = np.zeros(artifacts.nl, dtype=np.float32)

    for rank, local_idx in enumerate(range(len(indices)), start=1):
        sample_id = indices[local_idx]
        load_bt = x_units_selected[local_idx].T * artifacts.average_load
        predicted_constraints = constraints_init[local_idx]

        full_mask = np.ones_like(unit_binary[local_idx], dtype=bool)
        u_full, y_full, z_full = _build_dicts_from_u_and_mask(unit_binary[local_idx], full_mask, artifacts.init_u)
        u_high, y_high, z_high, high_count, high_total = _build_high_conf_dicts(
            unit_binary[local_idx],
            unit_probs_ntg[local_idx],
            artifacts.init_u,
            reliability_low_gate,
            reliability_high_gate,
        )
        high_conf_ratio = high_count / high_total if high_total else float("nan")
        solve_kwargs = {
            "case_data": artifacts.case_data,
            "load_bt": load_bt,
            "storage_data": artifacts.storage_data,
            "use_storage": spec.use_storage,
            "line_limit_rate": spec.line_limit_rate,
            "new_add_num": new_add_num,
            "add_constr_gate": add_constr_gate,
            "time_limit": time_limit,
        }

        baseline_default = _solve_strategy(
            **solve_kwargs,
            initial_constraints=all_constraints,
            is_warm_start=False,
            use_initial_enable=True,
            fix_int=False,
            u_start={},
            y_start={},
            z_start={},
            mip_gap=default_mip_gap,
        )
        raw_rows.append(_build_raw_record(spec, sample_id, METHOD_ENABLE_ALL, baseline_default, default_mip_gap, float("nan"), high_conf_ratio))

        no_initial = _solve_strategy(
            **solve_kwargs,
            initial_constraints=no_constraints,
            is_warm_start=False,
            use_initial_enable=False,
            fix_int=False,
            u_start={},
            y_start={},
            z_start={},
            mip_gap=default_mip_gap,
        )
        raw_rows.append(_build_raw_record(spec, sample_id, METHOD_NO_INITIAL, no_initial, default_mip_gap, float("nan"), high_conf_ratio))

        predict_constraints = _solve_strategy(
            **solve_kwargs,
            initial_constraints=predicted_constraints,
            is_warm_start=False,
            use_initial_enable=True,
            fix_int=False,
            u_start={},
            y_start={},
            z_start={},
            mip_gap=default_mip_gap,
        )
        raw_rows.append(_build_raw_record(spec, sample_id, METHOD_PREDICT_CONSTRAINTS, predict_constraints, default_mip_gap, float("nan"), high_conf_ratio))

        full_warm = _solve_strategy(
            **solve_kwargs,
            initial_constraints=predicted_constraints,
            is_warm_start=True,
            use_initial_enable=True,
            fix_int=False,
            u_start=u_full,
            y_start=y_full,
            z_start=z_full,
            mip_gap=default_mip_gap,
        )
        raw_rows.append(_build_raw_record(spec, sample_id, METHOD_FULL_WARM, full_warm, default_mip_gap, float("nan"), high_conf_ratio))

        high_warm = _solve_strategy(
            **solve_kwargs,
            initial_constraints=predicted_constraints,
            is_warm_start=True,
            use_initial_enable=True,
            fix_int=False,
            u_start=u_high,
            y_start=y_high,
            z_start=z_high,
            mip_gap=default_mip_gap,
        )
        raw_rows.append(_build_raw_record(spec, sample_id, METHOD_HIGH_WARM, high_warm, default_mip_gap, float("nan"), high_conf_ratio))

        for low, high in extra_gate_pairs:
            u_extra, y_extra, z_extra, extra_count, extra_total = _build_high_conf_dicts(
                unit_binary[local_idx], unit_probs_ntg[local_idx], artifacts.init_u, low, high
            )
            extra_result = _solve_strategy(
                **solve_kwargs,
                initial_constraints=predicted_constraints,
                is_warm_start=True,
                use_initial_enable=True,
                fix_int=False,
                u_start=u_extra,
                y_start=y_extra,
                z_start=z_extra,
                mip_gap=default_mip_gap,
            )
            extra_label = f"{METHOD_HIGH_WARM} (g^low = {low:g}, g^high = {high:g})"
            extra_ratio = extra_count / extra_total if extra_total else float("nan")
            raw_rows.append(_build_raw_record(spec, sample_id, extra_label, extra_result, default_mip_gap, float("nan"), extra_ratio))

        # This strict reference is used only for the fixed-method objective gap.
        # It has the same optimal objective as the all-constraint baseline after
        # constraint filtering, but is much faster because it starts from the
        # predicted constraints and high-confidence warm start.
        strict_baseline = _solve_strategy(
            **solve_kwargs,
            initial_constraints=predicted_constraints,
            is_warm_start=True,
            use_initial_enable=True,
            fix_int=False,
            u_start=u_high,
            y_start=y_high,
            z_start=z_high,
            mip_gap=strict_mip_gap,
        )
        raw_rows.append(_build_raw_record(spec, sample_id, METHOD_STRICT_BASELINE, strict_baseline, strict_mip_gap, float("nan"), high_conf_ratio))
        strict_baseline_gap_pct = float(strict_baseline.get("actual_mip_gap_pct", float("nan")))

        fixed = _solve_strategy(
            **solve_kwargs,
            initial_constraints=predicted_constraints,
            is_warm_start=False,
            use_initial_enable=True,
            fix_int=True,
            u_start=u_high,
            y_start=y_high,
            z_start=z_high,
            mip_gap=strict_mip_gap,
        )
        fixed_degradation = float("nan")
        fixed_gap = float("nan")
        if strict_baseline["feasible"] == 1 and fixed["feasible"] == 1:
            fixed_degradation = _degradation_pct(float(strict_baseline["objective"]), float(fixed["objective"]))
            fixed_gap = _combine_fixed_gap_pct(fixed_degradation, strict_baseline_gap_pct)
        raw_rows.append(
            _build_raw_record(
                spec,
                sample_id,
                METHOD_FIX_HIGH,
                fixed,
                strict_mip_gap,
                fixed_gap,
                high_conf_ratio,
                degradation_pct=fixed_degradation,
                reference_gap_pct=strict_baseline_gap_pct,
            )
        )

        extra_fixed_summaries: List[str] = []
        for low, high in extra_gate_pairs:
            u_extra, y_extra, z_extra, extra_count, extra_total = _build_high_conf_dicts(
                unit_binary[local_idx], unit_probs_ntg[local_idx], artifacts.init_u, low, high
            )
            fixed_extra = _solve_strategy(
                **solve_kwargs,
                initial_constraints=predicted_constraints,
                is_warm_start=False,
                use_initial_enable=True,
                fix_int=True,
                u_start=u_extra,
                y_start=y_extra,
                z_start=z_extra,
                mip_gap=strict_mip_gap,
            )
            fixed_extra_degradation = float("nan")
            fixed_extra_gap = float("nan")
            if strict_baseline["feasible"] == 1 and fixed_extra["feasible"] == 1:
                fixed_extra_degradation = _degradation_pct(float(strict_baseline["objective"]), float(fixed_extra["objective"]))
                fixed_extra_gap = _combine_fixed_gap_pct(fixed_extra_degradation, strict_baseline_gap_pct)
            fixed_extra_label = f"{METHOD_FIX_HIGH} (g^low = {low:g}, g^high = {high:g})"
            fixed_extra_ratio = extra_count / extra_total if extra_total else float("nan")
            raw_rows.append(
                _build_raw_record(
                    spec,
                    sample_id,
                    fixed_extra_label,
                    fixed_extra,
                    strict_mip_gap,
                    fixed_extra_gap,
                    fixed_extra_ratio,
                    degradation_pct=fixed_extra_degradation,
                    reference_gap_pct=strict_baseline_gap_pct,
                )
            )
            gap_text = fixed_extra_gap if np.isfinite(fixed_extra_gap) else "N/A"
            extra_fixed_summaries.append(f"{low:g}/{high:g}: gap={gap_text}")

        print(
            f"[{spec.case_label}] sample {rank}/{len(indices)} idx={sample_id}: "
            f"HC={high_count}/{high_total}, baseline_solve={baseline_default['solve_time_s']}, "
            f"fixed_gap={fixed_gap if np.isfinite(fixed_gap) else 'N/A'}"
            + (f", extra_fixed={'; '.join(extra_fixed_summaries)}" if extra_fixed_summaries else "")
        )

    table_source_rows = [r for r in raw_rows if r["Method type"] != METHOD_STRICT_BASELINE]
    aggregated_rows = _aggregate_case_rows(spec, table_source_rows, default_mip_gap)
    summary = {
        "case_label": spec.case_label,
        "selected_samples": len(indices),
        "nt": artifacts.nt,
        "ng": artifacts.ng,
        "nl": artifacts.nl,
        "avg_high_conf_ratio": _finite_mean(r["high_conf_ratio"] for r in table_source_rows),
        "dry_run": False,
    }
    return aggregated_rows, raw_rows, summary


def _apply_table_style(ws, start_row: int, end_row: int, start_col: int, end_col: int) -> None:
    thin = Side(style="thin", color="000000")
    medium = Side(style="medium", color="000000")
    for row in ws.iter_rows(min_row=start_row, max_row=end_row, min_col=start_col, max_col=end_col):
        for cell in row:
            cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for c in range(start_col, end_col + 1):
        ws.cell(start_row, c).border = Border(left=thin, right=thin, top=medium, bottom=thin)
        ws.cell(end_row, c).border = Border(left=thin, right=thin, top=thin, bottom=medium)


def _write_excel(rows: List[Dict[str, Any]], out_path: str, sample_count_text: str) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "TABLE IV"

    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    bold_font = Font(bold=True)
    title_font = Font(bold=True, size=14)
    header_fill = PatternFill("solid", fgColor="F2F2F2")

    ws["A1"] = "TABLE IV"
    ws["A2"] = f"Optimization Efficiency Comparison of Different Methods (base-test samples={sample_count_text})"
    ws.merge_cells("A1:H1")
    ws.merge_cells("A2:H2")
    ws["A1"].font = title_font
    ws["A2"].font = bold_font
    ws["A1"].alignment = center
    ws["A2"].alignment = center

    for col_idx, header in enumerate(TABLE_HEADERS, start=1):
        cell = ws.cell(4, col_idx)
        cell.value = header
        cell.font = bold_font
        cell.fill = header_fill
        cell.alignment = center

    current_row = 5
    case_start_rows: Dict[str, int] = {}
    case_end_rows: Dict[str, int] = {}
    for row in rows:
        case = row["Case"]
        if case not in case_start_rows:
            case_start_rows[case] = current_row
        for col_idx, header in enumerate(TABLE_HEADERS, start=1):
            cell = ws.cell(current_row, col_idx)
            cell.value = row.get(header, "")
            cell.alignment = center
        case_end_rows[case] = current_row
        current_row += 1

    for case, start in case_start_rows.items():
        end = case_end_rows[case]
        if end > start:
            ws.merge_cells(start_row=start, start_column=1, end_row=end, end_column=1)
        ws.cell(start, 1).alignment = center
        ws.cell(start, 1).font = bold_font

    for case, start in case_start_rows.items():
        end = case_end_rows[case]
        case_rows = rows[(start - 5) : (end - 4)]
        for offset, row in enumerate(case_rows):
            if row["Method type"] == METHOD_HIGH_WARM:
                for c in range(2, 9):
                    ws.cell(start + offset, c).font = bold_font

    _apply_table_style(ws, 4, ws.max_row, 1, 8)
    widths = {1: 18, 2: 48, 3: 16, 4: 16, 5: 15, 6: 15, 7: 14, 8: 12}
    for idx, width in widths.items():
        ws.column_dimensions[get_column_letter(idx)].width = width
    ws.row_dimensions[1].height = 24
    ws.row_dimensions[2].height = 30
    ws.row_dimensions[4].height = 30
    wb.save(out_path)


def _write_raw_csv(raw_rows: List[Dict[str, Any]], out_path: str) -> None:
    if not raw_rows:
        return
    fieldnames = list(raw_rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow(row)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TABLE IV optimization-efficiency Excel table.")
    parser.add_argument("--sample-count", type=int, default=5, help="Number of base test samples per case. <=0 means all available test samples.")
    parser.add_argument("--sample-start", type=int, default=0, help="Start index within the base test split.")
    parser.add_argument("--output", type=str, default="table_IV_optimization_efficiency.xlsx")
    parser.add_argument("--raw-output", type=str, default="", help="Optional CSV path for per-sample raw results.")
    parser.add_argument("--case", dest="cases", action="append", default=[], help="Case label filter; may be repeated.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold for unit commitment predictions.")
    parser.add_argument(
        "--reliability-low-gate",
        type=float,
        default=None,
        help="Optional global override. If omitted, case1888rte uses 0.05/0.95 and other cases use 0.1/0.9.",
    )
    parser.add_argument(
        "--reliability-high-gate",
        type=float,
        default=None,
        help="Optional global override. Must be set together with --reliability-low-gate.",
    )
    parser.add_argument(
        "--extra-high-confidence-gate",
        dest="extra_gate_pairs",
        action="append",
        type=_parse_gate_pair,
        default=[],
        help=(
            "Optional extra high-confidence gate, formatted as low,high. "
            "For case1888rte, extra gates are evaluated for both high-confidence warm start "
            "and fixed high-confidence prediction. May be repeated."
        ),
    )
    parser.add_argument("--default-mip-gap", type=float, default=1e-3, help="Default MIPGap; 1e-3 means 0.1%.")
    parser.add_argument("--strict-mip-gap", type=float, default=1e-5, help="MIPGap for fixed high-confidence evaluation; 1e-5 means 0.001%.")
    parser.add_argument("--time-limit", type=float, default=1200.0)
    parser.add_argument("--new-add-num", type=int, default=10)
    parser.add_argument("--add-constr-gate", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gurobi-output-flag", type=int, default=0)
    parser.add_argument("--gurobi-threads", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Load models/data and run ML predictions, but skip UC solves.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if (args.reliability_low_gate is None) != (args.reliability_high_gate is None):
        raise ValueError("--reliability-low-gate and --reliability-high-gate must be set together.")
    if args.reliability_low_gate is not None:
        _validate_gate_pair(float(args.reliability_low_gate), float(args.reliability_high_gate), "reliability gates")
    if args.default_mip_gap <= 0 or args.strict_mip_gap <= 0:
        raise ValueError("MIP gaps must be positive.")

    os.environ["UC_GUROBI_OUTPUTFLAG"] = str(args.gurobi_output_flag)
    if args.gurobi_threads > 0:
        os.environ["UC_GUROBI_THREADS"] = str(args.gurobi_threads)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    selected_specs = CASE_SPECS
    if args.cases:
        requested = set(args.cases)
        selected_specs = [spec for spec in CASE_SPECS if spec.case_label in requested]
        missing = requested - {spec.case_label for spec in selected_specs}
        if missing:
            raise ValueError(f"Unknown case label(s): {sorted(missing)}")

    print("================ TABLE IV generation settings ================")
    print(f"cases: {[spec.case_label for spec in selected_specs]}")
    print(f"sample_start: {args.sample_start}")
    print(f"sample_count: {args.sample_count if args.sample_count > 0 else 'all'}")
    print(f"default_mip_gap: {args.default_mip_gap} ({_format_solver_gap_limit(args.default_mip_gap)})")
    print(f"strict_mip_gap: {args.strict_mip_gap} ({_format_solver_gap_limit(args.strict_mip_gap)})")
    if args.reliability_low_gate is not None:
        print(f"high_conf_gate: global override [{args.reliability_low_gate}, {args.reliability_high_gate}]")
    else:
        print(f"high_conf_gate: case-specific defaults {DEFAULT_CASE_RELIABILITY_GATES}, default={DEFAULT_RELIABILITY_GATES}")
    print(f"extra_high_conf_gates: {args.extra_gate_pairs}")
    print("================================================================")

    all_table_rows: List[Dict[str, Any]] = []
    all_raw_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for spec in selected_specs:
        reliability_low_gate, reliability_high_gate = _resolve_reliability_gates(spec, args)
        _validate_gate_pair(reliability_low_gate, reliability_high_gate, f"{spec.case_label} reliability gates")
        print(f"[{spec.case_label}] using high-confidence gate: [{reliability_low_gate}, {reliability_high_gate}]")
        case_extra_gate_pairs = []
        if spec.case_label == "case1888rte":
            case_extra_gate_pairs = _merge_gate_pairs(
                DEFAULT_CASE_EXTRA_GATE_PAIRS.get(spec.case_label, []),
                args.extra_gate_pairs,
            )
        table_rows, raw_rows, summary = _build_result_rows_for_case(
            spec=spec,
            sample_count=args.sample_count,
            sample_start=args.sample_start,
            threshold=args.threshold,
            reliability_low_gate=reliability_low_gate,
            reliability_high_gate=reliability_high_gate,
            extra_gate_pairs=case_extra_gate_pairs,
            default_mip_gap=args.default_mip_gap,
            strict_mip_gap=args.strict_mip_gap,
            time_limit=args.time_limit,
            new_add_num=args.new_add_num,
            add_constr_gate=args.add_constr_gate,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
        all_table_rows.extend(table_rows)
        all_raw_rows.extend(raw_rows)
        summaries.append(summary)
        print(
            f"[summary] {summary['case_label']}: samples={summary['selected_samples']}, "
            f"NT={summary['nt']}, NG={summary['ng']}, NL={summary['nl']}, "
            f"avg_HC_ratio={summary['avg_high_conf_ratio']:.4f}"
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.dry_run:
        print("Dry run completed. No UC solves or Excel output were generated.")
        return
    if not all_table_rows:
        raise RuntimeError("No table rows generated.")

    sample_count_text = str(args.sample_count if args.sample_count > 0 else "all")
    _write_excel(all_table_rows, args.output, sample_count_text)
    print(f"Saved Excel table to {args.output}")
    if args.raw_output:
        _write_raw_csv(all_raw_rows, args.raw_output)
        print(f"Saved per-sample raw results to {args.raw_output}")


if __name__ == "__main__":
    main()
