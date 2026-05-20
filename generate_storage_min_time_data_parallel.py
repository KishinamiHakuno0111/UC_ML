"""
Generate storage UC datasets with weakened minimum up/down time constraints.

This script combines the three-stage workflow in:
- generate_data_parallel.py: generate the 365 calendar-day base samples.
- ml_training_model.py: train a constraints_enable predictor from the 365 samples.
- generate_all_data_parallel.py: append randomized load samples using that predictor.

Default profiles:
- storage 0%  with min up/down time scaled to 0.3x.
- storage 50% with min up/down time scaled to 0.7x.

By default this script writes new, non-overwriting dataset names:
- training_data_case1888rte_storage_data_case1888rte_0%_minud0.3.npz
- all_training_data_case1888rte_storage_data_case1888rte_0%_minud0.3.npz
- training_data_case1888rte_storage_data_case1888rte_50%_minud0.7.npz
- all_training_data_case1888rte_storage_data_case1888rte_50%_minud0.7.npz
"""

import argparse
import copy
import gzip
import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import scipy.io
import torch

from ml_basic_functions import (
    get_data,
    get_load,
    magnify_peak_valley_difference,
    ml_load_model,
    train_model,
)
from uc_gurobi_constraint_filter import run_constraint_filter


USE_NPZ = True
DEFAULT_ENABLE_MODEL_DIR = "module_save"
DEFAULT_CONSTRAINTS_MODEL_MODULE = "DNN"
DEFAULT_CONSTRAINTS_HIDDEN_SIZES = [256, 256, 256]


PROFILE_DEFAULTS = {
    "0": {
        "source_storage_data_name": "storage_data_case1888rte_0%",
        "output_storage_data_name": "storage_data_case1888rte_0%_minud0.3",
        "min_time_scale": 0.3,
    },
    "50": {
        "source_storage_data_name": "storage_data_case1888rte_50%",
        "output_storage_data_name": "storage_data_case1888rte_50%_minud0.7",
        "min_time_scale": 0.7,
    },
}


# Worker globals for the 365-day stage.
_W365_STORAGE_DATA = None
_W365_LINE_LIMIT_RATE = None
_W365_NEW_ADD_NUM = None
_W365_MAGNIFY_RATE = None
_W365_USE_ENABLE_MODEL = None
_W365_ENABLE_MODEL = None
_W365_MIN_TIME_SCALE = None
_W365_MIN_TIME_ROUNDING = None


# Worker globals for the randomized extra-data stage.
_WEXTRA_ORIGINAL_DATA = None
_WEXTRA_STORAGE_DATA = None
_WEXTRA_DELTA_INPUT_LOAD = None
_WEXTRA_CENTER_INPUT_LOAD = None
_WEXTRA_AVERAGE_LOAD = None
_WEXTRA_ENABLE_MODEL = None
_WEXTRA_LINE_LIMIT_RATE = None
_WEXTRA_NEW_ADD_NUM = None
_WEXTRA_TIME_NUM = None
_WEXTRA_NG = None
_WEXTRA_NL = None
_WEXTRA_USE_ENABLE_MODEL = None


def _load_array(path):
    npz = path.replace(".mat", ".npz")
    if os.path.exists(npz):
        return dict(np.load(npz, allow_pickle=True))
    return scipy.io.loadmat(path)


def _load_storage_or_solar(npz_path):
    if os.path.exists(npz_path):
        return dict(np.load(npz_path, allow_pickle=True))
    mat_path = npz_path.replace(".npz", ".mat")
    if os.path.exists(mat_path):
        return scipy.io.loadmat(mat_path)
    raise FileNotFoundError(f"Neither {npz_path} nor {mat_path} found")


def get_history_data(dict_name, data_name):
    d = _load_array(os.path.join(dict_name, data_name))
    history_input_data = d["input_data"]
    history_output_data = d["output_data"]
    history_output_enable_data = d["enable_data"]
    history_l_data = d["l_data"]
    if history_input_data.ndim == 3:
        history_input_data = np.transpose(history_input_data, (0, 2, 1))
    if history_output_data.ndim == 3:
        history_output_data = np.transpose(history_output_data, (0, 2, 1))
    if history_l_data.ndim == 3:
        history_l_data = np.transpose(history_l_data, (0, 2, 1))
    return history_input_data, history_output_data, history_output_enable_data, history_l_data


def _save_npz(
    file_path,
    input_data,
    output_data,
    enable_data,
    l_data,
    average_load,
    active_lines=None,
    min_time_scale=None,
    min_time_rounding=None,
    source_storage_data_name=None,
):
    payload = dict(
        input_data=np.transpose(input_data, (0, 2, 1)),
        output_data=np.transpose(output_data, (0, 2, 1)),
        enable_data=enable_data,
        l_data=np.transpose(l_data, (0, 2, 1)),
        average_load=float(average_load),
    )
    if active_lines is not None:
        payload["active_lines"] = np.asarray(active_lines)
    if min_time_scale is not None:
        payload["min_time_scale"] = float(min_time_scale)
    if min_time_rounding is not None:
        payload["min_time_rounding"] = str(min_time_rounding)
    if source_storage_data_name is not None:
        payload["source_storage_data_name"] = str(source_storage_data_name)
    np.savez_compressed(file_path, **payload)


def _make_seed(base_seed, sample_idx):
    return int((base_seed + 1000003 * sample_idx) % (2**32 - 1))


def try_get_from_list(x, t):
    return x[t] if type(x) == list else x


def get_ptdf(data):
    bus_names, buses = zip(*data["Buses"].items())
    line_names, lines = (
        zip(*data["Transmission lines"].items())
        if "Transmission lines" in data
        else ([], [])
    )
    bus_idx_map = {bus_name: idx for idx, bus_name in enumerate(bus_names)}
    susceptance = np.diag([line["Susceptance (S)"] for line in lines])
    incidence = np.zeros([len(lines), len(buses)])
    for line_idx, line in enumerate(lines):
        source_bus_idx = bus_idx_map[line["Source bus"]]
        target_bus_idx = bus_idx_map[line["Target bus"]]
        incidence[line_idx, source_bus_idx] = 1
        incidence[line_idx, target_bus_idx] = -1
    incidence = incidence[:, 1:]
    laplacian = incidence.T @ susceptance @ incidence
    ptdf = np.zeros([len(lines), len(buses)])
    ptdf[:, 1:] = np.linalg.solve(laplacian, incidence.T @ susceptance).T
    return ptdf


def get_class(ptdf, active_lines, gate):
    active_line_num = len(active_lines)
    bus_num = np.size(ptdf, 1)
    is_affected = np.zeros([bus_num, active_line_num]).tolist()
    for i in range(bus_num):
        for j in range(active_line_num):
            line_index = active_lines[j]
            if abs(ptdf[line_index, i]) > gate:
                is_affected[i][j] = 1
    return is_affected


def _get_time_multiplier(data):
    time_step = data["Parameters"].get("Time step (min)", 60)
    assert 60 % time_step == 0
    return int(60 / time_step)


def _scaled_period_count(value_hours, scale, time_multiplier, rounding):
    periods = float(value_hours) * int(time_multiplier)
    scaled = periods * float(scale)
    if rounding == "ceil":
        new_periods = int(math.ceil(scaled))
    elif rounding == "floor":
        new_periods = int(math.floor(scaled))
    elif rounding == "round":
        new_periods = int(round(scaled))
    else:
        raise ValueError(f"Unsupported rounding mode: {rounding}")
    return max(1, new_periods)


def scale_min_up_down_times(data, scale, rounding="ceil"):
    """Scale generator min up/down times in-place and keep integer UC periods."""
    time_multiplier = _get_time_multiplier(data)
    for gen in data["Generators"].values():
        old_up = gen.get("Minimum uptime (h)", 1)
        old_down = gen.get("Minimum downtime (h)", 1)
        new_up_periods = _scaled_period_count(old_up, scale, time_multiplier, rounding)
        new_down_periods = _scaled_period_count(old_down, scale, time_multiplier, rounding)
        # unit_commitment_gurobi.py uses these values inside range(), so for
        # hourly case1888rte they must remain Python ints rather than 1.0/2.0.
        if time_multiplier == 1:
            gen["Minimum uptime (h)"] = int(new_up_periods)
            gen["Minimum downtime (h)"] = int(new_down_periods)
        else:
            gen["Minimum uptime (h)"] = new_up_periods / time_multiplier
            gen["Minimum downtime (h)"] = new_down_periods / time_multiplier
    return data


def assert_integer_min_up_down_periods(data):
    """Fail early if scaled min up/down times would become float periods."""
    time_multiplier = _get_time_multiplier(data)
    bad = []
    for gen_i, gen in enumerate(data["Generators"].values()):
        up_periods = gen.get("Minimum uptime (h)", 1) * time_multiplier
        down_periods = gen.get("Minimum downtime (h)", 1) * time_multiplier
        if not float(up_periods).is_integer() or not float(down_periods).is_integer():
            bad.append((gen_i, up_periods, down_periods))
    if bad:
        preview = bad[:5]
        raise ValueError(
            "Minimum up/down times must map to integer time periods before UC solve. "
            f"Bad examples: {preview}"
        )


def summarize_min_up_down_times(data, scale=None, rounding="ceil"):
    work = copy.deepcopy(data)
    if scale is not None:
        scale_min_up_down_times(work, scale, rounding)
    time_multiplier = _get_time_multiplier(work)
    units = list(work["Generators"].values())
    up = [
        int(round(unit.get("Minimum uptime (h)", 1) * time_multiplier))
        for unit in units
    ]
    down = [
        int(round(unit.get("Minimum downtime (h)", 1) * time_multiplier))
        for unit in units
    ]
    return {
        "time_multiplier": time_multiplier,
        "ng": len(units),
        "up_dist": sorted((int(k), int(v)) for k, v in _counter_items(up)),
        "down_dist": sorted((int(k), int(v)) for k, v in _counter_items(down)),
        "up_avg": float(np.mean(up)),
        "down_avg": float(np.mean(down)),
        "up_gt1": int(np.sum(np.asarray(up) > 1)),
        "down_gt1": int(np.sum(np.asarray(down) > 1)),
    }


def _counter_items(values):
    keys, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return zip(keys.tolist(), counts.tolist())


def _build_date_tasks(case_dict, case_type, case_name, data_year):
    month_day = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    tasks = []
    day_idx = 0
    base_path = case_dict + "/"
    case = case_type + "/" + case_name + "/"
    for month in range(1, 13):
        month_str = ("0" + str(month)) if month < 10 else str(month)
        day_num = month_day[month - 1]
        for day in range(1, day_num + 1):
            day_str = ("0" + str(day)) if day < 10 else str(day)
            file_name = data_year + "-" + month_str + "-" + day_str + ".json.gz"
            case_file_path = base_path + case + file_name
            tasks.append((day_idx, file_name, case_file_path))
            day_idx += 1
    return tasks


def _auto_enable_model_name(case_name, source_storage_data_name):
    return (
        f"constraints_DNN_training_data_{case_name}_{source_storage_data_name}"
        "_random_1000_50000_[256, 256, 256]_0.0002"
    )


def _parse_hidden_sizes(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _constraints_model_name(args, training_data_name):
    hidden_sizes = _parse_hidden_sizes(args.constraints_hidden_sizes)
    shuffle_str = "random" if args.constraints_data_is_random else "not_random"
    return (
        f"constraints_{args.constraints_module_name}_{training_data_name}_{shuffle_str}"
        f"_{args.constraints_training_num}_{args.constraints_num_epochs}"
        f"_{hidden_sizes}_{args.constraints_learning_rate}"
    )


def train_constraints_model_after_365(args, training_npz_path):
    """Train constraints_enable predictor from the generated 365-day data."""
    os.makedirs(args.enable_model_dir, exist_ok=True)
    training_data_name = os.path.splitext(os.path.basename(training_npz_path))[0]
    data_path = training_npz_path
    data = _load_array(data_path)
    available_num = int(data["input_data"].shape[0])

    training_num = int(args.constraints_training_num)
    test_num = int(args.constraints_test_num)
    if training_num + test_num > available_num:
        test_num = max(1, min(test_num, available_num // 5))
        training_num = available_num - test_num
        print(
            f"[constraints train] requested split exceeds available data; "
            f"use training_num={training_num}, test_num={test_num}"
        )

    module_name = args.constraints_module_name
    if module_name not in ("DNN", "KAN"):
        raise ValueError(
            "constraints_enable targets are sample-level vectors; use DNN or KAN, "
            f"not {module_name!r}."
        )

    hidden_sizes = _parse_hidden_sizes(args.constraints_hidden_sizes)
    shuffle_str = "random" if args.constraints_data_is_random else "not_random"
    model_name = (
        f"constraints_{module_name}_{training_data_name}_{shuffle_str}"
        f"_{training_num}_{args.constraints_num_epochs}_{hidden_sizes}"
        f"_{args.constraints_learning_rate}"
    )
    module_path = os.path.join(args.enable_model_dir, model_name)

    import random as _random

    _random.seed(args.constraints_random_seed)
    print(
        f"[constraints train] data={training_data_name}, model={model_name}, "
        f"training_num={training_num}, test_num={test_num}"
    )

    (
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        test_inputs,
        test_targets,
        _time_num,
        input_size,
        output_size,
        _average_load,
        train_size,
    ) = get_data(
        module_name,
        data_path,
        test_num=test_num,
        training_num=training_num,
        data_is_random=args.constraints_data_is_random,
        cutted_by_time=False,
        recognize_constr=True,
        random_seed=args.constraints_random_seed,
        val_ratio=args.constraints_val_ratio,
    )

    meta = {
        "model_type": module_name,
        "training_num": training_num,
        "test_num": test_num,
        "val_ratio": args.constraints_val_ratio,
        "data_is_random": args.constraints_data_is_random,
        "cutted_by_time": False,
        "data_name": training_data_name,
        "data_type": "constraints",
        "use_scheduler": args.constraints_use_scheduler,
        "warmup_ratio": args.constraints_warmup_ratio,
        "early_stopping_patience": args.constraints_early_stopping_patience,
        "learning_rate": args.constraints_learning_rate,
        "weight_decay": args.constraints_weight_decay,
        "source": "generate_storage_min_time_data_parallel.py",
    }
    with open(module_path + "_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    model, loss_record, val_loss_record, plt_result, test_result = train_model(
        module_name,
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
        input_size,
        output_size,
        module_path,
        hidden_sizes=hidden_sizes,
        learning_rate=args.constraints_learning_rate,
        num_epochs=args.constraints_num_epochs,
        dropout_prob=args.constraints_dropout_prob,
        num_pieces=args.constraints_num_pieces,
        is_maxout=args.constraints_is_maxout,
        stop_gap=args.constraints_stop_gap,
        weight_decay=args.constraints_weight_decay,
        recognize_constr=True,
        val_inputs=val_inputs,
        val_targets=val_targets,
        use_scheduler=args.constraints_use_scheduler,
        warmup_ratio=args.constraints_warmup_ratio,
        gradient_clip_val=args.constraints_gradient_clip_val,
        accumulate_steps=args.constraints_accumulate_steps,
        early_stopping_patience=args.constraints_early_stopping_patience,
        early_stopping_metric=args.constraints_early_stopping_metric,
        reliability_low_gate=args.constraints_reliability_low_gate,
        reliability_high_gate=args.constraints_reliability_high_gate,
        save_name=args.constraints_test_result_name,
    )

    if plt_result is not None:
        plt_result.savefig(module_path + "_curve.jpeg", format="jpeg", dpi=150)
        plt_result.close("all")

    del model, loss_record, val_loss_record, test_result
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[constraints train done] {model_name}")
    return model_name


def _init_365_worker(
    storage_data_path,
    line_limit_rate,
    new_add_num,
    magnify_rate,
    use_enable_model,
    enable_model_name,
    enable_model_dir,
    min_time_scale,
    min_time_rounding,
    gurobi_threads_per_worker,
    gurobi_output_flag,
    torch_threads_per_worker,
):
    global _W365_STORAGE_DATA
    global _W365_LINE_LIMIT_RATE
    global _W365_NEW_ADD_NUM
    global _W365_MAGNIFY_RATE
    global _W365_USE_ENABLE_MODEL
    global _W365_ENABLE_MODEL
    global _W365_MIN_TIME_SCALE
    global _W365_MIN_TIME_ROUNDING

    os.environ["UC_GUROBI_THREADS"] = str(int(gurobi_threads_per_worker))
    os.environ["UC_GUROBI_OUTPUTFLAG"] = str(int(gurobi_output_flag))

    _W365_STORAGE_DATA = _load_storage_or_solar(storage_data_path)
    _W365_LINE_LIMIT_RATE = float(line_limit_rate)
    _W365_NEW_ADD_NUM = int(new_add_num)
    _W365_MAGNIFY_RATE = float(magnify_rate)
    _W365_USE_ENABLE_MODEL = bool(use_enable_model)
    _W365_MIN_TIME_SCALE = float(min_time_scale)
    _W365_MIN_TIME_ROUNDING = str(min_time_rounding)

    if torch_threads_per_worker is not None:
        threads = max(1, int(torch_threads_per_worker))
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)

    if _W365_USE_ENABLE_MODEL:
        enable_model, *_ = ml_load_model(enable_model_name, enable_model_dir)
        enable_model = enable_model.to(torch.device("cpu"))
        enable_model.eval()
        _W365_ENABLE_MODEL = enable_model
    else:
        _W365_ENABLE_MODEL = None


def _solve_365_day(day_idx, file_name, case_file_path):
    with gzip.open(case_file_path) as f:
        data = json.loads(f.read())
    scale_min_up_down_times(data, _W365_MIN_TIME_SCALE, _W365_MIN_TIME_ROUNDING)
    assert_integer_min_up_down_periods(data)

    load_data = get_load(data)
    new_load_data = magnify_peak_valley_difference(load_data, _W365_MAGNIFY_RATE)

    unit_names, units = zip(*data["Generators"].items())
    ng = len(units)
    line_names, lines = (
        zip(*data["Transmission lines"].items())
        if "Transmission lines" in data
        else ([], [])
    )
    nl = len(lines)

    time_horizon = data["Parameters"].get("Time horizon (h)") or data["Parameters"].get("Time (h)")
    time_multiplier = _get_time_multiplier(data)
    nt = time_horizon * time_multiplier

    bus_names, buses = zip(*data["Buses"].items())
    nb = len(buses)

    if _W365_USE_ENABLE_MODEL:
        b_num, t_num = new_load_data.shape
        ml_input_i = np.transpose(new_load_data).astype(np.float32).reshape((b_num * t_num))
        ml_input_i = torch.from_numpy(ml_input_i).unsqueeze(0)
        with torch.no_grad():
            constraints_enable = _W365_ENABLE_MODEL(ml_input_i).squeeze(0).cpu().numpy()
        use_initial_enable = True
    else:
        constraints_enable = 0
        use_initial_enable = False

    (
        _model,
        _result,
        u_start,
        _y_start,
        _z_start,
        _p_start,
        _r_start,
        _c_start,
        l_start,
        enable_constraints,
        _model_time,
        _solve_t,
        _load_data_t,
        _round_num,
        no_answer,
    ) = run_constraint_filter(
        data,
        is_warm_start=False,
        use_new_input=True,
        use_storage=True,
        use_new_min_up_down_time=True,
        use_solar=False,
        add_more_constr=False,
        storage_data=_W365_STORAGE_DATA,
        new_input=new_load_data,
        line_limit_rate=_W365_LINE_LIMIT_RATE,
        new_add_num=_W365_NEW_ADD_NUM,
        add_constr_gate=0.7,
        initial_constraints=constraints_enable,
        use_initial_enable=use_initial_enable,
    )

    if no_answer:
        return {"status": "no_answer", "day_idx": day_idx, "file_name": file_name}

    input_array = np.asarray(new_load_data, dtype=np.float32)
    output_array = np.array(
        [[u_start[i, t] for t in range(nt)] for i in range(ng)],
        dtype=np.float32,
    )
    if isinstance(l_start, np.ndarray):
        l_array = l_start.astype(np.float32, copy=False)
    else:
        l_array = np.array(
            [[l_start[i, t] for t in range(nt)] for i in range(nl)],
            dtype=np.float32,
        )
    enable_array = np.asarray(enable_constraints, dtype=np.float32)

    normal_flow_limit = np.array(
        [_W365_LINE_LIMIT_RATE * line.get("Normal flow limit (MW)", 1e6) for line in lines],
        dtype=np.float32,
    )
    line_max_abs = np.max(np.abs(l_array), axis=1)
    active_line_hits = np.where((normal_flow_limit - line_max_abs) < 1e-3)[0].tolist()

    return {
        "status": "ok",
        "day_idx": day_idx,
        "file_name": file_name,
        "input_array": input_array,
        "output_array": output_array,
        "l_array": l_array,
        "enable_array": enable_array,
        "active_line_hits": active_line_hits,
        "nb": nb,
        "ng": ng,
        "nl": nl,
        "nt": nt,
    }


def generate_365_scaled_data_parallel(
    case_dict,
    case_type,
    case_name,
    ptdf_date,
    data_year,
    data_dist_name,
    source_storage_data_name,
    output_storage_data_name,
    min_time_scale,
    min_time_rounding,
    new_add_num,
    magnify_rate,
    line_limit_rate,
    use_enable_model,
    enable_model_name,
    enable_model_dir,
    num_workers,
    gurobi_threads_per_worker,
    gurobi_output_flag,
    torch_threads_per_worker,
    max_pending_factor,
):
    start_time = time.time()
    ptdf_path = case_dict + "/" + case_type + "/" + case_name + "/" + ptdf_date + ".json.gz"
    with gzip.open(ptdf_path) as f:
        ptdf_data = json.loads(f.read())
    ptdf = get_ptdf(ptdf_data)

    date_tasks = _build_date_tasks(case_dict, case_type, case_name, data_year)
    num_workers = _resolve_workers(num_workers)
    max_pending = max(num_workers, num_workers * int(max_pending_factor))
    storage_npz = os.path.join(data_dist_name, source_storage_data_name + ".npz")

    print(
        f"[365] output={output_storage_data_name}, min_time_scale={min_time_scale}, "
        f"rounding={min_time_rounding}, workers={num_workers}, days={len(date_tasks)}, "
        f"use_enable_model={use_enable_model}"
    )

    init_args = (
        storage_npz,
        line_limit_rate,
        new_add_num,
        magnify_rate,
        use_enable_model,
        enable_model_name,
        enable_model_dir,
        min_time_scale,
        min_time_rounding,
        gurobi_threads_per_worker,
        gurobi_output_flag,
        torch_threads_per_worker,
    )

    pending = {}
    completed = {}
    next_emit = 0
    submitted = 0
    no_answer_num = 0
    active_lines = set()
    input_list = []
    output_list = []
    enable_list = []
    l_list = []
    nb = nt = None

    with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_365_worker, initargs=init_args) as executor:
        while next_emit < len(date_tasks) or pending:
            while submitted < len(date_tasks) and len(pending) < max_pending:
                day_idx, file_name, case_file_path = date_tasks[submitted]
                future = executor.submit(_solve_365_day, day_idx, file_name, case_file_path)
                pending[future] = day_idx
                submitted += 1

            if not pending:
                break

            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                idx = pending.pop(future)
                completed[idx] = future.result()

            while next_emit in completed:
                result = completed.pop(next_emit)
                file_name = result["file_name"]
                if result["status"] == "ok":
                    input_list.append(result["input_array"])
                    output_list.append(result["output_array"])
                    enable_list.append(result["enable_array"])
                    l_list.append(result["l_array"])
                    for line_i in result["active_line_hits"]:
                        active_lines.add(int(line_i))
                    nb = result["nb"]
                    nt = result["nt"]
                    print(f"[365 ok] day={next_emit + 1}/{len(date_tasks)}, file={file_name}")
                else:
                    no_answer_num += 1
                    print(f"[365 skip] day={next_emit + 1}/{len(date_tasks)}, file={file_name}")
                next_emit += 1

    data_num = len(input_list)
    if data_num == 0:
        raise RuntimeError("No feasible 365-day data generated.")

    input_array = np.asarray(input_list, dtype=np.float32)
    output_array = np.asarray(output_list, dtype=np.float32)
    enable_array = np.asarray(enable_list, dtype=np.float32)
    l_array = np.asarray(l_list, dtype=np.float32)

    average_load = float(np.sum(input_array) / (data_num * nt * nb))
    input_array = input_array / average_load

    active_lines_sorted = sorted(active_lines)
    gate = 0.005
    _ = get_class(ptdf, active_lines_sorted, gate) if active_lines_sorted else []

    training_npz = os.path.join(
        data_dist_name,
        f"training_data_{case_name}_{output_storage_data_name}.npz",
    )
    all_npz = os.path.join(
        data_dist_name,
        f"all_training_data_{case_name}_{output_storage_data_name}.npz",
    )

    common_kwargs = dict(
        active_lines=np.array(active_lines_sorted),
        min_time_scale=min_time_scale,
        min_time_rounding=min_time_rounding,
        source_storage_data_name=source_storage_data_name,
    )
    _save_npz(
        training_npz,
        input_array,
        output_array,
        enable_array,
        l_array,
        average_load,
        **common_kwargs,
    )
    _save_npz(
        all_npz,
        input_array,
        output_array,
        enable_array,
        l_array,
        average_load,
        **common_kwargs,
    )

    elapsed = time.time() - start_time
    print(
        f"[365 done] output={output_storage_data_name}, data={data_num}, "
        f"no_answer={no_answer_num}, seconds={elapsed:.2f}"
    )
    return training_npz, all_npz


def _init_extra_worker(
    original_data,
    storage_data,
    delta_input_load,
    center_input_load,
    average_load,
    line_limit_rate,
    new_add_num,
    time_num,
    ng,
    nl,
    enable_model_name,
    enable_model_dir,
    use_enable_model,
    gurobi_threads_per_worker,
    gurobi_output_flag,
    torch_threads_per_worker,
):
    global _WEXTRA_ORIGINAL_DATA
    global _WEXTRA_STORAGE_DATA
    global _WEXTRA_DELTA_INPUT_LOAD
    global _WEXTRA_CENTER_INPUT_LOAD
    global _WEXTRA_AVERAGE_LOAD
    global _WEXTRA_ENABLE_MODEL
    global _WEXTRA_LINE_LIMIT_RATE
    global _WEXTRA_NEW_ADD_NUM
    global _WEXTRA_TIME_NUM
    global _WEXTRA_NG
    global _WEXTRA_NL
    global _WEXTRA_USE_ENABLE_MODEL

    os.environ["UC_GUROBI_THREADS"] = str(int(gurobi_threads_per_worker))
    os.environ["UC_GUROBI_OUTPUTFLAG"] = str(int(gurobi_output_flag))

    _WEXTRA_ORIGINAL_DATA = original_data
    _WEXTRA_STORAGE_DATA = storage_data
    _WEXTRA_DELTA_INPUT_LOAD = delta_input_load
    _WEXTRA_CENTER_INPUT_LOAD = center_input_load
    _WEXTRA_AVERAGE_LOAD = float(average_load)
    _WEXTRA_LINE_LIMIT_RATE = float(line_limit_rate)
    _WEXTRA_NEW_ADD_NUM = int(new_add_num)
    _WEXTRA_TIME_NUM = int(time_num)
    _WEXTRA_NG = int(ng)
    _WEXTRA_NL = int(nl)
    _WEXTRA_USE_ENABLE_MODEL = bool(use_enable_model)

    if torch_threads_per_worker is not None:
        threads = max(1, int(torch_threads_per_worker))
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)

    if _WEXTRA_USE_ENABLE_MODEL:
        enable_model, *_ = ml_load_model(enable_model_name, enable_model_dir)
        enable_model = enable_model.to(torch.device("cpu"))
        enable_model.eval()
        _WEXTRA_ENABLE_MODEL = enable_model
    else:
        _WEXTRA_ENABLE_MODEL = None


def _generate_extra_one(sample_idx, seed):
    rng = np.random.default_rng(seed)
    rand = 2.0 * rng.random(_WEXTRA_DELTA_INPUT_LOAD.shape) - 1.0
    input_i = rand * _WEXTRA_DELTA_INPUT_LOAD + _WEXTRA_CENTER_INPUT_LOAD

    if _WEXTRA_USE_ENABLE_MODEL:
        b_num, t_num = input_i.shape
        ml_input_i = np.transpose(input_i).astype(np.float32).reshape((b_num * t_num))
        ml_input_i = torch.from_numpy(ml_input_i).unsqueeze(0)
        with torch.no_grad():
            constraints_enable = _WEXTRA_ENABLE_MODEL(ml_input_i).squeeze(0).cpu().numpy()
        use_initial_enable = True
    else:
        constraints_enable = 0
        use_initial_enable = False

    (
        _model,
        _result,
        u_start,
        _y_start,
        _z_start,
        _p_start,
        _r_start,
        _c_start,
        l_start,
        enable_constraints,
        _model_time,
        _solve_t,
        _load_data_t,
        _round_num,
        no_answer,
    ) = run_constraint_filter(
        _WEXTRA_ORIGINAL_DATA,
        is_warm_start=False,
        use_new_input=True,
        use_storage=True,
        use_new_min_up_down_time=True,
        use_solar=False,
        add_more_constr=False,
        storage_data=_WEXTRA_STORAGE_DATA,
        new_input=input_i,
        line_limit_rate=_WEXTRA_LINE_LIMIT_RATE,
        new_add_num=_WEXTRA_NEW_ADD_NUM,
        add_constr_gate=0.7,
        initial_constraints=constraints_enable,
        use_initial_enable=use_initial_enable,
    )

    if no_answer:
        return {"status": "no_answer", "sample_idx": sample_idx}

    output_array = np.array(
        [[u_start[i, t] for t in range(_WEXTRA_TIME_NUM)] for i in range(_WEXTRA_NG)],
        dtype=np.float32,
    )
    if isinstance(l_start, np.ndarray):
        l_array = l_start.astype(np.float32, copy=False)
    else:
        l_array = np.array(
            [[l_start[i, t] for t in range(_WEXTRA_TIME_NUM)] for i in range(_WEXTRA_NL)],
            dtype=np.float32,
        )

    return {
        "status": "ok",
        "sample_idx": sample_idx,
        "input_array": (input_i / _WEXTRA_AVERAGE_LOAD).astype(np.float32),
        "output_array": output_array,
        "enable_array": np.asarray(enable_constraints, dtype=np.float32),
        "l_array": l_array,
    }


def generate_extra_scaled_data_parallel(
    new_data_num,
    save_gap,
    line_limit_rate,
    dict_name,
    training_data,
    source_storage_data_name,
    case_dist,
    case_type,
    case_name,
    ptdf_date,
    new_add_num,
    min_time_scale,
    min_time_rounding,
    use_enable_model,
    enable_model_name,
    enable_model_dir,
    num_workers,
    gurobi_threads_per_worker,
    gurobi_output_flag,
    torch_threads_per_worker,
    max_pending_factor,
    seed,
):
    d = _load_array(os.path.join(dict_name, training_data))
    input_data_raw = d["input_data"]
    if input_data_raw.ndim == 3:
        input_data_raw = np.transpose(input_data_raw, (0, 2, 1))
    average_load = float(np.asarray(d["average_load"]).item())
    input_load = input_data_raw * average_load
    _sample_num, _bus_num, time_num = input_load.shape
    max_input_load = np.max(input_load, axis=0)
    min_input_load = np.min(input_load, axis=0)
    delta_input_load = max_input_load - min_input_load
    center_input_load = (min_input_load + max_input_load) / 2.0

    storage_data = _load_array(os.path.join(dict_name, source_storage_data_name + ".mat"))

    ptdf_path = case_dist + "/" + case_type + "/" + case_name + "/" + ptdf_date + ".json.gz"
    with gzip.open(ptdf_path) as f:
        original_data = json.loads(f.read())
    scale_min_up_down_times(original_data, min_time_scale, min_time_rounding)
    assert_integer_min_up_down_periods(original_data)

    unit_names, units = zip(*original_data["Generators"].items())
    ng = len(units)
    line_names, lines = (
        zip(*original_data["Transmission lines"].items())
        if "Transmission lines" in original_data
        else ([], [])
    )
    nl = len(lines)

    history_data_name = "all_" + training_data
    (
        history_input_data,
        history_output_data,
        history_output_enable_data,
        history_l_data,
    ) = get_history_data(dict_name, training_data)

    num_workers = _resolve_workers(num_workers)
    max_pending = max(num_workers, int(max_pending_factor) * num_workers)

    print(
        f"[extra] training_data={training_data}, new_data_num={new_data_num}, "
        f"min_time_scale={min_time_scale}, workers={num_workers}, use_enable_model={use_enable_model}"
    )

    init_args = (
        original_data,
        storage_data,
        delta_input_load,
        center_input_load,
        average_load,
        line_limit_rate,
        new_add_num,
        time_num,
        ng,
        nl,
        enable_model_name,
        enable_model_dir,
        use_enable_model,
        gurobi_threads_per_worker,
        gurobi_output_flag,
        torch_threads_per_worker,
    )

    pending = {}
    completed = {}
    no_answer_num = 0
    next_emit = 0
    submitted = 0
    accepted = 0

    input_save = []
    output_save = []
    enable_save = []
    l_save = []

    def _flush_cache():
        nonlocal history_input_data
        nonlocal history_output_data
        nonlocal history_output_enable_data
        nonlocal history_l_data
        if len(input_save) == 0:
            return
        batch_input = np.asarray(input_save, dtype=np.float32)
        batch_output = np.asarray(output_save, dtype=np.float32)
        batch_enable = np.asarray(enable_save, dtype=np.float32)
        batch_l = np.asarray(l_save, dtype=np.float32)
        history_input_data = np.concatenate((history_input_data, batch_input), axis=0)
        history_output_data = np.concatenate((history_output_data, batch_output), axis=0)
        history_output_enable_data = np.concatenate((history_output_enable_data, batch_enable), axis=0)
        history_l_data = np.concatenate((history_l_data, batch_l), axis=0)
        input_save.clear()
        output_save.clear()
        enable_save.clear()
        l_save.clear()

    with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_extra_worker, initargs=init_args) as executor:
        while next_emit < new_data_num or pending:
            while submitted < new_data_num and len(pending) < max_pending:
                task_seed = _make_seed(seed, submitted)
                future = executor.submit(_generate_extra_one, submitted, task_seed)
                pending[future] = submitted
                submitted += 1

            if not pending:
                break

            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                idx = pending.pop(future)
                completed[idx] = future.result()

            while next_emit in completed:
                result = completed.pop(next_emit)
                if result["status"] == "ok":
                    input_save.append(result["input_array"])
                    output_save.append(result["output_array"])
                    enable_save.append(result["enable_array"])
                    l_save.append(result["l_array"])
                    accepted += 1
                    print(f"[extra ok] sample={next_emit}, accepted={accepted}")

                    if (next_emit % save_gap) == 0 and next_emit != 0:
                        _flush_cache()
                        file_path = os.path.join(dict_name, history_data_name)
                        _save_npz(
                            file_path,
                            history_input_data,
                            history_output_data,
                            history_output_enable_data,
                            history_l_data,
                            average_load,
                            active_lines=d.get("active_lines"),
                            min_time_scale=min_time_scale,
                            min_time_rounding=min_time_rounding,
                            source_storage_data_name=source_storage_data_name,
                        )
                        print(f"[extra autosave] sample={next_emit}, total={len(history_input_data)}")
                else:
                    no_answer_num += 1
                    print(f"[extra skip] sample={next_emit}, no feasible answer")
                next_emit += 1

    _flush_cache()
    file_path = os.path.join(dict_name, history_data_name)
    _save_npz(
        file_path,
        history_input_data,
        history_output_data,
        history_output_enable_data,
        history_l_data,
        average_load,
        active_lines=d.get("active_lines"),
        min_time_scale=min_time_scale,
        min_time_rounding=min_time_rounding,
        source_storage_data_name=source_storage_data_name,
    )
    print(
        f"[extra done] output={history_data_name}, accepted={accepted}, "
        f"no_answer={no_answer_num}, total={len(history_input_data)}"
    )
    return file_path


def _resolve_workers(num_workers):
    if num_workers is None:
        cpu_num = os.cpu_count() or 1
        return max(1, cpu_num - 1)
    return max(1, int(num_workers))


def _profile_ids(value):
    value = str(value).lower()
    if value == "both":
        return ["0", "50"]
    if value in ("0", "0%", "zero"):
        return ["0"]
    if value in ("50", "50%", "half"):
        return ["50"]
    raise ValueError(f"Unsupported profile selection: {value}")


def _profile_from_args(args, profile_id):
    defaults = PROFILE_DEFAULTS[profile_id]
    if profile_id == "0":
        source = args.source_storage_name_0 or defaults["source_storage_data_name"]
        output = args.output_storage_name_0 or defaults["output_storage_data_name"]
        scale = args.min_time_scale_0
        enable_model_name = args.enable_model_name_0
    else:
        source = args.source_storage_name_50 or defaults["source_storage_data_name"]
        output = args.output_storage_name_50 or defaults["output_storage_data_name"]
        scale = args.min_time_scale_50
        enable_model_name = args.enable_model_name_50

    if enable_model_name is None:
        training_data_name = f"training_data_{args.case_name}_{output}"
        enable_model_name = _constraints_model_name(args, training_data_name)

    return {
        "source_storage_data_name": source,
        "output_storage_data_name": output,
        "min_time_scale": float(scale),
        "enable_model_name": enable_model_name,
    }


def _paths_for_profile(args, profile):
    training_path = os.path.join(
        args.data_dist_name,
        f"training_data_{args.case_name}_{profile['output_storage_data_name']}.npz",
    )
    all_path = os.path.join(
        args.data_dist_name,
        f"all_training_data_{args.case_name}_{profile['output_storage_data_name']}.npz",
    )
    return training_path, all_path


def _check_new_outputs(args, profile):
    training_path, all_path = _paths_for_profile(args, profile)
    if not args.overwrite:
        existing = [p for p in (training_path, all_path) if os.path.exists(p)]
        if existing:
            raise FileExistsError(
                "Output already exists. Use --overwrite or choose another output name: "
                + ", ".join(existing)
            )
    return training_path, all_path


def _check_existing_365_output(args, profile):
    training_path, all_path = _paths_for_profile(args, profile)
    if not os.path.exists(training_path):
        raise FileNotFoundError(
            f"365-day training data not found: {training_path}. "
            "Run with --stage 365 or --stage all first."
        )
    return training_path, all_path


def analyze_min_times(args):
    path = os.path.join(
        args.case_dict,
        args.case_type,
        args.case_name,
        args.ptdf_date + ".json.gz",
    )
    with gzip.open(path) as f:
        data = json.loads(f.read())
    base = summarize_min_up_down_times(data)
    print("[min-time analysis] original:", base)
    for profile_id in _profile_ids(args.profiles):
        profile = _profile_from_args(args, profile_id)
        scaled = summarize_min_up_down_times(
            data,
            scale=profile["min_time_scale"],
            rounding=args.min_time_rounding,
        )
        print(
            f"[min-time analysis] profile={profile_id}, "
            f"scale={profile['min_time_scale']}, rounding={args.min_time_rounding}: {scaled}"
        )


def run_profile(args, profile):
    if args.stage in ("all", "365"):
        training_path, all_path = _check_new_outputs(args, profile)
    else:
        training_path, all_path = _check_existing_365_output(args, profile)

    enable_model_name_for_extra = profile["enable_model_name"]

    if args.stage in ("all", "365"):
        generate_365_scaled_data_parallel(
            case_dict=args.case_dict,
            case_type=args.case_type,
            case_name=args.case_name,
            ptdf_date=args.ptdf_date,
            data_year=args.data_year,
            data_dist_name=args.data_dist_name,
            source_storage_data_name=profile["source_storage_data_name"],
            output_storage_data_name=profile["output_storage_data_name"],
            min_time_scale=profile["min_time_scale"],
            min_time_rounding=args.min_time_rounding,
            new_add_num=args.new_add_num,
            magnify_rate=args.magnify_rate,
            line_limit_rate=args.line_limit_rate,
            use_enable_model=False,
            enable_model_name=enable_model_name_for_extra,
            enable_model_dir=args.enable_model_dir,
            num_workers=args.num_workers,
            gurobi_threads_per_worker=args.gurobi_threads_per_worker,
            gurobi_output_flag=args.gurobi_output_flag,
            torch_threads_per_worker=args.torch_threads_per_worker,
            max_pending_factor=args.max_pending_factor,
        )

    if args.stage in ("all", "train"):
        if args.train_enable_model_after_365 or args.stage == "train":
            enable_model_name_for_extra = train_constraints_model_after_365(
                args,
                training_path,
            )

    if args.stage in ("all", "extra"):
        if args.use_enable_model_extra:
            model_path = os.path.join(args.enable_model_dir, enable_model_name_for_extra + ".pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"constraints_enable model not found for extra stage: {model_path}. "
                    "Run --stage train first, keep --train-enable-model-after-365 enabled, "
                    "or pass --enable-model-name-0/--enable-model-name-50."
                )
        training_data_name = os.path.basename(training_path)
        generate_extra_scaled_data_parallel(
            new_data_num=args.extra_data_num,
            save_gap=args.save_gap,
            line_limit_rate=args.line_limit_rate,
            dict_name=args.data_dist_name,
            training_data=training_data_name,
            source_storage_data_name=profile["source_storage_data_name"],
            case_dist=args.case_dict,
            case_type=args.case_type,
            case_name=args.case_name,
            ptdf_date=args.ptdf_date,
            new_add_num=args.new_add_num,
            min_time_scale=profile["min_time_scale"],
            min_time_rounding=args.min_time_rounding,
            use_enable_model=args.use_enable_model_extra,
            enable_model_name=enable_model_name_for_extra,
            enable_model_dir=args.enable_model_dir,
            num_workers=args.num_workers,
            gurobi_threads_per_worker=args.gurobi_threads_per_worker,
            gurobi_output_flag=args.gurobi_output_flag,
            torch_threads_per_worker=args.torch_threads_per_worker,
            max_pending_factor=args.max_pending_factor,
            seed=args.seed,
        )

    print(f"[profile done] {profile['output_storage_data_name']}")
    print(f"  training: {training_path}")
    print(f"  all:      {all_path}")


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate case1888rte storage datasets with scaled minimum up/down times."
    )
    parser.add_argument("--profiles", type=str, default="both", help="both, 0, or 50")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "365", "train", "extra"])
    parser.add_argument("--analyze-only", action="store_true")

    parser.add_argument("--case-dict", type=str, default="case_data")
    parser.add_argument("--case-type", type=str, default="matpower")
    parser.add_argument("--case-name", type=str, default="case1888rte")
    parser.add_argument("--ptdf-date", type=str, default="2017-01-01")
    parser.add_argument("--data-year", type=str, default="2017")
    parser.add_argument("--data-dist-name", type=str, default="training_data")

    parser.add_argument("--source-storage-name-0", type=str, default=None)
    parser.add_argument("--source-storage-name-50", type=str, default=None)
    parser.add_argument("--output-storage-name-0", type=str, default=None)
    parser.add_argument("--output-storage-name-50", type=str, default=None)
    parser.add_argument("--min-time-scale-0", type=float, default=0.3)
    parser.add_argument("--min-time-scale-50", type=float, default=0.7)
    parser.add_argument("--min-time-rounding", type=str, default="ceil", choices=["ceil", "floor", "round"])

    parser.add_argument("--extra-data-num", type=int, default=1000)
    parser.add_argument("--save-gap", type=int, default=50)
    parser.add_argument("--new-add-num", type=int, default=5)
    parser.add_argument("--magnify-rate", type=float, default=1.0)
    parser.add_argument("--line-limit-rate", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--train-enable-model-after-365",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After the 365-day stage, train a constraints_enable model from those 365 samples.",
    )
    parser.add_argument(
        "--use-enable-model-extra",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use neural network to predict initial constraints_enable in the extra-data stage.",
    )
    parser.add_argument("--enable-model-name-0", type=str, default=None)
    parser.add_argument("--enable-model-name-50", type=str, default=None)
    parser.add_argument("--enable-model-dir", type=str, default=DEFAULT_ENABLE_MODEL_DIR)

    parser.add_argument("--constraints-module-name", type=str, default=DEFAULT_CONSTRAINTS_MODEL_MODULE, choices=["DNN", "KAN"])
    parser.add_argument("--constraints-hidden-sizes", type=str, default="256,256,256")
    parser.add_argument("--constraints-training-num", type=int, default=300)
    parser.add_argument("--constraints-test-num", type=int, default=65)
    parser.add_argument("--constraints-data-is-random", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--constraints-random-seed", type=int, default=42)
    parser.add_argument("--constraints-val-ratio", type=float, default=0.1)
    parser.add_argument("--constraints-num-epochs", type=int, default=50000)
    parser.add_argument("--constraints-stop-gap", type=int, default=50)
    parser.add_argument("--constraints-learning-rate", type=float, default=0.0002)
    parser.add_argument("--constraints-weight-decay", type=float, default=0.001)
    parser.add_argument("--constraints-use-scheduler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--constraints-warmup-ratio", type=float, default=0.1)
    parser.add_argument("--constraints-gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--constraints-accumulate-steps", type=int, default=1)
    parser.add_argument("--constraints-early-stopping-patience", type=int, default=50)
    parser.add_argument("--constraints-early-stopping-metric", type=str, default="val_loss", choices=["val_loss", "train_loss"])
    parser.add_argument("--constraints-dropout-prob", type=float, default=0.3)
    parser.add_argument("--constraints-num-pieces", type=int, default=2)
    parser.add_argument("--constraints-is-maxout", action="store_true")
    parser.add_argument("--constraints-reliability-low-gate", type=float, default=0.1)
    parser.add_argument("--constraints-reliability-high-gate", type=float, default=0.9)
    parser.add_argument("--constraints-test-result-name", type=str, default="test_result_constraints_minud.xlsx")

    parser.add_argument("--num-workers", type=int, default=max((os.cpu_count() or 1) // 2, 1))
    parser.add_argument("--gurobi-threads-per-worker", type=int, default=1)
    parser.add_argument("--gurobi-output-flag", type=int, default=0)
    parser.add_argument("--torch-threads-per-worker", type=int, default=1)
    parser.add_argument("--max-pending-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260428)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    analyze_min_times(args)
    if not args.analyze_only:
        for profile_id in _profile_ids(args.profiles):
            run_profile(args, _profile_from_args(args, profile_id))
