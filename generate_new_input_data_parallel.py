"""
Parallel data generation for UC input perturbation.

This script keeps the same core logic as generate_new_input_data.py, but
dispatches per-sample solve tasks to multiple worker processes so multi-core
CPUs can be utilized.
"""

import argparse
import copy
import gzip
import json
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import torch

from ml_basic_functions import ml_load_model
from uc_gurobi_constraint_filter import run_constraint_filter


PERTURB_ZERO_RATIO = 2.0
PERTURB_NZ_RATIO = 1.5
LOAD_COMPENSATION_RATIO = 1.0
MIN_SWITCH_COMPLEXITY_RATIO = 0.8
SAVE_INTERVAL = 50

DEFAULT_ENABLE_MODEL_NAME = (
    "constraints_DNN_all_training_data_case1888rte_pro_random_1000_50000_[256, 256, 256]_0.0002"
)
DEFAULT_ENABLE_MODEL_DIR = "module_save"


# Worker globals (initialized once per process)
_W_TEMPLATE_DATA = None
_W_KEY_LIST = None
_W_STORAGE_DATA = None
_W_ENABLE_MODEL = None
_W_AVERAGE_LOAD = None
_W_PERTURB_ZERO = None
_W_PERTURB_NZ = None
_W_LOAD_COMPENSATION_RATIO = None
_W_MIN_SWITCH_COMPLEXITY_RATIO = None
_W_LINE_LIMIT_RATE = None
_W_NEW_ADD_NUM = None
_W_TIME_NUM = None
_W_NG = None
_W_NL = None


def _load_array(path):
    npz = path.replace(".mat", ".npz")
    if os.path.exists(npz):
        return dict(np.load(npz, allow_pickle=True))
    import scipy.io

    return scipy.io.loadmat(path)


def get_history_data(dict_name, data_name):
    d = _load_array(os.path.join(dict_name, data_name))
    for key in ("input_data", "output_data", "l_data"):
        if key in d and d[key].ndim == 3:
            d[key] = np.transpose(d[key], (0, 2, 1))
    return d["input_data"], d["output_data"], d["enable_data"], d["l_data"]


def _gen_perturbation(input_load_i, average_load, perturb_zero, perturb_nz,
                      load_compensation_ratio, rng):
    bus_num, time_num = input_load_i.shape
    bus_is_zero = np.sum(np.abs(input_load_i), axis=1) < 1e-3
    zero_mask = bus_is_zero.reshape(-1, 1)
    nonzero_mask = ~zero_mask

    renewable = perturb_zero * rng.random((bus_num, time_num)) * zero_mask
    load_cap = perturb_nz * np.ones_like(input_load_i, dtype=float) * nonzero_mask
    compensation_ratio = max(0.0, float(load_compensation_ratio))
    renewable_sum = np.sum(renewable, axis=0, keepdims=True)

    if compensation_ratio > 0:
        capacity_sum = np.sum(load_cap, axis=0, keepdims=True)
        valid_t = renewable_sum[0] > 1e-9
        scale = np.ones_like(renewable_sum)
        scale[:, valid_t] = np.minimum(
            1.0,
            capacity_sum[:, valid_t] / (renewable_sum[:, valid_t] * compensation_ratio),
        )
        renewable = renewable * scale
        renewable_sum = np.sum(renewable, axis=0, keepdims=True)

    delta_zero = -renewable
    extra_load_shape = (
        np.maximum(input_load_i, 0.0) + 0.05 * average_load
    ) * rng.random((bus_num, time_num)) * nonzero_mask
    target_extra_load = renewable_sum * compensation_ratio
    delta_nz = _distribute_extra_load(extra_load_shape, target_extra_load, load_cap)

    return delta_zero + delta_nz


def _distribute_extra_load(shape, target, cap):
    delta = np.zeros_like(shape, dtype=float)
    time_num = shape.shape[1]
    for t in range(time_num):
        remaining_target = float(target[0, t])
        if remaining_target <= 1e-9:
            continue

        residual_cap = cap[:, t].astype(float, copy=True)
        active = residual_cap > 1e-9
        while remaining_target > 1e-9 and np.any(active):
            weights = np.where(active, shape[:, t], 0.0)
            weight_sum = float(np.sum(weights))
            if weight_sum <= 1e-12:
                weights = active.astype(float)
                weight_sum = float(np.sum(weights))

            proposed = remaining_target * weights / weight_sum
            added = np.minimum(proposed, residual_cap)
            added_sum = float(np.sum(added))
            if added_sum <= 1e-9:
                break

            delta[:, t] += added
            residual_cap -= added
            remaining_target -= added_sum
            active = residual_cap > 1e-9

    return delta


def _count_switch_complexity(output):
    if output.shape[1] <= 1:
        return 0
    return int(np.sum(np.abs(np.diff(output, axis=1)) > 0.5))


def _passes_switch_complexity_filter(new_output, base_output, min_ratio):
    base_complexity = _count_switch_complexity(base_output)
    new_complexity = _count_switch_complexity(new_output)
    required_complexity = int(np.floor(base_complexity * float(min_ratio)))
    return new_complexity >= required_complexity, new_complexity, base_complexity, required_complexity


def _save(npz_path, input_data, output_data, enable_data, l_data, average_load):
    np.savez_compressed(
        npz_path,
        input_data=np.transpose(input_data, (0, 2, 1)),
        output_data=np.transpose(output_data, (0, 2, 1)),
        enable_data=enable_data,
        l_data=np.transpose(l_data, (0, 2, 1)),
        average_load=float(average_load),
    )


def _make_seed(base_seed, attempt_id, source_i):
    return int((base_seed + 1000003 * attempt_id + 9176 * source_i) % (2**32 - 1))


def _worker_init(
    template_data,
    key_list,
    storage_data,
    average_load,
    perturb_zero,
    perturb_nz,
    load_compensation_ratio,
    min_switch_complexity_ratio,
    line_limit_rate,
    new_add_num,
    time_num,
    ng,
    nl,
    enable_model_name,
    enable_model_dir,
    gurobi_threads_per_worker,
    gurobi_output_flag,
    torch_threads_per_worker,
):
    global _W_TEMPLATE_DATA
    global _W_KEY_LIST
    global _W_STORAGE_DATA
    global _W_ENABLE_MODEL
    global _W_AVERAGE_LOAD
    global _W_PERTURB_ZERO
    global _W_PERTURB_NZ
    global _W_LOAD_COMPENSATION_RATIO
    global _W_MIN_SWITCH_COMPLEXITY_RATIO
    global _W_LINE_LIMIT_RATE
    global _W_NEW_ADD_NUM
    global _W_TIME_NUM
    global _W_NG
    global _W_NL

    os.environ["UC_GUROBI_THREADS"] = str(int(gurobi_threads_per_worker))
    os.environ["UC_GUROBI_OUTPUTFLAG"] = str(int(gurobi_output_flag))

    _W_TEMPLATE_DATA = template_data
    _W_KEY_LIST = key_list
    _W_STORAGE_DATA = storage_data
    _W_AVERAGE_LOAD = float(average_load)
    _W_PERTURB_ZERO = float(perturb_zero)
    _W_PERTURB_NZ = float(perturb_nz)
    _W_LOAD_COMPENSATION_RATIO = float(load_compensation_ratio)
    _W_MIN_SWITCH_COMPLEXITY_RATIO = float(min_switch_complexity_ratio)
    _W_LINE_LIMIT_RATE = float(line_limit_rate)
    _W_NEW_ADD_NUM = int(new_add_num)
    _W_TIME_NUM = int(time_num)
    _W_NG = int(ng)
    _W_NL = int(nl)

    if torch_threads_per_worker is not None:
        threads = max(1, int(torch_threads_per_worker))
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)

    enable_model, *_ = ml_load_model(enable_model_name, enable_model_dir)
    enable_model = enable_model.to(torch.device("cpu"))
    enable_model.eval()
    _W_ENABLE_MODEL = enable_model


def _worker_solve_one(attempt_id, source_i, input_load_i, base_output_i, seed):
    rng = np.random.default_rng(seed)
    delta = _gen_perturbation(
        input_load_i,
        _W_AVERAGE_LOAD,
        _W_PERTURB_ZERO,
        _W_PERTURB_NZ,
        _W_LOAD_COMPENSATION_RATIO,
        rng,
    )
    input_i_mw = input_load_i + delta

    data_copy = copy.deepcopy(_W_TEMPLATE_DATA)
    for k_i, key in enumerate(_W_KEY_LIST):
        data_copy["Buses"][key]["Load (MW)"] = input_i_mw[k_i, :].tolist()

    ml_input = (input_i_mw.T / _W_AVERAGE_LOAD).astype(np.float32).reshape(-1)
    ml_input_t = torch.from_numpy(ml_input).unsqueeze(0)
    with torch.no_grad():
        constraints_enable = _W_ENABLE_MODEL(ml_input_t).squeeze(0).cpu().numpy()

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
        data_copy,
        is_warm_start=False,
        use_new_input=True,
        use_storage=True,
        use_new_min_up_down_time=True,
        use_solar=False,
        add_more_constr=False,
        storage_data=_W_STORAGE_DATA,
        new_input=input_i_mw,
        line_limit_rate=_W_LINE_LIMIT_RATE,
        new_add_num=_W_NEW_ADD_NUM,
        add_constr_gate=0.7,
        initial_constraints=constraints_enable,
        use_initial_enable=True,
    )

    if no_answer:
        return {
            "status": "no_answer",
            "attempt_id": attempt_id,
            "source_i": source_i,
        }

    new_output = np.array(
        [[u_start[i, t] for t in range(_W_TIME_NUM)] for i in range(_W_NG)],
        dtype=np.float32,
    )
    pass_complexity, new_complexity, base_complexity, required_complexity = (
        _passes_switch_complexity_filter(
            new_output,
            base_output_i,
            _W_MIN_SWITCH_COMPLEXITY_RATIO,
        )
    )
    if not pass_complexity:
        return {
            "status": "skip",
            "attempt_id": attempt_id,
            "source_i": source_i,
            "new_complexity": new_complexity,
            "base_complexity": base_complexity,
            "required_complexity": required_complexity,
        }

    if isinstance(l_start, np.ndarray):
        l_array = l_start.astype(np.float32, copy=False)
    else:
        l_array = np.array(
            [[l_start[i, t] for t in range(_W_TIME_NUM)] for i in range(_W_NL)],
            dtype=np.float32,
        )

    return {
        "status": "ok",
        "attempt_id": attempt_id,
        "source_i": source_i,
        "new_complexity": new_complexity,
        "base_complexity": base_complexity,
        "input_norm": (input_i_mw / _W_AVERAGE_LOAD).astype(np.float32),
        "new_output": new_output,
        "enable_constraints": np.asarray(enable_constraints, dtype=np.float32),
        "l_array": l_array,
    }


def generate_extra_data_parallel(
    new_data_num=1000,
    line_limit_rate=1,
    dict_name="training_data",
    storage_data_name="storage_data_1888rte.npz",
    case_dist="case_data",
    case_type="matpower",
    case_name="case1888rte_pro",
    ptdf_date="2017-01-01",
    new_add_num=10,
    perturb_zero_ratio=PERTURB_ZERO_RATIO,
    perturb_nz_ratio=PERTURB_NZ_RATIO,
    load_compensation_ratio=LOAD_COMPENSATION_RATIO,
    min_switch_complexity_ratio=MIN_SWITCH_COMPLEXITY_RATIO,
    save_interval=SAVE_INTERVAL,
    output_suffix="newinput_parallel",
    enable_model_name=DEFAULT_ENABLE_MODEL_NAME,
    enable_model_dir=DEFAULT_ENABLE_MODEL_DIR,
    num_workers=None,
    gurobi_threads_per_worker=1,
    gurobi_output_flag=0,
    torch_threads_per_worker=1,
    max_pending_factor=2,
    seed=20260428,
):
    if num_workers is None:
        cpu_num = os.cpu_count() or 1
        num_workers = max(1, cpu_num - 1)
    num_workers = max(1, int(num_workers))
    max_pending = max(num_workers * int(max_pending_factor), num_workers)

    base_data_file = f"all_training_data_{case_name}.npz"
    d = _load_array(os.path.join(dict_name, base_data_file))
    input_raw = d["input_data"]
    average_load = float(np.asarray(d["average_load"]).item())

    if input_raw.ndim == 3:
        input_raw = np.transpose(input_raw, (0, 2, 1))
    input_load = input_raw * average_load
    sample_num, bus_num, time_num = input_load.shape

    output_raw = d["output_data"]
    if output_raw.ndim == 3:
        output_raw = np.transpose(output_raw, (0, 2, 1))
    base_output_ref = output_raw.copy()

    storage_data = _load_array(os.path.join(dict_name, storage_data_name))

    ptdf_path = f"{case_dist}/{case_type}/{case_name}/{ptdf_date}.json.gz"
    with gzip.open(ptdf_path) as f:
        original_data = json.loads(f.read())

    key_list = list(original_data["Buses"].keys())
    unit_names, units = zip(*original_data["Generators"].items())
    ng = len(units)
    line_names, lines = (
        zip(*original_data["Transmission lines"].items())
        if "Transmission lines" in original_data
        else ([], [])
    )
    nl = len(lines)

    out_data_name = f"all_training_data_{case_name}_{output_suffix}"
    out_npz = os.path.join(dict_name, out_data_name + ".npz")

    if os.path.exists(out_npz):
        h_in, h_out, h_en, h_l = get_history_data(dict_name, out_data_name + ".npz")
        history_input = h_in.copy()
        history_output = h_out.copy()
        history_enable = h_en.copy()
        history_l = h_l.copy()
    else:
        history_input = input_raw.copy()
        history_output = output_raw.copy()
        history_enable = d["enable_data"].copy()
        history_l = d["l_data"].copy()
        if history_l.ndim == 3:
            history_l = np.transpose(history_l, (0, 2, 1))

    perturb_zero = perturb_zero_ratio * average_load
    perturb_nz = perturb_nz_ratio * average_load

    target_data_num = min(new_data_num, sample_num)
    max_attempts = max(sample_num * 10, target_data_num * 20)

    skip_count = 0
    written_count = 0
    no_answer_count = 0
    attempt_count = 0

    print(
        f"[parallel] workers={num_workers}, gurobi_threads_per_worker={gurobi_threads_per_worker}, "
        f"target={target_data_num}, max_attempts={max_attempts}"
    )

    init_args = (
        original_data,
        key_list,
        storage_data,
        average_load,
        perturb_zero,
        perturb_nz,
        load_compensation_ratio,
        min_switch_complexity_ratio,
        line_limit_rate,
        new_add_num,
        time_num,
        ng,
        nl,
        enable_model_name,
        enable_model_dir,
        gurobi_threads_per_worker,
        gurobi_output_flag,
        torch_threads_per_worker,
    )

    with ProcessPoolExecutor(max_workers=num_workers, initializer=_worker_init, initargs=init_args) as executor:
        pending = {}

        while written_count < target_data_num and (attempt_count < max_attempts or pending):
            while (
                attempt_count < max_attempts
                and len(pending) < max_pending
                and written_count < target_data_num
            ):
                source_i = attempt_count % sample_num
                task_seed = _make_seed(seed, attempt_count, source_i)
                future = executor.submit(
                    _worker_solve_one,
                    attempt_count,
                    source_i,
                    input_load[source_i],
                    base_output_ref[source_i],
                    task_seed,
                )
                pending[future] = attempt_count
                attempt_count += 1

            if not pending:
                break

            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                _ = pending.pop(future)
                result = future.result()
                status = result["status"]

                if status == "no_answer":
                    no_answer_count += 1
                    continue

                if status == "skip":
                    skip_count += 1
                    continue

                if status != "ok":
                    raise RuntimeError(f"Unexpected worker status: {status}")

                if written_count >= target_data_num:
                    continue

                write_i = written_count
                history_input[write_i] = result["input_norm"]
                history_output[write_i] = result["new_output"]
                history_enable[write_i, :] = result["enable_constraints"]
                history_l[write_i, :, :] = result["l_array"]

                written_count += 1
                print(
                    f"[ok] write={written_count}/{target_data_num}, "
                    f"source={result['source_i']}, "
                    f"switch_complexity={result['new_complexity']}/{result['base_complexity']}, "
                    f"attempt={result['attempt_id'] + 1}/{max_attempts}"
                )

                if written_count > 0 and written_count % save_interval == 0:
                    _save(
                        out_npz,
                        history_input,
                        history_output,
                        history_enable,
                        history_l,
                        average_load,
                    )
                    print(f"[autosave] written={written_count}")

    _save(out_npz, history_input, history_output, history_enable, history_l, average_load)
    print(
        f"\nDone: valid={written_count}/{target_data_num}, attempts={attempt_count}, "
        f"skip={skip_count}, no_answer={no_answer_count}"
    )
    return written_count


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Parallel UC data generation.")
    parser.add_argument("--new-data-num", type=int, default=1365)
    parser.add_argument("--line-limit-rate", type=float, default=1.0)
    parser.add_argument("--dict-name", type=str, default="training_data")
    parser.add_argument("--storage-data-name", type=str, default="storage_data_case1888rte.npz")
    parser.add_argument("--case-dist", type=str, default="case_data")
    parser.add_argument("--case-type", type=str, default="matpower")
    parser.add_argument("--case-name", type=str, default="case1888rte_pro")
    parser.add_argument("--ptdf-date", type=str, default="2017-01-01")
    parser.add_argument("--new-add-num", type=int, default=10)
    parser.add_argument("--perturb-zero-ratio", type=float, default=PERTURB_ZERO_RATIO)
    parser.add_argument("--perturb-nz-ratio", type=float, default=PERTURB_NZ_RATIO)
    parser.add_argument("--load-compensation-ratio", type=float, default=LOAD_COMPENSATION_RATIO)
    parser.add_argument("--min-switch-complexity-ratio", type=float, default=MIN_SWITCH_COMPLEXITY_RATIO)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument("--output-suffix", type=str, default="newinput")
    parser.add_argument("--enable-model-name", type=str, default=DEFAULT_ENABLE_MODEL_NAME)
    parser.add_argument("--enable-model-dir", type=str, default=DEFAULT_ENABLE_MODEL_DIR)
    parser.add_argument("--num-workers", type=int, default=max((os.cpu_count() or 1) // 2, 1))
    parser.add_argument("--gurobi-threads-per-worker", type=int, default=1)
    parser.add_argument("--gurobi-output-flag", type=int, default=0)
    parser.add_argument("--torch-threads-per-worker", type=int, default=1)
    parser.add_argument("--max-pending-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260428)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    generate_extra_data_parallel(
        new_data_num=args.new_data_num,
        line_limit_rate=args.line_limit_rate,
        dict_name=args.dict_name,
        storage_data_name=args.storage_data_name,
        case_dist=args.case_dist,
        case_type=args.case_type,
        case_name=args.case_name,
        ptdf_date=args.ptdf_date,
        new_add_num=args.new_add_num,
        perturb_zero_ratio=args.perturb_zero_ratio,
        perturb_nz_ratio=args.perturb_nz_ratio,
        load_compensation_ratio=args.load_compensation_ratio,
        min_switch_complexity_ratio=args.min_switch_complexity_ratio,
        save_interval=args.save_interval,
        output_suffix=args.output_suffix,
        enable_model_name=args.enable_model_name,
        enable_model_dir=args.enable_model_dir,
        num_workers=args.num_workers,
        gurobi_threads_per_worker=args.gurobi_threads_per_worker,
        gurobi_output_flag=args.gurobi_output_flag,
        torch_threads_per_worker=args.torch_threads_per_worker,
        max_pending_factor=args.max_pending_factor,
        seed=args.seed,
    )
