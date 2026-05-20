"""
Parallel version of generate_all_data.py.

Design:
- Worker process handles one sample generation + UC solve.
- Main process keeps result order by sample index and performs all file writes.
"""

import argparse
import gzip
import json
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import scipy.io
import torch

from ml_basic_functions import ml_load_model
from uc_gurobi_constraint_filter import run_constraint_filter


DEFAULT_ENABLE_MODEL_NAME = "constraints_DNN_training_data_case1888rte_storage_data_case1888rte_200%_random_1000_50000_[256, 256, 256]_0.0002"
DEFAULT_ENABLE_MODEL_DIR = "module_save"

_W_ORIGINAL_DATA = None
_W_STORAGE_DATA = None
_W_DELTA_INPUT_LOAD = None
_W_CENTER_INPUT_LOAD = None
_W_AVERAGE_LOAD = None
_W_ENABLE_MODEL = None
_W_LINE_LIMIT_RATE = None
_W_NEW_ADD_NUM = None
_W_TIME_NUM = None
_W_NG = None
_W_NL = None
_W_USE_ENABLE_MODEL = None


def _load_array(path):
    npz = path.replace(".mat", ".npz")
    if os.path.exists(npz):
        return dict(np.load(npz, allow_pickle=True))
    return scipy.io.loadmat(path)


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


def _save_npz(file_path, input_data, output_data, enable_data, l_data, average_load):
    np.savez_compressed(
        file_path,
        input_data=np.transpose(input_data, (0, 2, 1)),
        output_data=np.transpose(output_data, (0, 2, 1)),
        enable_data=enable_data,
        l_data=np.transpose(l_data, (0, 2, 1)),
        average_load=float(average_load),
    )


def _make_seed(base_seed, sample_idx):
    return int((base_seed + 1000003 * sample_idx) % (2**32 - 1))


def _worker_init(
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
    global _W_ORIGINAL_DATA
    global _W_STORAGE_DATA
    global _W_DELTA_INPUT_LOAD
    global _W_CENTER_INPUT_LOAD
    global _W_AVERAGE_LOAD
    global _W_ENABLE_MODEL
    global _W_LINE_LIMIT_RATE
    global _W_NEW_ADD_NUM
    global _W_TIME_NUM
    global _W_NG
    global _W_NL
    global _W_USE_ENABLE_MODEL

    os.environ["UC_GUROBI_THREADS"] = str(int(gurobi_threads_per_worker))
    os.environ["UC_GUROBI_OUTPUTFLAG"] = str(int(gurobi_output_flag))

    _W_ORIGINAL_DATA = original_data
    _W_STORAGE_DATA = storage_data
    _W_DELTA_INPUT_LOAD = delta_input_load
    _W_CENTER_INPUT_LOAD = center_input_load
    _W_AVERAGE_LOAD = float(average_load)
    _W_LINE_LIMIT_RATE = float(line_limit_rate)
    _W_NEW_ADD_NUM = int(new_add_num)
    _W_TIME_NUM = int(time_num)
    _W_NG = int(ng)
    _W_NL = int(nl)
    _W_USE_ENABLE_MODEL = bool(use_enable_model)

    if torch_threads_per_worker is not None:
        threads = max(1, int(torch_threads_per_worker))
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)

    if _W_USE_ENABLE_MODEL:
        enable_model, *_ = ml_load_model(enable_model_name, enable_model_dir)
        enable_model = enable_model.to(torch.device("cpu"))
        enable_model.eval()
        _W_ENABLE_MODEL = enable_model
    else:
        _W_ENABLE_MODEL = None


def _worker_generate_one(sample_idx, seed):
    rng = np.random.default_rng(seed)
    rand = 2.0 * rng.random(_W_DELTA_INPUT_LOAD.shape) - 1.0
    input_i = rand * _W_DELTA_INPUT_LOAD + _W_CENTER_INPUT_LOAD

    if _W_USE_ENABLE_MODEL:
        b_num, t_num = input_i.shape
        ml_input_i = np.transpose(input_i).astype(np.float32).reshape((b_num * t_num))
        ml_input_i = torch.from_numpy(ml_input_i).unsqueeze(0)
        with torch.no_grad():
            constraints_enable = _W_ENABLE_MODEL(ml_input_i).squeeze(0).cpu().numpy()
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
        _W_ORIGINAL_DATA,
        is_warm_start=False,
        use_new_input=True,
        use_storage=True,
        use_new_min_up_down_time=True,
        use_solar=False,
        add_more_constr=False,
        storage_data=_W_STORAGE_DATA,
        new_input=input_i,
        line_limit_rate=_W_LINE_LIMIT_RATE,
        new_add_num=_W_NEW_ADD_NUM,
        add_constr_gate=0.7,
        initial_constraints=constraints_enable,
        use_initial_enable=use_initial_enable,
    )

    if no_answer:
        return {"status": "no_answer", "sample_idx": sample_idx}

    output_array = np.array(
        [[u_start[i, t] for t in range(_W_TIME_NUM)] for i in range(_W_NG)],
        dtype=np.float32,
    )
    if isinstance(l_start, np.ndarray):
        l_array = l_start.astype(np.float32, copy=False)
    else:
        l_array = np.array(
            [[l_start[i, t] for t in range(_W_TIME_NUM)] for i in range(_W_NL)],
            dtype=np.float32,
        )

    return {
        "status": "ok",
        "sample_idx": sample_idx,
        "input_array": (input_i / _W_AVERAGE_LOAD).astype(np.float32),
        "output_array": output_array,
        "enable_array": np.asarray(enable_constraints, dtype=np.float32),
        "l_array": l_array,
    }


def generate_extra_data_parallel(
    new_data_num=1000,
    save_gap=50,
    line_limit_rate=1,
    dict_name="training_data",
    training_data="training_data.mat",
    storage_data_name="storage_data",
    case_dist="case_data",
    case_type="matpower",
    case_name="case1888rte",
    ptdf_date="2017-01-01",
    new_add_num=5,
    use_enable_model=True,
    enable_model_name=DEFAULT_ENABLE_MODEL_NAME,
    enable_model_dir=DEFAULT_ENABLE_MODEL_DIR,
    num_workers=None,
    gurobi_threads_per_worker=1,
    gurobi_output_flag=0,
    torch_threads_per_worker=1,
    max_pending_factor=2,
    seed=20260428,
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

    storage_data = _load_array(os.path.join(dict_name, storage_data_name + ".mat"))

    ptdf_path = case_dist + "/" + case_type + "/" + case_name + "/" + ptdf_date + ".json.gz"
    with gzip.open(ptdf_path) as f:
        original_data = json.loads(f.read())

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

    if num_workers is None:
        cpu_num = os.cpu_count() or 1
        num_workers = max(1, cpu_num - 1)
    num_workers = max(1, int(num_workers))
    max_pending = max(num_workers, int(max_pending_factor) * num_workers)

    print(
        f"[parallel] workers={num_workers}, gurobi_threads_per_worker={gurobi_threads_per_worker}, "
        f"new_data_num={new_data_num}, use_enable_model={use_enable_model}"
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

    with ProcessPoolExecutor(max_workers=num_workers, initializer=_worker_init, initargs=init_args) as executor:
        while next_emit < new_data_num or pending:
            while submitted < new_data_num and len(pending) < max_pending:
                task_seed = _make_seed(seed, submitted)
                future = executor.submit(_worker_generate_one, submitted, task_seed)
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
                    print(f"[ok] sample={next_emit}, accepted={accepted}")

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
                        )
                        print(f"[autosave] sample={next_emit}, total={len(history_input_data)}")
                else:
                    no_answer_num += 1
                    print(f"[skip] sample={next_emit}, no feasible answer")
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
    )
    print(f"try to add {new_data_num} new data, {no_answer_num} data have no answer")
    return 0


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Parallel version of generate_all_data.py")
    parser.add_argument("--new-data-num", type=int, default=1000)
    parser.add_argument("--save-gap", type=int, default=50)
    parser.add_argument("--line-limit-rate", type=float, default=1.0)
    parser.add_argument("--dict-name", type=str, default="training_data")
    parser.add_argument("--training-data", type=str, default="training_data_case1888rte_pro.npz")
    parser.add_argument("--storage-data-name", type=str, default="storage_data_case1888rte")
    parser.add_argument("--case-dist", type=str, default="case_data")
    parser.add_argument("--case-type", type=str, default="matpower")
    parser.add_argument("--case-name", type=str, default="case1888rte")
    parser.add_argument("--ptdf-date", type=str, default="2017-01-01")
    parser.add_argument("--new-add-num", type=int, default=5)
    parser.add_argument(
        "--use-enable-model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use neural network to predict initial constraints_enable.",
    )
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
        save_gap=args.save_gap,
        line_limit_rate=args.line_limit_rate,
        dict_name=args.dict_name,
        training_data=args.training_data,
        storage_data_name=args.storage_data_name,
        case_dist=args.case_dist,
        case_type=args.case_type,
        case_name=args.case_name,
        ptdf_date=args.ptdf_date,
        new_add_num=args.new_add_num,
        use_enable_model=args.use_enable_model,
        enable_model_name=args.enable_model_name,
        enable_model_dir=args.enable_model_dir,
        num_workers=args.num_workers,
        gurobi_threads_per_worker=args.gurobi_threads_per_worker,
        gurobi_output_flag=args.gurobi_output_flag,
        torch_threads_per_worker=args.torch_threads_per_worker,
        max_pending_factor=args.max_pending_factor,
        seed=args.seed,
    )
