"""
Parallel version of generate_data.py.

Design:
- Each worker solves one day case independently.
- Main process merges results in calendar order and writes final dataset.
"""

import argparse
import gzip
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import scipy.io
import torch

from ml_basic_functions import get_load, magnify_peak_valley_difference
from ml_basic_functions import ml_load_model
from uc_gurobi_constraint_filter import run_constraint_filter


USE_NPZ = True
DEFAULT_ENABLE_MODEL_NAME = "constraints_DNN_training_data_case1888rte_pro_random_1000_50000_[256, 256, 256]_0.0002"
DEFAULT_ENABLE_MODEL_DIR = "module_save"

_W_STORAGE_DATA = None
_W_LINE_LIMIT_RATE = None
_W_NEW_ADD_NUM = None
_W_MAGNIFY_RATE = None
_W_IS_DECREASED = None
_W_USE_ENABLE_MODEL = None
_W_ENABLE_MODEL = None


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


def _load_storage_or_solar(npz_path):
    if os.path.exists(npz_path):
        return dict(np.load(npz_path, allow_pickle=True))
    mat_path = npz_path.replace(".npz", ".mat")
    if os.path.exists(mat_path):
        return scipy.io.loadmat(mat_path)
    raise FileNotFoundError(f"Neither {npz_path} nor {mat_path} found")


def _worker_init(
    storage_data_path,
    line_limit_rate,
    new_add_num,
    magnify_rate,
    is_decreased,
    use_enable_model,
    enable_model_name,
    enable_model_dir,
    gurobi_threads_per_worker,
    gurobi_output_flag,
    torch_threads_per_worker,
):
    global _W_STORAGE_DATA
    global _W_LINE_LIMIT_RATE
    global _W_NEW_ADD_NUM
    global _W_MAGNIFY_RATE
    global _W_IS_DECREASED
    global _W_USE_ENABLE_MODEL
    global _W_ENABLE_MODEL

    os.environ["UC_GUROBI_THREADS"] = str(int(gurobi_threads_per_worker))
    os.environ["UC_GUROBI_OUTPUTFLAG"] = str(int(gurobi_output_flag))

    _W_STORAGE_DATA = _load_storage_or_solar(storage_data_path)
    _W_LINE_LIMIT_RATE = float(line_limit_rate)
    _W_NEW_ADD_NUM = int(new_add_num)
    _W_MAGNIFY_RATE = float(magnify_rate)
    _W_IS_DECREASED = bool(is_decreased)
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


def _worker_solve_day(day_idx, file_name, case_file_path):
    with gzip.open(case_file_path) as f:
        data = json.loads(f.read())

    load_data = get_load(data)
    new_load_data = magnify_peak_valley_difference(load_data, _W_MAGNIFY_RATE)

    unit_names, units = zip(*data["Generators"].items())
    ng = len(units)
    line_names, lines = (
        zip(*data["Transmission lines"].items())
        if "Transmission lines" in data
        else ([], [])
    )
    nl = len(lines)

    time_horizon = data["Parameters"].get("Time horizon (h)") or data["Parameters"].get("Time (h)")
    time_step = data["Parameters"].get("Time step (min)", 60)
    assert 60 % time_step == 0
    time_multiplier = int(60 / time_step)
    nt = time_horizon * time_multiplier

    bus_names, buses = zip(*data["Buses"].items())
    nb = len(buses)

    if _W_USE_ENABLE_MODEL:
        b_num, t_num = new_load_data.shape
        ml_input_i = np.transpose(new_load_data).astype(np.float32).reshape((b_num * t_num))
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
        data,
        is_warm_start=False,
        use_new_input=True,
        use_storage=True,
        use_new_min_up_down_time=True,
        use_solar=False,
        add_more_constr=False,
        storage_data=_W_STORAGE_DATA,
        new_input=new_load_data,
        line_limit_rate=_W_LINE_LIMIT_RATE,
        new_add_num=_W_NEW_ADD_NUM,
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

    active_line_hits = []
    if _W_IS_DECREASED:
        normal_flow_limit = np.array(
            [_W_LINE_LIMIT_RATE * line.get("Normal flow limit (MW)", 1e6) for line in lines],
            dtype=np.float32,
        )
        line_max_abs = np.max(np.abs(l_array), axis=1)
        delta = normal_flow_limit - line_max_abs
        active_line_hits = np.where(delta < 1e-3)[0].tolist()

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


def generate_365_data_parallel(
    case_dict="case_data",
    case_type="matpower",
    case_name="case1888rte",
    ptdf_date="2017-01-01",
    data_year="2017",
    data_dist_name="training_data",
    storage_data_name="storage_data",
    new_add_num=5,
    magnify_rate=1.05,
    line_limit_rate=1.0,
    is_decreased=False,
    use_enable_model=False,
    enable_model_name=DEFAULT_ENABLE_MODEL_NAME,
    enable_model_dir=DEFAULT_ENABLE_MODEL_DIR,
    num_workers=None,
    gurobi_threads_per_worker=1,
    gurobi_output_flag=0,
    torch_threads_per_worker=1,
    max_pending_factor=2,
):
    start_time = time.time()
    ptdf_path = case_dict + "/" + case_type + "/" + case_name + "/" + ptdf_date + ".json.gz"
    with gzip.open(ptdf_path) as f:
        ptdf_data = json.loads(f.read())
    ptdf = get_ptdf(ptdf_data)

    date_tasks = _build_date_tasks(case_dict, case_type, case_name, data_year)
    if num_workers is None:
        cpu_num = os.cpu_count() or 1
        num_workers = max(1, cpu_num - 1)
    num_workers = max(1, int(num_workers))
    max_pending = max(num_workers, num_workers * int(max_pending_factor))

    storage_npz = os.path.join(data_dist_name, storage_data_name + ".npz")

    print(
        f"[parallel] workers={num_workers}, gurobi_threads_per_worker={gurobi_threads_per_worker}, "
        f"days={len(date_tasks)}, use_enable_model={use_enable_model}"
    )

    init_args = (
        storage_npz,
        line_limit_rate,
        new_add_num,
        magnify_rate,
        is_decreased,
        use_enable_model,
        enable_model_name,
        enable_model_dir,
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
    nb = ng = nl = nt = None

    with ProcessPoolExecutor(max_workers=num_workers, initializer=_worker_init, initargs=init_args) as executor:
        while next_emit < len(date_tasks) or pending:
            while submitted < len(date_tasks) and len(pending) < max_pending:
                day_idx, file_name, case_file_path = date_tasks[submitted]
                future = executor.submit(_worker_solve_day, day_idx, file_name, case_file_path)
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
                    ng = result["ng"]
                    nl = result["nl"]
                    nt = result["nt"]
                    print(f"[ok] day={next_emit + 1}/{len(date_tasks)}, file={file_name}")
                else:
                    no_answer_num += 1
                    print(f"[skip] day={next_emit + 1}/{len(date_tasks)}, file={file_name}")
                next_emit += 1

    data_num = len(input_list)
    if data_num == 0:
        raise RuntimeError("No feasible data generated. All daily solves returned no_answer.")

    input_array = np.asarray(input_list, dtype=np.float32)
    output_array = np.asarray(output_list, dtype=np.float32)
    output_array_l_enable = np.asarray(enable_list, dtype=np.float32)
    l_array = np.asarray(l_list, dtype=np.float32)

    average_load = float(np.sum(input_array) / (data_num * nt * nb))
    input_array = input_array / average_load

    active_lines_sorted = sorted(active_lines)

    if is_decreased:
        gate = 0.005
        bus_line_active = get_class(ptdf, active_lines_sorted, gate)
        bus_classes = np.zeros(nb)
        affected_list = []
        class_num = 0
        for i in range(nb):
            line_is_affected = bus_line_active[i]
            if line_is_affected not in affected_list:
                affected_list.append(line_is_affected)
                bus_classes[i] = class_num
                class_num += 1
            else:
                class_index = affected_list.index(line_is_affected)
                bus_classes[i] = class_index
        class_len = len(affected_list)

        input_array_decreased = np.zeros([data_num, class_len, nt], dtype=np.float32)
        for i in range(nb):
            class_i = int(bus_classes[i])
            input_array_decreased[:, class_i, :] = (
                input_array_decreased[:, class_i, :] + input_array[:, i, :]
            )

        dict_name = data_dist_name
        file_name = "training_data_" + case_name + "_" + storage_data_name + "_decreased.mat"
        npz_name = "training_data_" + case_name + "_" + storage_data_name + "_decreased.npz"
        file_path = os.path.join(dict_name, file_name)
        npz_path = os.path.join(dict_name, npz_name)
        if USE_NPZ:
            np.savez_compressed(
                npz_path,
                input_data=np.transpose(input_array_decreased, (0, 2, 1)),
                output_data=np.transpose(output_array, (0, 2, 1)),
                l_data=np.transpose(l_array, (0, 2, 1)),
                average_load=float(average_load),
                active_lines=np.array(active_lines_sorted),
            )
        else:
            scipy.io.savemat(
                file_path,
                {
                    "input_data": input_array_decreased,
                    "output_data": output_array,
                    "l_data": l_array,
                    "average_load": average_load,
                    "active_lines": active_lines_sorted,
                },
            )
    else:
        dict_name = data_dist_name
        file_name = "training_data_" + case_name + "_" + storage_data_name + ".mat"
        file_name_all_data = "all_training_data_" + case_name + "_" + storage_data_name + ".mat"
        npz_name = "training_data_" + case_name + "_" + storage_data_name + ".npz"
        npz_name_all = "all_training_data_" + case_name + "_" + storage_data_name + ".npz"
        file_path = os.path.join(dict_name, file_name)
        file_path_all_data = os.path.join(dict_name, file_name_all_data)
        npz_path = os.path.join(dict_name, npz_name)
        npz_path_all = os.path.join(dict_name, npz_name_all)

        if USE_NPZ:
            common = dict(
                average_load=float(average_load),
                active_lines=np.array(active_lines_sorted),
                enable_data=output_array_l_enable,
            )
            np.savez_compressed(
                npz_path,
                input_data=np.transpose(input_array, (0, 2, 1)),
                output_data=np.transpose(output_array, (0, 2, 1)),
                l_data=np.transpose(l_array, (0, 2, 1)),
                **common,
            )
            np.savez_compressed(
                npz_path_all,
                input_data=np.transpose(input_array, (0, 2, 1)),
                output_data=np.transpose(output_array, (0, 2, 1)),
                l_data=np.transpose(l_array, (0, 2, 1)),
                **common,
            )
        else:
            scipy.io.savemat(
                file_path,
                {
                    "input_data": input_array,
                    "output_data": output_array,
                    "l_data": l_array,
                    "average_load": average_load,
                    "active_lines": active_lines_sorted,
                    "enable_data": output_array_l_enable,
                },
            )
            scipy.io.savemat(
                file_path_all_data,
                {
                    "input_data": input_array,
                    "output_data": output_array,
                    "l_data": l_array,
                    "average_load": average_load,
                    "active_lines": active_lines_sorted,
                    "enable_data": output_array_l_enable,
                },
            )
        print(f"generate {data_num} data over!")

    end_time = time.time()
    all_time = end_time - start_time
    print(f"generate {data_num} data cost {all_time} seconds!")
    print(f"no_answer_num={no_answer_num}")
    return 0


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Parallel version of generate_data.py")
    parser.add_argument("--case-dict", type=str, default="case_data")
    parser.add_argument("--case-type", type=str, default="matpower")
    parser.add_argument("--case-name", type=str, default="case1888rte")
    parser.add_argument("--ptdf-date", type=str, default="2017-01-01")
    parser.add_argument("--data-year", type=str, default="2017")
    parser.add_argument("--data-dist-name", type=str, default="training_data")
    parser.add_argument("--storage-data-name", type=str, default="storage_data_case1888rte_0%")
    parser.add_argument("--new-add-num", type=int, default=5)
    parser.add_argument("--magnify-rate", type=float, default=1.0)
    parser.add_argument("--line-limit-rate", type=float, default=1.0)
    parser.add_argument("--is-decreased", action="store_true")
    parser.add_argument(
        "--use-enable-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use neural network to predict initial constraints_enable.",
    )
    parser.add_argument("--enable-model-name", type=str, default=DEFAULT_ENABLE_MODEL_NAME)
    parser.add_argument("--enable-model-dir", type=str, default=DEFAULT_ENABLE_MODEL_DIR)
    parser.add_argument("--num-workers", type=int, default=max((os.cpu_count() or 1) // 2, 1))
    parser.add_argument("--gurobi-threads-per-worker", type=int, default=1)
    parser.add_argument("--gurobi-output-flag", type=int, default=0)
    parser.add_argument("--torch-threads-per-worker", type=int, default=1)
    parser.add_argument("--max-pending-factor", type=int, default=2)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    generate_365_data_parallel(
        case_dict=args.case_dict,
        case_type=args.case_type,
        case_name=args.case_name,
        ptdf_date=args.ptdf_date,
        data_year=args.data_year,
        data_dist_name=args.data_dist_name,
        storage_data_name=args.storage_data_name,
        new_add_num=args.new_add_num,
        magnify_rate=args.magnify_rate,
        line_limit_rate=args.line_limit_rate,
        is_decreased=args.is_decreased,
        use_enable_model=args.use_enable_model,
        enable_model_name=args.enable_model_name,
        enable_model_dir=args.enable_model_dir,
        num_workers=args.num_workers,
        gurobi_threads_per_worker=args.gurobi_threads_per_worker,
        gurobi_output_flag=args.gurobi_output_flag,
        torch_threads_per_worker=args.torch_threads_per_worker,
        max_pending_factor=args.max_pending_factor,
    )
