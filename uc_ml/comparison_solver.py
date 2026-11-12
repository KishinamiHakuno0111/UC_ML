"""Adapter to the existing UC mathematical model, without editing its source.

The AST cut returns the model immediately before its original optimize call.
Constraint generation, status guards and optional numerical settings live here.
"""

from __future__ import annotations

import ast
import hashlib
import os
import time
import traceback
from pathlib import Path

import numpy as np

from .comparison_common import read_json, save_npz, sha256, write_json

ROBUST_PARAMETERS = dict(Method=2, BarHomogeneous=1, NumericFocus=3,
                         Aggregate=0, ScaleFlag=2, Presolve=0, Crossover=0)
TERMINAL = ("legacy_accepted", "no_answer", "infeasible")


def model_builder(source):
    import gurobipy as gp
    tree = ast.parse(source)
    names = {"try_get_from_list", "timeseries", "map_list", "set_warmstart", "run_uc"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    function = next(node for node in nodes if node.name == "run_uc")
    cuts = [i for i, node in enumerate(function.body)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "optimize"]
    if len(cuts) != 1:
        raise ValueError("Unsupported run_uc source: expected one top-level optimize call")
    prefix = function.body[:cuts[0]]
    signature = hashlib.sha256(ast.dump(ast.Module(body=prefix, type_ignores=[])).encode()).hexdigest()
    function.body = prefix + [ast.Return(value=ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]))]
    namespace = dict(os=os, time=time, np=np, gp=gp, GRB=gp.GRB)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "unchanged_uc_model", "exec"), namespace)
    return namespace["run_uc"], signature


def derive_start_stop(u, initial):
    difference = np.diff(np.column_stack([initial, u]), axis=1)
    return np.maximum(difference, 0), np.maximum(-difference, 0)


def mathematical_hash(model):
    model.update()
    matrix = model.getA().tocsr()
    digest = hashlib.sha256()
    for array in (matrix.data, matrix.indices, matrix.indptr,
                  model.getAttr("RHS"), model.getAttr("Sense"), model.getAttr("Obj"),
                  model.getAttr("LB"), model.getAttr("UB"), model.getAttr("VType"),
                  [model.ModelSense, model.ObjCon]):
        value = np.ascontiguousarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def check_solution(model):
    value = np.asarray(model.getAttr("X"), dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite solver vector")
    residual = model.getA() @ value - np.asarray(model.getAttr("RHS"), dtype=np.float64)
    senses = np.asarray(model.getAttr("Sense"))
    violation = np.where(senses == "=", abs(residual),
                         np.where(senses == "<", np.maximum(residual, 0), np.maximum(-residual, 0)))
    row_error = float(np.max(violation, initial=0))
    bounds = max(float(np.max(np.asarray(model.getAttr("LB")) - value, initial=0)),
                 float(np.max(value - np.asarray(model.getAttr("UB")), initial=0)))
    integers = np.asarray(model.getAttr("VType")) != "C"
    integer_error = float(np.max(abs(value[integers] - np.rint(value[integers])), initial=0))
    objective = float(np.dot(np.asarray(model.getAttr("Obj"), dtype=np.float64), value) + model.ObjCon)
    if row_error > 1e-3 or bounds > 1e-5 or integer_error > 1e-5:
        raise ValueError(f"Invalid saved solution: rows={row_error}, bounds={bounds}, integrality={integer_error}")
    if not np.isclose(objective, model.ObjVal, rtol=1e-8, atol=1e-3):
        raise ValueError("Objective reconstruction failed")
    return dict(passed=True, max_row_residual=row_error, max_bound_violation=bounds,
                max_integer_error=integer_error, reconstructed_objective=objective), value


def strict_numeric(result):
    return (result.get("status") == "error" and result.get("error_kind") == "solver_status"
            and bool(result.get("rounds")) and result["rounds"][-1]["status"] == 12
            and result["rounds"][-1]["solutions"] == 0)


def solve(task):
    """One create-only result. Incomplete directories are evidence, not valid results."""
    import gurobipy as gp
    folder = Path(task["folder"])
    folder.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    write_json(folder / "task.json", task)
    result = dict(status="error", rounds=[], error_kind="exception", worker_pid=os.getpid())
    model = None
    try:
        build, signature = model_builder(Path(task["solver"]).read_text(encoding="utf-8-sig"))
        net = read_json(task["network"])
        with np.load(task["storage"], allow_pickle=False) as data:
            storage = {key: np.atleast_2d(data[key]) for key in data.files}
        with np.load(task["input"], allow_pickle=False) as data:
            load = np.asarray(data["load_MW"], dtype=np.float64).T
            u = data["u"].T if "u" in data.files else None
            enabled = data["enabled"].copy() if "enabled" in data.files else np.zeros(len(net.get("Transmission lines", {})))
        if load.ndim != 2 or not np.isfinite(load).all():
            raise ValueError("Invalid load array")
        result.update(build_ast_sha256=signature, input_sha256=sha256(task["input"]),
                      gurobi_version=list(gp.gurobi.version()), numeric_parameters=ROBUST_PARAMETERS if task.get("robust") else {})
        while True:
            loc = build(net, use_new_input=True, use_storage=True, use_new_min_up_down_time=True,
                        new_input=load, storage_data=storage, enable_constraints=enabled,
                        is_warm_start=False, fix_int=False, mip_gap=task["mip_gap"])
            model = loc["model"]
            model.update()
            if u is not None:
                if not np.isin(u, (0, 1)).all():
                    raise ValueError("Commitment must be binary")
                y, z = derive_start_stop(u, loc["init_u"])
                for name, values in (("u", u), ("y", y), ("z", z)):
                    for (g, t), value in np.ndenumerate(values):
                        variable = loc[name][g, t]
                        if value < variable.LB - 1e-8 or value > variable.UB + 1e-8:
                            result.update(status="infeasible", reason="fixed_binary_conflicts_with_original_bound")
                            return result
                        variable.LB = variable.UB = int(value)
                model.update()
                if any(v.LB != v.UB for v in model.getVars() if v.VType != "C"):
                    raise ValueError("Not all integer variables were fixed")
            model.Params.OutputFlag = 0
            model.Params.Threads = 1
            model.Params.Seed = 0
            before = mathematical_hash(model)
            if task.get("robust"):
                for key, value in ROBUST_PARAMETERS.items():
                    model.setParam(key, value)
                if mathematical_hash(model) != before:
                    raise ValueError("Numerical options changed the mathematical model")
            round_index = len(result["rounds"])
            model.Params.LogFile = str(folder / f"round_{round_index:03}.log")
            start = time.perf_counter()
            model.optimize()
            record = dict(status=int(model.Status), solutions=int(model.SolCount),
                          elapsed_seconds=time.perf_counter() - start, mathematical_sha256=before)
            result["rounds"].append(record)
            write_json(folder / f"round_{round_index:03}.json", record)
            if model.Status == gp.GRB.INFEASIBLE:
                result["status"] = "infeasible"
                break
            if model.Status != gp.GRB.OPTIMAL or model.SolCount == 0:
                result["error_kind"] = "solver_status"
                raise RuntimeError(f"Unresolved solver status {model.Status}, solutions {model.SolCount}")
            audit, vector = check_solution(model)
            arrays = {}
            for name, size in ([(name, loc["NG"]) for name in ("u", "y", "z", "p", "r", "c")]
                    + [(name, loc["sto_num"]) for name in ("pch", "pdis", "stopower", "sto_c")]
                    + [("l_punish", loc["NL"])]):
                arrays[name] = np.array([[loc[name][i, t].X for t in range(loc["NT"])] for i in range(size)], dtype=np.float64).reshape(size, loc["NT"])
            arrays["balance_slack"] = np.array([loc["flow_equal_punish"][t].X for t in range(loc["NT"])])
            if u is not None and not np.array_equal(np.rint(arrays["u"]), u):
                raise ValueError("Saved commitment differs from fixed neighbor")
            flows = loc["unit_to_line_matrix"] @ arrays["p"] - loc["ptdf"] @ load
            flows += loc["sto_to_line_matrix"] @ (arrays["pdis"] - arrays["pch"])
            limits = np.asarray(loc["normal_flow_limit"])
            ratio = np.max(abs(flows) / limits[:, None], axis=1) if len(limits) else np.zeros(0)
            ratio[limits >= 1e6] = 0
            slack = float(np.max(arrays["l_punish"], initial=0) + np.max(arrays["balance_slack"], initial=0))
            if slack > 1e-3 or float(np.max(ratio, initial=0)) <= 1:
                save_npz(folder / "solution.npz", **arrays, flows=flows, all_variables=vector,
                         enabled=enabled, load_MW=load)
                result.update(status="no_answer" if slack > 1e-3 else "legacy_accepted",
                              objective=float(model.ObjVal), bound=float(model.ObjBound), gap=float(model.MIPGap),
                              economic_cost=float(arrays["c"].sum() + (np.asarray(loc["startup_cost"])[:, None] * arrays["y"]).sum() + arrays["sto_c"].sum()),
                              slack=slack, solution_sha256=sha256(folder / "solution.npz"), audit=audit)
                break
            top = np.sort(ratio)[-min(10, len(ratio)):]
            new = enabled.copy()
            new[np.isin(ratio, top) & (ratio >= .7)] = 1
            if np.array_equal(new, enabled):
                raise RuntimeError("Constraint generation made no progress")
            enabled = new
            model.dispose()
            model = None
    except Exception:
        result.update(status="error", traceback=traceback.format_exc())
    finally:
        if model is not None:
            model.dispose()
        result.update(elapsed_seconds=time.perf_counter() - started,
                      cpu_seconds=time.process_time() - cpu_started)
        write_json(folder / "result.json", result)
    return result
