"""Nearest-schedule KNN-50 accuracy, cost and timing comparison."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import gzip
import multiprocessing as mp
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from .comparison_common import (bind_run, combine_metrics, disk_guard, fixed_split,
    mean_defined, metrics, predict, read_json, resolve_config, run_lock, save_npz,
    sha256, status, trusted_model, write_json)
from .comparison_solver import TERMINAL, solve, strict_numeric


def nearest(inputs, test_id, train_ids, k=50):
    if test_id in train_ids or len(set(train_ids)) != len(train_ids) or len(train_ids) < k:
        raise ValueError("Neighbors must be distinct training IDs, excluding the test ID")
    query = np.asarray(inputs[test_id], dtype=np.float64)
    distances = [(float(np.sum((np.asarray(inputs[i], dtype=np.float64) - query)**2)), int(i))
                 for i in train_ids]
    return sorted(distances)[:k]


def select_candidate(rows):
    accepted = [(i, row) for i, row in enumerate(rows) if row["status"] == "legacy_accepted"]
    return min(accepted, key=lambda item: (item[1]["objective"], item[0]))[0] if accepted else None


def timing(lookup, batch, selection, candidates, clean):
    return dict(actual8_seconds=lookup + batch + selection if clean else None,
                ideal50_seconds=lookup + max(r["elapsed_seconds"] for r in candidates) + selection if clean else None,
                invalid_reason=None if clean else "resumed_or_numerically_retried_candidate_batch",
                lookup_seconds=lookup, selection_seconds=selection,
                candidate_cpu_seconds=sum(r["cpu_seconds"] for r in candidates))


def prepare_case(spec, base, destination):
    with np.load(base / spec["data"], allow_pickle=False) as data:
        x = np.asarray(data["input_data"][:1100], dtype=np.float32)
        y = np.asarray(data["output_data"][:1100], dtype=np.float32)
        average = float(data["average_load"])
    if x.ndim != 3 or x.shape[:2] != (1100, 36) or y.shape[:2] != (1100, 36):
        raise ValueError("Expected at least 1100 samples with 36 periods")
    if not np.isfinite(x).all() or not np.isin(y, (0, 1)).all() or not np.isfinite(average) or average <= 0:
        raise ValueError("Invalid input data/labels/normalization")
    with gzip.open(base / spec["network"], "rt", encoding="utf-8") as stream:
        import json
        network = json.load(stream)
    if len(network["Buses"]) != x.shape[2] or len(network["Generators"]) != y.shape[2]:
        raise ValueError("Dataset/network dimensions differ")
    horizon = network["Parameters"].get("Time horizon (h)") or network["Parameters"].get("Time (h)")
    if horizon != 36 or network["Parameters"].get("Time step (min)", 60) != 60:
        raise ValueError("The comparison uses 36 hourly periods")
    scale = float(spec.get("min_time_scale", 1))
    if scale != 1:
        for generator in network["Generators"].values():
            for key in ("Minimum uptime (h)", "Minimum downtime (h)"):
                generator[key] = max(1, int(np.ceil(generator.get(key, 1) * scale)))
    path = destination / "network.json"
    if path.exists():
        if read_json(path) != network:
            raise ValueError("Prepared network changed")
    else:
        write_json(path, network)
    return x, y, average, path


def saved_result(folder):
    path = folder / "result.json"
    if not path.exists():
        return None
    row = read_json(path)
    if (folder / "task.json").exists() and "input_sha256" in row:
        task = read_json(folder / "task.json")
        if sha256(task["input"]) != row["input_sha256"]:
            raise ValueError(f"Saved solver input changed: {folder}")
    if row["status"] in ("legacy_accepted", "no_answer"):
        if not row.get("audit", {}).get("passed") or sha256(folder / "solution.npz") != row["solution_sha256"]:
            raise ValueError(f"Invalid saved result: {folder}")
    if row["status"] not in TERMINAL and not strict_numeric(row):
        raise RuntimeError(f"Non-numerical error retained at {folder}; no automatic retry")
    return row


def attempt_folder(parent):
    """Preserve a partial process-crash attempt and start only its unresolved key."""
    parent.mkdir(parents=True, exist_ok=True)
    attempts = sorted(parent.glob("attempt_*"))
    for folder in attempts:
        row = saved_result(folder)
        if row is not None:
            return folder, row
    index = len(attempts)
    return parent / f"attempt_{index:03}", None


def execute_pending(pool, tasks):
    """At most eight in flight; after a fatal error drain, but do not submit more."""
    todo = iter(tasks)
    active, rows, fatal = {}, {}, None
    def submit():
        item = next(todo, None)
        if item is not None:
            key, task = item
            active[pool.submit(solve, task)] = key
    for _ in range(8):
        submit()
    while active:
        finished, _ = wait(active, return_when=FIRST_COMPLETED)
        for future in finished:
            key = active.pop(future)
            try:
                row = future.result()
                rows[key] = row
                if row["status"] not in TERMINAL and not strict_numeric(row):
                    fatal = RuntimeError(f"Non-numerical solver/check failure for {key}")
            except Exception as exc:
                fatal = exc
        if fatal is None:
            while len(active) < 8:
                before = len(active)
                submit()
                if len(active) == before:
                    break
    if fatal:
        raise fatal
    return rows


def compare_test(pool, output, name, index, x, y, average, common_task):
    test_id = fixed_split()["test"][index]
    folder = output / "results" / name / f"test_{index:03}"
    completed = folder / "comparison.json"
    if completed.exists():
        row = read_json(completed)
        audit = read_json(folder / "evaluation_audit.json")
        if not audit["passed"] or sha256(completed) != audit["comparison_sha256"]:
            raise ValueError("Completed test comparison changed")
        if row["case"] != name or row["index"] != index or row["source_id"] != test_id:
            raise ValueError("Completed test key changed")
        for path, expected in row["result_bindings"].items():
            if sha256(output / path) != expected:
                raise ValueError(f"Completed result changed: {path}")
            saved_result((output / path).parent)
        return row
    existed = folder.exists()
    folder.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    neighbors = nearest(x, test_id, fixed_split()["train"])
    lookup = time.perf_counter() - begin
    ids = [sid for _, sid in neighbors]
    neighbor_record = dict(test_id=test_id, source_ids=ids, squared_distances=[d for d, _ in neighbors])
    if (folder / "neighbors.json").exists():
        if read_json(folder / "neighbors.json") != neighbor_record:
            raise ValueError("Neighbor binding changed")
    else:
        write_json(folder / "neighbors.json", neighbor_record)
    rows, paths, pending = {}, {}, []
    def prepare_rank(rank, source):
        key = f"candidate_{rank:02}" if source is not None else "reference"
        parent = folder / key
        path, row = attempt_folder(parent)
        paths[rank] = path
        if row is not None:
            rows[rank] = row
            return
        input_path = parent / "input.npz"
        # Preserve the archived float32 MW reconstruction before solver conversion.
        arrays = dict(load_MW=x[test_id] * average)
        if source is not None:
            arrays["u"] = y[source]
        if not input_path.exists():
            save_npz(input_path, **arrays)
        else:
            with np.load(input_path, allow_pickle=False) as archive:
                if set(archive.files) != set(arrays) or any(not np.array_equal(archive[k], v) for k, v in arrays.items()):
                    raise ValueError("Interrupted candidate input changed")
        pending.append((rank, dict(common_task, input=str(input_path), folder=str(path))))
    start = time.perf_counter()
    for rank, source in enumerate(ids):
        prepare_rank(rank, source)
    rows.update(execute_pending(pool, pending))
    batch = time.perf_counter() - start
    # The unrestricted reference is not part of the KNN prediction latency.
    prepare_rank(50, None)
    rows.update(execute_pending(pool, [v for v in pending if v[0] == 50]))
    session = folder / f"session_{time.time_ns()}.json"
    write_json(session, dict(candidate_batch_seconds=batch, lookup_seconds=lookup,
                             newly_submitted_ranks=[r for r, _ in pending], resumed=existed))
    deferred = [rank for rank, row in rows.items() if strict_numeric(row)]
    if deferred:
        write_json(folder / "deferred.json", dict(ranks=deferred), replace=True)
        return None
    return finish_test(output, name, index, y, folder, rows, paths, lookup, batch, not existed)


def finish_test(output, name, index, y, folder, rows, paths, lookup, batch, clean):
    if set(rows) != set(range(51)) or any(row["status"] not in TERMINAL for row in rows.values()):
        raise ValueError("50 resolved candidates and one reference are required")
    started = time.perf_counter()
    candidates = [rows[i] for i in range(50)]
    best = select_candidate(candidates)
    selection = time.perf_counter() - started
    neighbors = read_json(folder / "neighbors.json")
    test_id = neighbors["test_id"]
    score = metrics(y[neighbors["source_ids"][best]], y[test_id]) if best is not None else None
    reference = rows[50]
    deviation = None
    if best is not None and reference["status"] == "legacy_accepted" and reference["objective"] != 0:
        deviation = 100 * (rows[best]["objective"] - reference["objective"]) / abs(reference["objective"])
    result = dict(case=name, index=index, source_id=test_id, failed_all50=best is None,
                  selected_rank=best, accuracy=score, candidate_counts=dict(Counter(r["status"] for r in candidates)),
                  cost_deviation_percent=deviation, reference_status=reference["status"],
                  reference_seconds=reference["elapsed_seconds"],
                  **timing(lookup, batch, selection, candidates, clean),
                  result_paths={str(k): str(v.relative_to(output)) for k, v in paths.items()},
                  result_bindings={str((v / "result.json").relative_to(output)): sha256(v / "result.json") for v in paths.values()})
    # Every attempt remains inspectable; retries are NOT hidden in clean latency averages.
    attempts = [read_json(path) for path in folder.glob("*/attempt_*/result.json")]
    attempts += [read_json(path) for path in folder.glob("*/numeric_resolution/result.json")]
    result["all_attempt_worker_seconds"] = sum(r["elapsed_seconds"] for r in attempts)
    result["all_attempt_cpu_seconds"] = sum(r["cpu_seconds"] for r in attempts)
    write_json(folder / "comparison.json", result)
    write_json(folder / "evaluation_audit.json", dict(passed=True, candidates=50, reference=1,
               selection="minimum_objective_among_accepted", comparison_sha256=sha256(folder / "comparison.json")))
    return result


def resolve_deferred(pool, output, name, index, y):
    folder = output / "results" / name / f"test_{index:03}"
    if (folder / "comparison.json").exists():
        return
    pending = read_json(folder / "deferred.json")["ranks"]
    paths, rows = {}, {}
    for rank in range(51):
        parent = folder / (f"candidate_{rank:02}" if rank < 50 else "reference")
        path, row = attempt_folder(parent)
        if rank in pending:
            resolution = parent / "numeric_resolution"
            if (resolution / "result.json").exists():
                row = saved_result(resolution)
            elif resolution.exists():
                raise RuntimeError(f"Interrupted numerical resolution retained: {resolution}")
            else:
                task = dict(read_json(path / "task.json"), folder=str(resolution), robust=True)
                row = pool.submit(solve, task).result()
            original = read_json(path / "result.json")
            # Settings may not change any rebuilt round shared with the first attempt.
            for before, after in zip(original["rounds"], row["rounds"]):
                if before["mathematical_sha256"] != after["mathematical_sha256"]:
                    raise ValueError("Numerical resolution model differs from original")
            if row["status"] not in TERMINAL:
                raise RuntimeError(f"Numerical resolution failed; no further retry: {resolution}")
            path = resolution
        paths[rank], rows[rank] = path, row
    sessions = [read_json(p) for p in sorted(folder.glob("session_*.json"))]
    clean = len(sessions) == 1 and not sessions[0]["resumed"] and all(k == 50 for k in pending)
    s = sessions[0]
    finish_test(output, name, index, y, folder, rows, paths, s["lookup_seconds"], s["candidate_batch_seconds"], clean)


def neural_predictions(spec, base, folder, x, y):
    import torch
    torch.set_num_threads(1)
    records = {}
    ids = fixed_split()["test"]
    for name, model_spec in spec.get("models", {}).items():
        path = folder / f"{name}_probabilities.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                probabilities = data["probabilities"]
                if not np.array_equal(data["source_ids"], ids):
                    raise ValueError("Neural test ordering changed")
        else:
            model = trusted_model(base / model_spec["path"])
            probabilities = predict(model, x[ids], model_spec.get("dnn_layout"))
            save_npz(path, probabilities=probabilities, source_ids=ids)
        if probabilities.shape != y[ids].shape:
            raise ValueError("Neural probability dimensions differ from frozen tests")
        records[name] = [metrics(p, y[i]) for p, i in zip(probabilities, ids)]
    return records


def summarize(output, case_names):
    summary = {}
    for name in case_names:
        rows = [read_json(path) for path in sorted((output / "results" / name).glob("*/comparison.json"))]
        if len({r["index"] for r in rows}) != len(rows) or any(
                r["case"] != name or not 0 <= r["index"] < 100 or
                r["source_id"] != fixed_split()["test"][r["index"]] for r in rows):
            raise ValueError("Unexpected or duplicate test key")
        good = [row for row in rows if not row["failed_all50"]]
        row = dict(completed=len(rows), target=100, failed_all50=len(rows)-len(good),
                   successful=len(good), failed_source_ids=[r["source_id"] for r in rows if r["failed_all50"]],
                   coverage_percent_over_completed=100*len(good)/len(rows) if rows else None,
                   knn_training_accuracy_percent=None, knn_high_confidence_accuracy_percent=None,
                   candidate_counts=dict(sum((Counter(r["candidate_counts"]) for r in rows), Counter())),
                   knn_micro=combine_metrics([r["accuracy"] for r in good]),
                   knn_macro_accuracy_percent=mean_defined([r["accuracy"]["accuracy_percent"] for r in good]),
                   knn_macro_changed_accuracy_percent=mean_defined([r["accuracy"]["changed_accuracy_percent"] for r in good]),
                   clean_timing_count=sum(r["actual8_seconds"] is not None for r in rows),
                   actual8_mean_seconds=mean_defined([r["actual8_seconds"] for r in rows]),
                   ideal50_mean_seconds=mean_defined([r["ideal50_seconds"] for r in rows]),
                   reference_mean_seconds=mean_defined([r["reference_seconds"] for r in rows]),
                   cost_pair_count=sum(r["cost_deviation_percent"] is not None for r in rows),
                   cost_deviation_mean_percent=mean_defined([r["cost_deviation_percent"] for r in rows]))
        neural = read_json(output / "inputs" / name / "neural_metrics.json")
        row["neural"] = {kind: dict(all_test=combine_metrics(values),
            knn_success_subset=combine_metrics([values[r["index"]] for r in good])) for kind, values in neural.items()}
        # This includes pending numerical tests as well as completed ones.
        attempts = [read_json(p) for p in (output / "results" / name).glob("*/*/attempt_*/result.json")]
        attempts += [read_json(p) for p in (output / "results" / name).glob("*/*/numeric_resolution/result.json")]
        row["all_attempt_counts_including_original_errors"] = dict(Counter(r["status"] for r in attempts))
        row["all_attempt_worker_seconds"] = sum(r["elapsed_seconds"] for r in attempts)
        unresolved = [read_json(p)["ranks"] for p in (output / "results" / name).glob("*/deferred.json")
                      if not (p.parent / "comparison.json").exists()]
        row["deferred_tests"] = len(unresolved)
        row["deferred_candidate_keys"] = sum(sum(k < 50 for k in ranks) for ranks in unresolved)
        row["deferred_reference_keys"] = sum(50 in ranks for ranks in unresolved)
        summary[name] = row
    write_json(output / "summary.json", summary, replace=True)
    return summary


def run(args):
    config, base = resolve_config(args.config)
    output = Path(args.output).resolve()
    if (output / "protocol.json").exists() and not args.resume:
        raise ValueError("Existing run: pass --resume; completed results will not be recomputed")
    cases = config["cases"]
    if not cases:
        raise ValueError("At least one case is required")
    solver = Path(__file__).resolve().parents[1] / "unit_commitment_gurobi.py"
    files = [solver, Path(__file__), Path(__file__).with_name("comparison_solver.py"),
             Path(__file__).with_name("comparison_common.py"), solver.with_name("ml_module_define.py")]
    for spec in cases.values():
        files.extend(base / spec[key] for key in ("data", "network", "storage"))
        files.extend(base / item["path"] for item in spec.get("models", {}).values())
    with run_lock(output):
        bind_run(output, config, files)
        disk_guard(output)
        try:
            with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn")) as pool:
                # All case smoke tests run before any remaining test. Smoke is part of 100.
                phases = [range(1)] if args.smoke else [range(1), range(1, 100)]
                for indices in phases:
                    for name, spec in cases.items():
                        destination = output / "inputs" / name
                        x, y, average, network = prepare_case(spec, base, destination)
                        if not (destination / "neural_metrics.json").exists():
                            write_json(destination / "neural_metrics.json", neural_predictions(spec, base, destination, x, y))
                            write_json(destination / "neural_audit.json", dict(
                                metrics_sha256=sha256(destination / "neural_metrics.json"),
                                probabilities={p.name: sha256(p) for p in destination.glob("*_probabilities.npz")}))
                        neural_audit = read_json(destination / "neural_audit.json")
                        if sha256(destination / "neural_metrics.json") != neural_audit["metrics_sha256"]:
                            raise ValueError("Neural metrics changed")
                        for filename, expected in neural_audit["probabilities"].items():
                            if sha256(destination / filename) != expected:
                                raise ValueError("Neural probability cache changed")
                        common = dict(network=str(network), storage=str((base / spec["storage"]).resolve()),
                                      solver=str(solver), mip_gap=1e-4)
                        for index in indices:
                            disk_guard(output)
                            status(output, "first_pass", case=name, test_index=index, workers=8)
                            row = compare_test(pool, output, name, index, x, y, average, common)
                            if index == 0 and row is None:
                                raise RuntimeError("Smoke test deferred; inspect before continuing production")
                        summarize(output, [n for n in cases if (output / "inputs" / n / "neural_metrics.json").exists()])
                if args.resolve_numeric and not args.smoke:
                    for name, spec in cases.items():
                        _, y, _, _ = prepare_case(spec, base, output / "inputs" / name)
                        for path in sorted((output / "results" / name).glob("*/deferred.json")):
                            index = int(path.parent.name.split("_")[1])
                            status(output, "numeric_resolution", case=name, test_index=index)
                            resolve_deferred(pool, output, name, index, y)
            summary = summarize(output, cases)
            complete = all(row["completed"] == 100 for row in summary.values())
            status(output, "complete" if complete else "smoke_complete" if args.smoke else "pending_numeric", summary=summary)
            if complete and not (output / "final_audit.json").exists():
                write_json(output / "final_audit.json", dict(passed=True, tests=100*len(cases),
                    candidates=5000*len(cases), references=100*len(cases),
                    scope="per-result solver checks, exact keys, selection and saved metrics",
                    summary_sha256=sha256(output / "summary.json")))
        except Exception as exc:
            status(output, "failed", error=str(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="fig_data/knn50")
    parser.add_argument("--smoke", action="store_true", help="One complete 50-candidate + reference test per case")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resolve-numeric", action="store_true", help="One same-model robust retry after all first-pass tests")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
