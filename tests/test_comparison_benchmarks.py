"""Fast comparison regressions; optional Gurobi tests use a tiny synthetic system."""

import contextlib
from concurrent.futures import Future, ProcessPoolExecutor
import copy
import io
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from uc_ml.comparison_common import (bind_run, combine_metrics, fixed_split, metrics,
    read_json, save_npz, sha256, write_json)
from uc_ml.comparison_solver import derive_start_stop, solve, strict_numeric
from uc_ml.knn50 import (attempt_folder, compare_test, execute_pending, nearest,
    resolve_deferred, select_candidate, timing)
from uc_ml.transfer_fields import field, layout, perturb, project, rng, system_field
from uc_ml.transfer_scenarios import (CELLS, SERIES, cell_comparison, legacy_projection, lr_factor,
    patience_update, plot_results, validate_cohort, validate_selection)

ROOT = Path(__file__).resolve().parents[1]


class ImmediatePool:
    def __init__(self):
        self.calls = []

    def submit(self, function, *args):
        self.calls.append(args)
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


class ComparisonTests(unittest.TestCase):
    def test_split_matches_archived_order(self):
        split = fixed_split()
        self.assertEqual([len(split[k]) for k in ("train", "val", "test")], [900, 100, 100])
        self.assertEqual(split["test"][0], 764)
        self.assertEqual(len(set(sum(split.values(), []))), 1100)

    def test_nearest_is_train_only_and_stable(self):
        x = np.zeros((60, 36, 2), dtype=np.float32)
        self.assertEqual([sid for _, sid in nearest(x, 59, list(reversed(range(55))))], list(range(50)))
        with self.assertRaises(ValueError):
            nearest(x, 3, list(range(55)))
        x[54] = 1
        x[59] = 1
        self.assertEqual(nearest(x, 59, list(range(55)))[0], (0., 54))

    def test_select_is_not_vote_or_accuracy(self):
        rows = [dict(status="no_answer", objective=-999),
                dict(status="legacy_accepted", objective=5, accuracy=1),
                dict(status="legacy_accepted", objective=3, accuracy=0),
                dict(status="legacy_accepted", objective=3, accuracy=1)]
        self.assertEqual(select_candidate(rows), 2)
        self.assertIsNone(select_candidate(rows[:1]))

    def test_changed_accuracy_uses_all_periods(self):
        y = np.zeros((1, 36, 2), dtype=np.uint8)
        y[0, 18:, 0] = 1
        result = metrics(np.zeros_like(y), y)
        self.assertEqual(result["entries"], 72)
        self.assertEqual(result["changed_entries"], 36)
        self.assertEqual(result["changed_accuracy_percent"], 50)
        self.assertEqual(result["accuracy_percent"], 75)
        self.assertIsNone(metrics(np.zeros_like(y), np.zeros_like(y))["changed_accuracy_percent"])
        self.assertIsNone(metrics(y[:0], y[:0])["accuracy_percent"])
        self.assertEqual(combine_metrics([result, result])["changed_entries"], 72)

    def test_binary_transitions(self):
        u = np.array([[0, 1, 1, 0], [1, 1, 0, 1]])
        y, z = derive_start_stop(u, [0, 1])
        np.testing.assert_array_equal(y-z, np.diff(np.column_stack([[0, 1], u]), axis=1))
        self.assertFalse(np.any(y*z))

    def test_timing_includes_failed_candidates(self):
        rows = [dict(elapsed_seconds=2, cpu_seconds=1, status="legacy_accepted"),
                dict(elapsed_seconds=20, cpu_seconds=19, status="no_answer")]
        result = timing(.2, 25, .1, rows, True)
        self.assertAlmostEqual(result["actual8_seconds"], 25.3)
        self.assertAlmostEqual(result["ideal50_seconds"], 20.3)
        self.assertEqual(result["candidate_cpu_seconds"], 20)
        invalid = timing(.2, 25, .1, rows, False)
        self.assertIsNone(invalid["actual8_seconds"])
        self.assertIsNone(invalid["ideal50_seconds"])

    def test_only_numeric_without_solution_is_deferred(self):
        numeric = dict(status="error", error_kind="solver_status", rounds=[dict(status=12, solutions=0)])
        self.assertTrue(strict_numeric(numeric))
        for bad in (dict(numeric, error_kind="hash_error"), dict(numeric, rounds=[]),
                    dict(numeric, rounds=[dict(status=9, solutions=0)]),
                    dict(numeric, rounds=[dict(status=12, solutions=1)])):
            self.assertFalse(strict_numeric(bad))

    def test_fatal_drain_does_not_submit_more(self):
        pool = ImmediatePool()
        with patch("uc_ml.knn50.solve", return_value=dict(status="error", error_kind="hash_error")):
            with self.assertRaises(RuntimeError):
                execute_pending(pool, [(i, {}) for i in range(50)])
        self.assertEqual(len(pool.calls), 8)

    def test_normal_scheduler_has_all_keys(self):
        pool = ImmediatePool()
        with patch("uc_ml.knn50.solve", return_value=dict(status="infeasible")):
            rows = execute_pending(pool, [(i, {}) for i in range(50)])
        self.assertEqual(set(rows), set(range(50)))

    def test_partial_attempt_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            old = parent / "attempt_000"
            old.mkdir()
            write_json(old / "evidence.json", {"interrupted": True})
            folder, result = attempt_folder(parent)
            self.assertIsNone(result)
            self.assertEqual(folder.name, "attempt_001")
            write_json(folder / "result.json", dict(status="infeasible"))
            folder2, result2 = attempt_folder(parent)
            self.assertEqual(folder2, folder)
            self.assertEqual(result2["status"], "infeasible")
            self.assertTrue((old / "evidence.json").exists())

    def test_completed_knn_test_not_recomputed(self):
        def fake_solver(task):
            folder = Path(task["folder"])
            folder.mkdir()
            row = dict(status="infeasible", elapsed_seconds=2., cpu_seconds=1.)
            write_json(folder / "task.json", task)
            write_json(folder / "result.json", row)
            return row
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            x = np.zeros((1100, 36, 2), dtype=np.float32)
            y = np.zeros((1100, 36, 1), dtype=np.uint8)
            pool = ImmediatePool()
            with patch("uc_ml.knn50.solve", side_effect=fake_solver):
                first = compare_test(pool, root, "tiny", 0, x, y, 1., {})
                second = compare_test(pool, root, "tiny", 0, x, y, 1., {})
            self.assertEqual(first, second)
            self.assertEqual(len(pool.calls), 51)
            self.assertTrue(first["failed_all50"])
            self.assertIsNone(first["accuracy"])
            self.assertEqual(first["candidate_counts"], {"infeasible": 50})

    def test_deferred_resolution_keeps_original_and_null_candidate_timing(self):
        def fake_solver(task):
            folder = Path(task["folder"])
            folder.mkdir()
            numeric = "candidate_02" in str(folder) and not task.get("robust")
            row = dict(status="error" if numeric else "infeasible", elapsed_seconds=2., cpu_seconds=1.,
                       error_kind="solver_status" if numeric else "exception",
                       rounds=[dict(status=12 if numeric else 3, solutions=0, mathematical_sha256="same")])
            write_json(folder / "task.json", task)
            write_json(folder / "result.json", row)
            return row
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            x = np.zeros((1100, 36, 2), dtype=np.float32)
            y = np.zeros((1100, 36, 1), dtype=np.uint8)
            pool = ImmediatePool()
            with patch("uc_ml.knn50.solve", side_effect=fake_solver):
                self.assertIsNone(compare_test(pool, root, "tiny", 0, x, y, 1., {}))
                original = root / "results/tiny/test_000/candidate_02/attempt_000/result.json"
                before = sha256(original)
                resolve_deferred(pool, root, "tiny", 0, y)
                resolve_deferred(pool, root, "tiny", 0, y)
            row = read_json(root / "results/tiny/test_000/comparison.json")
            self.assertEqual(len(pool.calls), 52)
            self.assertEqual(sha256(original), before)
            self.assertIsNone(row["actual8_seconds"])
            self.assertEqual(row["all_attempt_worker_seconds"], 104)
            self.assertTrue(row["failed_all50"])

    def test_binding_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bind_run(root, {"a": 1}, [])
            bind_run(root, {"a": 1}, [])
            with self.assertRaises(ValueError):
                bind_run(root, {"a": 2}, [])

    def test_schedule_and_true_best_separate_from_patience(self):
        for step, expected in ((0, 0), (5000, 1), (27500, .505), (50000, .01)):
            self.assertAlmostEqual(lr_factor(step), expected)
        self.assertTrue(all(lr_factor(i) >= lr_factor(i+1) for i in range(5000, 50000)))
        best, ref, count, changed = patience_update(.1-1e-6, .1, .1, 0)
        self.assertTrue(changed)
        self.assertEqual(count, 1)
        self.assertEqual(ref, .1)
        self.assertLess(best, ref)

    def test_historical_mapping_not_silently_replaced(self):
        raw = np.arange(2*36*3, dtype=float).reshape(2, 36, 3)
        matrix = np.array([[1.], [0.], [-1.]])
        actual = legacy_projection(raw, matrix)
        flat = raw.transpose(0, 2, 1).reshape(72, 3)
        np.testing.assert_array_equal(actual[:, :, 0], (flat @ matrix).reshape(2, 36))
        self.assertFalse(np.array_equal(actual[:, :, :1], raw @ matrix))

    def test_source_split_and_first_accept_guards(self):
        data, records = {}, []
        for split, ids in fixed_split().items():
            n = len(ids)
            data[split+"_input_MW"] = np.ones((n, 36, 2))
            data[split+"_labels"] = np.zeros((n, 36, 1), dtype=np.uint8)
            data[split+"_source_ids"] = np.asarray(ids)
            records.extend(dict(split=split, slot=i, attempt=0, source_id=sid, status="legacy_accepted") for i, sid in enumerate(ids))
        validate_cohort(data, .5, CELLS[0])
        self.assertEqual(validate_selection(records, data)["train"]["legacy_accepted"], 900)
        bad = copy.deepcopy(records)
        bad[0]["attempt"] = 1
        with self.assertRaises(ValueError):
            validate_selection(bad, data)
        data["train_source_ids"][0] = fixed_split()["test"][0]
        with self.assertRaises(ValueError):
            validate_cohort(data, .5, CELLS[0])

    def test_fields_zero_balance_and_shared_formula(self):
        base = np.ones((2, 36, 4), dtype=float)*10
        base[:, :, 2:] = 0
        active, zero = np.array([0, 1]), np.array([2, 3])
        weights = layout(42, 2)
        z = field(rng(42, 1), 2, [0, 1])
        s = system_field(rng(42, 2), 2)
        q = .05*field(rng(42, 3), 2, [0, 1])
        eta = project(q, base[:, :, active])
        zero_result = perturb(base, 0, 1, -1, active, zero, weights, z, s, eta)
        np.testing.assert_array_equal(zero_result, base)
        one = perturb(base, 1, 1, -1, active, zero, weights, z, s, eta)
        two = perturb(base, 2.5, 1, -1, active, zero, weights, z, s, eta)
        np.testing.assert_allclose(two, base+2.5*(one-base), atol=1e-12)
        np.testing.assert_allclose(two.sum(-1), base.sum(-1)*(1+.05*2.5+.01*2.5*s), atol=1e-12)
        np.testing.assert_array_equal(z, field(rng(42, 1), 2, [0, 1]))

    def test_plot_requires_all_nine_cells(self):
        rows = []
        for cell in CELLS:
            for i, stage in enumerate(SERIES):
                rows.append(dict(factor=.5, cell=cell, stage=stage, subset="all",
                    metrics=dict(accuracy_percent=95+i, changed_accuracy_percent=85+i)))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            plot_results(rows, output)
            self.assertTrue((output / "accuracy_percent_all.png").exists())
            self.assertEqual(len(read_json(output / "summary.json")), 4)
            with self.assertRaises(ValueError):
                plot_results(rows[:-1], output)

    def test_transfer_evaluation_and_reuse_with_supplied_models(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Sigmoid())
            torch.save(model, root / "model.pth")
            arrays, records = {}, []
            for split, ids in fixed_split().items():
                arrays[split+"_input_MW"] = np.ones((len(ids), 36, 2))
                arrays[split+"_labels"] = np.zeros((len(ids), 36, 1), np.uint8)
                arrays[split+"_source_ids"] = np.asarray(ids)
                records.extend(dict(split=split, slot=i, attempt=0, source_id=sid, status="legacy_accepted") for i, sid in enumerate(ids))
            save_npz(root / "cohort.npz", **arrays)
            write_json(root / "attempts.json", records)
            spec = dict(factor=0, cell="a_zero_b_zero", data="cohort.npz", attempts="attempts.json",
                        retrained_raw="model.pth", retrained_aggregated="model.pth")
            config = dict(average_load=1., frozen_raw="model.pth", frozen_aggregated="model.pth")
            preprocessing = dict(matrix=np.array([[1.], [0.]]), mean=np.zeros(2), std=np.ones(2))
            rows = cell_comparison(config, spec, root, root / "output", preprocessing, False)
            self.assertEqual(len(rows), 12)
            with patch("uc_ml.transfer_scenarios.trusted_model", side_effect=RuntimeError("unexpected inference")):
                reused = cell_comparison(config, spec, root, root / "output", preprocessing, False)
            self.assertEqual(rows, reused)


def tiny_network(periods=4):
    return {"Parameters": {"Time horizon (h)": periods, "Time step (min)": 60},
        "Buses": {"b1": {"Load (MW)": [0]*periods}, "b2": {"Load (MW)": [10]*periods}},
        "Generators": {"g1": {"Bus": "b1", "Production cost curve (MW)": [0, 100],
            "Production cost curve ($)": [0, 100], "Initial status (h)": 10,
            "Initial power (MW)": 10, "Minimum uptime (h)": 1, "Minimum downtime (h)": 1,
            "Ramp up limit (MW)": 100, "Ramp down limit (MW)": 100,
            "Startup limit (MW)": 100, "Shutdown limit (MW)": 100}},
        "Transmission lines": {"l1": {"Source bus": "b1", "Target bus": "b2", "Susceptance (S)": 1,
            "Normal flow limit (MW)": 100}},
        "Reserves": {"r1": {"Amount (MW)": [0]*periods}}}


@unittest.skipUnless(os.environ.get("UC_ML_TEST_GUROBI") == "1", "Set UC_ML_TEST_GUROBI=1 for tiny licensed solves")
class SolverIntegrationTests(unittest.TestCase):
    def test_original_build_and_same_math_robust_solve(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "network.json", tiny_network())
            storage = {"max_power": [0], "min_power": [0], "initial_power": [0],
                       "storage_position": [0], "Kbat": [1], "Eta": [1], "p_ch_max": [0], "p_dis_max": [0]}
            save_npz(root / "storage.npz", **storage)
            save_npz(root / "input.npz", load_MW=np.tile([0., 10.], (4, 1)), u=np.ones((4, 1)))
            task = dict(network=str(root / "network.json"), storage=str(root / "storage.npz"),
                        solver=str(ROOT / "unit_commitment_gurobi.py"), input=str(root / "input.npz"), mip_gap=1e-4)
            with contextlib.redirect_stdout(io.StringIO()):
                plain = solve(dict(task, folder=str(root / "plain")))
                robust = solve(dict(task, folder=str(root / "robust"), robust=True))
            self.assertEqual(plain["status"], "legacy_accepted", plain)
            self.assertEqual(robust["status"], "legacy_accepted", robust)
            self.assertEqual(plain["rounds"][0]["mathematical_sha256"], robust["rounds"][0]["mathematical_sha256"])
            self.assertAlmostEqual(plain["objective"], robust["objective"], places=4)
            self.assertTrue(plain["audit"]["passed"])

    def test_eight_worker_spawn(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "network.json", tiny_network())
            storage = {"max_power": [0], "min_power": [0], "initial_power": [0],
                       "storage_position": [0], "Kbat": [1], "Eta": [1], "p_ch_max": [0], "p_dis_max": [0]}
            save_npz(root / "storage.npz", **storage)
            save_npz(root / "input.npz", load_MW=np.tile([0., 10.], (4, 1)), u=np.ones((4, 1)))
            task = dict(network=str(root / "network.json"), storage=str(root / "storage.npz"),
                        solver=str(ROOT / "unit_commitment_gurobi.py"), input=str(root / "input.npz"), mip_gap=1e-4)
            with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn")) as pool:
                rows = execute_pending(pool, [(i, dict(task, folder=str(root / f"candidate_{i}"))) for i in range(16)])
            self.assertEqual(set(rows), set(range(16)))
            self.assertTrue(all(row["status"] == "legacy_accepted" for row in rows.values()))
            self.assertTrue(all(row["worker_pid"] != os.getpid() for row in rows.values()))
            self.assertLessEqual(len({row["worker_pid"] for row in rows.values()}), 8)


if __name__ == "__main__":
    unittest.main()
