"""Four-way transfer comparison on frozen 900/100/100 scenario cohorts.

Train, evaluate and plot using source-frozen aggregation and normalization.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import random
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from .comparison_common import (bind_run, disk_guard, fixed_split, mean_defined,
    metrics, predict, read_json, resolve_config, run_lock, save_npz, sha256,
    status, trusted_model, write_json)

CELLS = tuple(f"a_{a}_b_{b}" for a in ("minus", "zero", "plus") for b in ("minus", "zero", "plus"))
SERIES = ("frozen_raw", "frozen_aggregated", "retrained_raw", "retrained_aggregated")
LABELS = ("Frozen raw", "Frozen aggregated", "Retrained raw (900)", "Retrained aggregated (900)")
RECIPE = dict(seed=42, epochs=50000, warmup=5000, learning_rate=.0002,
              weight_decay=.001, hidden_dim=128, batch="full900", precision="FP32",
              validation_interval=50, patience=50, patience_min_delta=1e-5,
              clip_norm=1., minimum_lr_ratio=.01, optimizer="Adam")


def legacy_projection(raw, matrix):
    # Preserve the source checkpoint's historical axis layout, not raw @ matrix.
    samples, periods, buses = raw.shape
    flat = raw.transpose(0, 2, 1).reshape(samples * periods, buses)
    projected = flat @ matrix
    residual = flat.sum(1, keepdims=True) - projected.sum(1, keepdims=True)
    return np.concatenate([projected, residual], axis=1).reshape(samples, periods, -1)


def representations(mw, average, preprocessing):
    raw = np.asarray(mw, dtype=np.float64) / average
    aggregate = (legacy_projection(raw, preprocessing["matrix"]) - preprocessing["mean"]) / preprocessing["std"]
    if not np.isfinite(aggregate).all():
        raise ValueError("Invalid frozen preprocessing")
    return {"raw": raw.astype(np.float32), "aggregated": aggregate.astype(np.float32)}


def lr_factor(step):
    if step < 5000:
        return step / 5000
    progress = min(1., max(0., (step - 5000) / 45000))
    return .01 + .99 * .5 * (1 + math.cos(math.pi * progress))


def patience_update(loss, best, reference, count, minimum_delta=1e-5):
    improved_best = loss < best
    best = min(best, loss)
    if loss < reference - minimum_delta:
        return best, loss, 0, improved_best
    return best, reference, count + 1, improved_best


def validate_cohort(data, factor, cell):
    if factor < 0 or factor > 8 or (factor == 0 and cell != "a_zero_b_zero") or cell not in CELLS:
        raise ValueError("Unsupported factor/cell")
    split = fixed_split()
    dimensions = None
    for key, count in (("train", 900), ("val", 100), ("test", 100)):
        x, y, ids = data[key + "_input_MW"], data[key + "_labels"], data[key + "_source_ids"]
        if x.ndim != 3 or x.shape[:2] != (count, 36) or y.ndim != 3 or y.shape[:2] != (count, 36):
            raise ValueError("Each cell requires exactly 900/100/100 accepted 36-period samples")
        if not np.isfinite(x).all() or not np.isin(y, (0, 1)).all():
            raise ValueError("Invalid inputs or binary labels")
        if ids.shape != (count,) or not np.issubdtype(ids.dtype, np.integer) or not set(ids.tolist()) <= set(split[key]):
            raise ValueError(f"Source ID outside frozen {key} split")
        shape = (x.shape[2], y.shape[2])
        if dimensions is not None and dimensions != shape:
            raise ValueError("Split dimensions differ")
        dimensions = shape
    return dimensions


def validate_selection(records, data):
    """Check contiguous attempt sequences, first acceptance and full failure denominator."""
    counts = {}
    seen = set()
    allowed = {"legacy_accepted", "no_answer", "infeasible"}
    for split, count in (("train", 900), ("val", 100), ("test", 100)):
        group = [r for r in records if r["split"] == split]
        counts[split] = {key: sum(r["status"] == key for r in group) for key in allowed}
        for slot in range(count):
            sequence = sorted((r for r in group if r["slot"] == slot), key=lambda r: r["attempt"])
            if not sequence or [r["attempt"] for r in sequence] != list(range(len(sequence))):
                raise ValueError("Missing or duplicated attempt in first-accept sequence")
            if sequence[-1]["status"] != "legacy_accepted" or any(r["status"] not in ("no_answer", "infeasible") for r in sequence[:-1]):
                raise ValueError("Slot must end at its first accepted attempt")
            if sequence[-1]["source_id"] != int(data[split + "_source_ids"][slot]):
                raise ValueError("Accepted source does not match cohort")
            for row in sequence:
                if row["source_id"] not in fixed_split()[split] or row["status"] not in allowed:
                    raise ValueError("Invalid split or terminal status")
                seen.add((split, slot, row["attempt"]))
    if len(seen) != len(records):
        raise ValueError("Unexpected attempt keys")
    return counts


def train_model(folder, inputs, labels, val_inputs, val_labels):
    import torch
    from ml_module_define import TransformerModel
    if (folder / "training_result.json").exists():
        result = read_json(folder / "training_result.json")
        if not result["passed"] or sha256(folder / "model.pth") != result["weight_sha256"]:
            raise ValueError("Saved trained model changed")
        return trusted_model(folder / "model.pth")
    if folder.exists():
        raise RuntimeError(f"Partial training retained at {folder}; use a new run, not an implicit warm start")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for training; no implicit CPU fallback")
    folder.mkdir(parents=True)
    torch.set_num_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = TransformerModel(input_size=inputs.shape[-1], output_size=labels.shape[-1], hidden_dim=128).cuda()
    x, y, xv, yv = [torch.as_tensor(a, dtype=torch.float32, device="cuda") for a in (inputs, labels, val_inputs, val_labels)]
    optimizer = torch.optim.Adam(model.parameters(), lr=.0002, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    criterion = torch.nn.BCELoss()
    best = reference = float("inf")
    counter, best_epoch, history, best_state = 0, 0, [], None
    start = time.perf_counter()
    write_json(folder / "training_manifest.json", dict(recipe=RECIPE, from_scratch=True,
        input_size=inputs.shape[-1], output_size=labels.shape[-1], test_excluded=True,
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name()))
    with (folder / "training.log").open("x", encoding="utf-8") as log:
        for epoch in range(1, 50001):
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if epoch % 50:
                continue
            model.eval()
            with torch.no_grad():
                train_loss = float(criterion(model(x), y).item())
                val_loss = float(criterion(model(xv), yv).item())
            if not np.isfinite([train_loss, val_loss]).all():
                raise ValueError("Nonfinite validation loss")
            best, reference, counter, improved = patience_update(val_loss, best, reference, counter)
            if improved:
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                torch.save(best_state, folder / "checkpoint.best.pt")
            record = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                          lr=optimizer.param_groups[0]["lr"], patience=counter)
            history.append(record)
            print(record, file=log, flush=True)
            if counter >= 50:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        restored = float(criterion(model(xv), yv).item())
    if abs(restored - best) > 1e-6:
        raise ValueError("Best validation checkpoint was not restored")
    torch.save(model.cpu(), folder / "model.pth")
    write_json(folder / "training_result.json", dict(passed=True, best_epoch=best_epoch,
        best_validation_loss=best, restored_validation_loss=restored, history=history,
        elapsed_seconds=time.perf_counter()-start, weight_sha256=sha256(folder / "model.pth")))
    del x, y, xv, yv, optimizer
    torch.cuda.empty_cache()
    return model


def cell_comparison(config, spec, base, output, preprocessing, train):
    factor, cell = float(spec["factor"]), spec["cell"]
    folder = output / "cells" / f"amplitude_{factor:g}" / cell
    with np.load(base / spec["data"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    validate_cohort(data, factor, cell)
    counts = validate_selection(read_json(base / spec["attempts"]), data)
    if (folder / "evaluation_audit.json").exists():
        audit = read_json(folder / "evaluation_audit.json")
        if not audit["passed"] or sha256(folder / "comparison.json") != audit["comparison_sha256"]:
            raise ValueError("Completed comparison changed")
        for filename, expected in audit["probabilities"].items():
            if sha256(folder / filename) != expected:
                raise ValueError("Completed probability changed")
        rows = read_json(folder / "comparison.json")
        for stage in SERIES:
            with np.load(folder / (stage+"_probabilities.npz"), allow_pickle=False) as archive:
                if not np.array_equal(archive["source_ids"], data["test_source_ids"]):
                    raise ValueError("Saved test order changed")
                for row in [r for r in rows if r["stage"] == stage]:
                    mask = {"all": np.ones(100, bool), "history": data["test_source_ids"] < 365,
                            "uniform_augmentation": data["test_source_ids"] >= 365}[row["subset"]]
                    if metrics(archive["probabilities"][mask], data["test_labels"][mask]) != row["metrics"]:
                        raise ValueError("Saved metrics disagree with saved probabilities")
        return rows
    reps = {split: representations(data[split + "_input_MW"], config["average_load"], preprocessing)
            for split in ("train", "val", "test")}
    if not (folder / "ready_audit.json").exists():
        write_json(folder / "ready_audit.json", dict(passed=True, counts=counts,
            cohort_sha256=sha256(base / spec["data"]), attempts_sha256=sha256(base / spec["attempts"]),
            scope="bundle dimensions, split IDs, terminal sequences; upstream physical label checks required"))
    rows, bindings = [], {}
    for stage in SERIES:
        kind = "aggregated" if stage.endswith("aggregated") else "raw"
        path = folder / (stage + "_probabilities.npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                probabilities = z["probabilities"]
                if not np.array_equal(z["source_ids"], data["test_source_ids"]):
                    raise ValueError("Test source binding changed")
        else:
            if stage.startswith("frozen"):
                model = trusted_model(base / config[stage])
            elif stage in spec:
                model = trusted_model(base / spec[stage])
            elif train:
                model = train_model(folder / "models" / kind, reps["train"][kind], data["train_labels"],
                                    reps["val"][kind], data["val_labels"])
            else:
                model_folder = folder / "models" / kind
                if not (model_folder / "training_result.json").exists():
                    raise ValueError(f"No retrained checkpoint for {factor}/{cell}/{kind}; supply its path or use train-evaluate")
                result = read_json(model_folder / "training_result.json")
                if sha256(model_folder / "model.pth") != result["weight_sha256"]:
                    raise ValueError("Trained checkpoint changed")
                model = trusted_model(model_folder / "model.pth")
            probabilities = predict(model, reps["test"][kind])
            del model
            metrics(probabilities, data["test_labels"])
            save_npz(path, probabilities=probabilities, source_ids=data["test_source_ids"])
        bindings[path.name] = sha256(path)
        for subset, mask in (("all", np.ones(100, dtype=bool)),
                             ("history", data["test_source_ids"] < 365),
                             ("uniform_augmentation", data["test_source_ids"] >= 365)):
            rows.append(dict(factor=factor, mean_shift_magnitude_percent=5*factor, cell=cell,
                stage=stage, subset=subset, samples=int(mask.sum()),
                metrics=metrics(probabilities[mask], data["test_labels"][mask])))
    write_json(folder / "comparison.json", rows)
    write_json(folder / "evaluation_audit.json", dict(passed=True, models=4,
        comparison_sha256=sha256(folder / "comparison.json"), probabilities=bindings))
    return rows


def plot_results(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = [r for r in rows if r["subset"] == "all"]
    factors = sorted({r["factor"] for r in values})
    summary = []
    for factor in factors:
        for stage in SERIES:
            selected = [r for r in values if r["factor"] == factor and r["stage"] == stage]
            expected = {"a_zero_b_zero"} if factor == 0 else set(CELLS)
            if len(selected) != len(expected) or {r["cell"] for r in selected} != expected:
                raise ValueError(f"Missing/duplicate cells at {factor}/{stage}; no interpolation")
            summary.append(dict(factor=factor, stage=stage, cells=len(selected),
                **{key: mean_defined([r["metrics"][key] for r in selected])
                   for key in ("accuracy_percent", "changed_accuracy_percent")}))
    write_json(output / "summary.json", summary, replace=True)
    for metric, label in (("accuracy_percent", "Overall accuracy (%)"),
                          ("changed_accuracy_percent", "Changed-generator accuracy (%)")):
        for ceiling, suffix in ((None, "all"), (4, "through4")):
            xs = [f for f in factors if ceiling is None or f <= ceiling]
            if not xs:
                continue
            fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
            for stage, title, color, marker in zip(SERIES, LABELS,
                    ("#276FBF", "#C44536", "#37834A", "#8856A7"), ("o", "s", "^", "D")):
                ys = [next(r[metric] for r in summary if r["factor"] == f and r["stage"] == stage) for f in xs]
                ax.plot(np.array(xs)*5, [np.nan if v is None else v for v in ys], marker=marker, label=title, color=color)
            ax.set(xlabel="System total-load mean-shift magnitude (%)", ylabel=label)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
            fig.savefig(output / f"{metric}_{suffix}.png", dpi=220)
            fig.savefig(output / f"{metric}_{suffix}.pdf")
            plt.close(fig)
        nonzero = [f for f in factors if f > 0]
        if not nonzero:
            continue
        matrices = []
        for factor in nonzero:
            matrix = []
            for cell in CELLS:
                def accuracy(stage):
                    return next(r["metrics"][metric] for r in values if r["factor"] == factor and r["cell"] == cell and r["stage"] == stage)
                a, b = accuracy("frozen_aggregated"), accuracy("frozen_raw")
                matrix.append(a-b if a is not None and b is not None else np.nan)
            matrices.append(np.array(matrix).reshape(3, 3))
        finite = np.concatenate([m.ravel() for m in matrices])
        finite = finite[np.isfinite(finite)]
        limit = max(float(np.max(abs(finite))) if len(finite) else 0, .01)
        cols = min(4, len(nonzero))
        fig, axes = plt.subplots(math.ceil(len(nonzero)/cols), cols, squeeze=False,
                                 figsize=(2.7*cols, 2.6*math.ceil(len(nonzero)/cols)), constrained_layout=True)
        for ax, factor, matrix in zip(axes.flat, nonzero, matrices):
            im = ax.imshow(matrix, vmin=-limit, vmax=limit, cmap="RdBu")
            ax.set(xlabel="Zero-node mean shift", ylabel="System mean shift",
                   xticks=[0, 1, 2], xticklabels=["-", "0", "+"], yticks=[0, 1, 2], yticklabels=["-", "0", "+"])
            ax.set_title(f"Mean shift: {5*factor:g}%", fontsize=10)
            ax.xaxis.label.set_size(9)
            ax.yaxis.label.set_size(9)
            ax.tick_params(labelsize=9)
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, f"{value:+.2f}" if np.isfinite(value) else "N/A", ha="center", va="center",
                        color="white" if np.isfinite(value) and abs(value) > .6*limit else "black", fontsize=9)
        for ax in list(axes.flat)[len(nonzero):]:
            ax.set_visible(False)
        fig.colorbar(im, ax=list(axes.flat), label="Frozen aggregated - frozen raw (percentage points)", shrink=.8)
        fig.savefig(output / f"{metric}_frozen_gain_heatmaps.png", dpi=220)
        fig.savefig(output / f"{metric}_frozen_gain_heatmaps.pdf")
        plt.close(fig)


def run(args):
    config, base = resolve_config(args.config)
    if "source_data" in config:
        with np.load(base / config["source_data"], allow_pickle=False) as source:
            config["average_load"] = float(source["average_load"])
    if "average_load" not in config:
        raise ValueError("Supply source_data with its original average_load")
    if "cohorts" not in config:
        config["cohorts"] = [dict(factor=f, cell=c,
            data=f"{config['cohort_root']}/amplitude_{f:g}/{c}/cohort.npz",
            attempts=f"{config['cohort_root']}/amplitude_{f:g}/{c}/attempts.json")
            for f in config["factors"] for c in (("a_zero_b_zero",) if f == 0 else CELLS)]
    output = Path(args.output).resolve()
    keys = [(float(s["factor"]), s["cell"]) for s in config["cohorts"]]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate factor/cell")
    if config["average_load"] <= 0 or not np.isfinite(config["average_load"]):
        raise ValueError("Invalid source average_load")
    files = [Path(__file__), Path(__file__).with_name("comparison_common.py"),
             Path(__file__).resolve().parents[1] / "ml_module_define.py"]
    files += [base / config[k] for k in ("preprocessing", "frozen_raw", "frozen_aggregated")]
    if "source_data" in config:
        files.append(base / config["source_data"])
    for spec in config["cohorts"]:
        files += [base / spec[k] for k in ("data", "attempts", "retrained_raw", "retrained_aggregated") if k in spec]
    if (output / "protocol.json").exists() and not args.resume and args.command != "plot":
        raise ValueError("Existing run; use --resume to reuse completed models and comparisons")
    with run_lock(output):
        bind_run(output, dict(config=config, recipe=RECIPE), files)
        with np.load(base / config["preprocessing"], allow_pickle=False) as archive:
            preprocessing = {k: archive[k] for k in ("matrix", "mean", "std")}
        if np.any(preprocessing["std"] <= 0):
            raise ValueError("Frozen source standard deviations must be positive")
        rows = []
        try:
            for spec in config["cohorts"]:
                disk_guard(output)
                status(output, args.command, factor=spec["factor"], cell=spec["cell"])
                if args.command == "plot":
                    path = output / "cells" / f"amplitude_{float(spec['factor']):g}" / spec["cell"] / "evaluation_audit.json"
                    if not path.exists():
                        raise ValueError("Plot requires completed cell comparisons")
                rows.extend(cell_comparison(config, spec, base, output, preprocessing, args.command == "train-evaluate"))
            write_json(output / "comparisons.json", rows, replace=True)
            plot_results(rows, output)
            write_json(output / "final_audit.json", dict(passed=True, cells=len(keys),
                model_comparisons=4*len(keys), test_samples=100*len(keys),
                scope="frozen input bindings, cohort sequences, saved probabilities and metric recomputation",
                summary_sha256=sha256(output / "summary.json")), replace=True)
            status(output, "complete", cells=len(keys))
        except Exception as exc:
            status(output, "failed", error=str(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("evaluate", "train-evaluate", "plot"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="fig_data/transfer_scenarios")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
