"""Small, shared utilities for the opt-in comparison entry points."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def fixed_split():
    ids = list(range(1100))
    random.Random(42).shuffle(ids)
    return {"train": ids[:900], "val": ids[900:1000], "test": ids[1000:]}


def metrics(probability, truth):
    """Micro accuracy; changed generators include ALL periods, not just transitions."""
    probability, truth = np.asarray(probability), np.asarray(truth)
    if probability.shape != truth.shape or truth.ndim not in (2, 3):
        raise ValueError("Expected matching (time, generator) or (sample, time, generator)")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid probabilities")
    if not np.isin(truth, (0, 1)).all():
        raise ValueError("Labels must be binary")
    if truth.ndim == 2:
        probability, truth = probability[None], truth[None]
    correct = (probability >= .5) == truth
    changed = np.any(truth[:, 1:] != truth[:, :-1], axis=1)
    mask = np.broadcast_to(changed[:, None, :], truth.shape)
    total, changed_total = int(truth.size), int(mask.sum())
    count, changed_count = int(correct.sum()), int(correct[mask].sum())
    return dict(correct=count, entries=total, changed_correct=changed_count,
                changed_entries=changed_total,
                accuracy_percent=100 * count / total if total else None,
                changed_accuracy_percent=100 * changed_count / changed_total if changed_total else None)


def combine_metrics(rows):
    sums = {key: sum(row[key] for row in rows)
            for key in ("correct", "entries", "changed_correct", "changed_entries")}
    for prefix in ("", "changed_"):
        denominator = sums[prefix + "entries"]
        sums[prefix + "accuracy_percent"] = (
            100 * sums[prefix + "correct"] / denominator if denominator else None)
    return sums


def mean_defined(values):
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def disk_guard(path, minimum_gib=8):
    if shutil.disk_usage(path).free < minimum_gib * 1024**3:
        raise RuntimeError(f"Less than {minimum_gib} GiB free; existing files retained")


def status(output, stage, **details):
    value = dict(stage=stage, pid=os.getpid(), unix=time.time(), **details)
    write_json(Path(output) / "status.json", value, replace=True)
    print(json.dumps(value), flush=True)


@contextlib.contextmanager
def run_lock(output):
    """OS lock is released on process exit, including a crash; never delete another lock."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def bind_run(output, config, files):
    """Refuse silent mixing of settings, source data, or implementations on resume."""
    output = Path(output)
    binding = dict(config=config, files={str(Path(p).resolve()): sha256(p) for p in files})
    path = output / "protocol.json"
    if path.exists():
        if read_json(path) != binding:
            raise ValueError("Run inputs/settings changed; use a new output directory")
    else:
        write_json(path, binding)
    return binding


def resolve_config(path):
    """All user-provided paths are relative to the config file, not the working directory."""
    path = Path(path).resolve()
    return read_json(path), path.parent


def trusted_model(path):
    # Repository checkpoints contain Python model objects. Never load an untrusted file.
    import torch
    return torch.load(path, map_location="cpu", weights_only=False).eval()


def predict(model, inputs, dnn_layout=None):
    import torch
    model.eval()
    device = next(model.parameters()).device
    output = []
    with torch.inference_mode():
        for sample in inputs:
            value = torch.as_tensor(sample[None], dtype=torch.float32, device=device)
            if dnn_layout == "per_period":
                value = value.reshape(sample.shape)
            elif dnn_layout == "flattened":
                value = value.reshape(1, -1)
            result = model(value).detach().cpu().numpy()
            output.append(result.reshape(sample.shape[0], -1))
    return np.asarray(output)
