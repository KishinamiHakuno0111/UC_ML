"""
convert_mat_to_npz.py
====================
Batch-converts all .mat training files in training_data/ and metadata
.mat files in module_save/ to standard ML-friendly formats:

  - Training data (.mat)  →  .npz  (NumPy zipped)
  - Metadata   (.mat)      →  .json (sidecar alongside .pth)

Axis conventions for training data
---------------------------------
.mat stores:  (sample, bus,      time)   [original]
.npz stores:  (sample, time,      bus)    [PyTorch convention]
.mat stores:  (sample, generator, time)   [original]
.npz stores:  (sample, time,      gen)    [PyTorch convention]
.mat stores:  (sample, line,      time)   [original]
.npz stores:  (sample, time,      line)   [PyTorch convention]
enable_data is kept as (sample, line) — no transpose needed.

Run once; the old .mat files are NOT deleted automatically so you can
verify correctness before removing them manually.

Usage
-----
    python convert_mat_to_npz.py
"""

import os
import json
import scipy.io
import numpy as np
import shutil
from pathlib import Path

# ── Directories ──────────────────────────────────────────────────────────────
TRAIN_DIR   = Path("training_data")
MODULE_DIR  = Path("module_save")

# ── Fields we care about ────────────────────────────────────────────────────
TRAIN_FIELDS = ["input_data", "output_data", "enable_data", "l_data",
                "average_load", "active_lines"]


def _transpose_hwt(arr):
    """(samples, H, W) → (samples, W, H) — time always last in .mat, first in npz."""
    if arr.ndim == 3:
        return np.transpose(arr, (0, 2, 1))
    return arr


def convert_training_mat(mat_path: Path, npz_path: Path):
    """Load one .mat training file and save as .npz with correct axis order."""
    print(f"  converting {mat_path.name} …", end=" ", flush=True)
    try:
        d = scipy.io.loadmat(str(mat_path))
    except Exception as e:
        print(f"SKIP (load error: {e})")
        return False

    out = {}
    for key in TRAIN_FIELDS:
        if key in d:
            arr = d[key]
            # squeeze trivial dimensions that scipy sometimes adds
            while arr.ndim > 0 and arr.shape[0] == 1:
                arr = arr[0]
            # convert axis order for 3-D arrays
            if key in ("input_data", "output_data", "l_data"):
                arr = _transpose_hwt(arr)
            out[key] = arr.astype(np.float32) if arr.dtype.kind == 'f' else arr

    np.savez_compressed(str(npz_path), **out)

    # quick sanity check
    loaded = dict(np.load(str(npz_path), allow_pickle=True))
    expected_keys = {k for k in out if k in TRAIN_FIELDS}
    assert set(loaded.keys()) == expected_keys, (
        f"key mismatch: {loaded.keys()} vs {expected_keys}"
    )
    print(f"OK  {out['input_data'].shape if 'input_data' in out else '?'}")
    return True


def convert_metadata_mat(mat_path: Path, json_path: Path):
    """Extract scalar fields from a model-metadata .mat and save as .json."""
    print(f"  metadata {mat_path.name} …", end=" ", flush=True)
    try:
        d = scipy.io.loadmat(str(mat_path))
    except Exception as e:
        print(f"SKIP (load error: {e})")
        return False

    meta = {}
    for key in d:
        if key.startswith("__"):
            continue
        val = d[key]
        # flatten scipy's weird (1,1) → 0D, (N,) → list
        if isinstance(val, np.ndarray):
            if val.size == 1:
                val = val.item()
            elif val.ndim == 1:
                val = val.tolist()
            elif val.ndim == 2 and val.shape[0] == 1:
                val = val[0].tolist()
            elif val.ndim == 2 and val.shape[1] == 1:
                val = val[:, 0].tolist()
            else:
                # multi-dimensional arrays: flatten to list of lists with shape annotation
                val = {"_shape": list(val.shape), "_values": val.flatten().tolist()}
        meta[key] = val

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"OK  ({len(meta)} fields)")
    return True


def main():
    TRAIN_DIR.mkdir(exist_ok=True)

    print("\n=== Training data (.mat → .npz) ===")
    mat_files = sorted(TRAIN_DIR.glob("*.mat"))
    if not mat_files:
        print("  No .mat files found in training_data/")
    for mat_path in mat_files:
        # skip storage data (only has a few scalar fields — keep as .mat or convert separately)
        # Only convert files that look like training data (have input_data)
        try:
            keys = scipy.io.whosmat(str(mat_path))
            has_input = any(k[0] == "input_data" for k in keys)
        except Exception:
            has_input = False

        if not has_input:
            print(f"  skipping {mat_path.name} (not a training-data file)")
            continue

        # use original parent / name with .npz suffix so multi-dot names work
        npz_path = mat_path.with_suffix('.npz')
        convert_training_mat(mat_path, npz_path)

    print("\n=== Metadata (.mat → .json) in module_save/ ===")
    if MODULE_DIR.exists():
        mat_files_meta = sorted(MODULE_DIR.glob("*.mat"))
        if not mat_files_meta:
            print("  No .mat files found in module_save/")
        for mat_path in mat_files_meta:
            json_path = MODULE_DIR / (mat_path.stem + ".json")
            convert_metadata_mat(mat_path, json_path)
    else:
        print("  module_save/ not found, skipping")

    print("\nDone. Verify the .npz files in training_data/ before removing old .mat files.")
    print("  Files are saved alongside their .mat counterparts with .npz extension.")


if __name__ == "__main__":
    main()
