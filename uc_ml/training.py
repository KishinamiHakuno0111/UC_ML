"""Configurable training entrypoint built on the legacy ML utilities."""

from __future__ import annotations

import ast
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    """Training parameters exposed by the public CLI."""

    data: str
    data_dir: str = "training_data"
    model_type: str = "transformer"
    target: str = "units"
    training_num: int = 1000
    test_num: int = 100
    val_ratio: float = 0.1
    random_seed: int = 42
    data_is_random: bool = True
    epochs: int = 50000
    stop_gap: int = 50
    learning_rate: float = 0.0002
    weight_decay: float = 0.001
    hidden_sizes: str | int | list[int] | None = None
    dropout_prob: float = 0.3
    num_pieces: int = 2
    is_maxout: bool = False
    use_scheduler: bool = True
    warmup_ratio: float = 0.1
    gradient_clip_val: float = 1.0
    accumulate_steps: int = 1
    early_stopping_patience: int = 50
    early_stopping_metric: str = "val_loss"
    reliability_low_gate: float = 0.1
    reliability_high_gate: float = 0.9
    output_dir: str = "module_save"
    fig_dir: str = "fig_data"
    result_xlsx: str = "test_result.xlsx"
    cutted_by_time: bool | None = None


def _parse_hidden_sizes(value: str | int | list[int] | None, model_type: str) -> Any:
    if value is None:
        if model_type in ("DNN", "KAN"):
            return [256, 256, 256]
        if model_type == "LSTM_new":
            return [64 for _ in range(36)]
        return 128

    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return [int(v) for v in value]

    text = str(value).strip()
    if model_type in ("DNN", "KAN", "LSTM_new"):
        if text.startswith("["):
            parsed = ast.literal_eval(text)
            return [int(v) for v in parsed]
        return [int(part.strip()) for part in text.split(",") if part.strip()]
    return int(text)


def _infer_cutted_by_time(model_type: str, target: str, override: bool | None) -> bool:
    if override is not None:
        return bool(override)
    if target == "constraints":
        return False
    return model_type == "DNN"


def _resolve_data_path(config: TrainConfig) -> str:
    path = Path(config.data)
    if path.is_absolute() or path.parent != Path("."):
        return os.fspath(path)
    return os.fspath(Path(config.data_dir) / path)


def train_from_config(config: TrainConfig) -> str:
    """Train one model and return the saved model stem."""
    from ml_basic_functions import get_data, train_model

    if config.model_type not in ("DNN", "KAN", "transformer", "LSTM_new", "BiLSTM"):
        raise ValueError(f"Unsupported model_type: {config.model_type}")
    if config.target not in ("units", "constraints"):
        raise ValueError("target must be 'units' or 'constraints'.")
    if config.reliability_low_gate >= config.reliability_high_gate:
        raise ValueError("reliability_low_gate must be smaller than reliability_high_gate.")

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.fig_dir).mkdir(parents=True, exist_ok=True)

    hidden_sizes = _parse_hidden_sizes(config.hidden_sizes, config.model_type)
    recognize_constraints = config.target == "constraints"
    cutted_by_time = _infer_cutted_by_time(config.model_type, config.target, config.cutted_by_time)
    shuffle_str = "random" if config.data_is_random else "not_random"
    data_name = Path(config.data).name
    module_stem = (
        f"{config.target}_{config.model_type}_{data_name}_{shuffle_str}"
        f"_{config.training_num}_{config.epochs}_{hidden_sizes}_{config.learning_rate}"
    )
    module_path = os.fspath(Path(config.output_dir) / module_stem)
    data_path = _resolve_data_path(config)

    print("Loading data...")
    (
        train_inputs,
        train_targets,
        val_inputs,
        val_targets,
        test_inputs,
        test_targets,
        time_num,
        input_size,
        output_size,
        average_load,
        train_size,
    ) = get_data(
        config.model_type,
        data_path,
        test_num=config.test_num,
        training_num=config.training_num,
        data_is_random=config.data_is_random,
        cutted_by_time=cutted_by_time,
        recognize_constr=recognize_constraints,
        random_seed=config.random_seed,
        val_ratio=config.val_ratio,
    )

    metadata = asdict(config)
    metadata.update(
        {
            "model_type": config.model_type,
            "data_name": data_name,
            "data_type": config.target,
            "cutted_by_time": cutted_by_time,
            "hidden_sizes_resolved": hidden_sizes,
            "time_num": int(time_num),
            "input_size": int(input_size),
            "output_size": int(output_size),
            "average_load": float(average_load),
            "train_size": int(train_size),
        }
    )
    with open(module_path + "_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"Training {config.target} model: type={config.model_type}, "
        f"data={data_path}, input_size={input_size}, output_size={output_size}, "
        f"train/val/test={len(train_inputs)}/{len(val_inputs)}/{len(test_inputs)}"
    )

    start = time.time()
    model, loss_record, val_loss_record, plt_result, _test_result = train_model(
        config.model_type,
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
        input_size,
        output_size,
        module_path,
        hidden_sizes=hidden_sizes,
        learning_rate=config.learning_rate,
        num_epochs=config.epochs,
        dropout_prob=config.dropout_prob,
        num_pieces=config.num_pieces,
        is_maxout=config.is_maxout,
        stop_gap=config.stop_gap,
        weight_decay=config.weight_decay,
        recognize_constr=recognize_constraints,
        val_inputs=val_inputs,
        val_targets=val_targets,
        use_scheduler=config.use_scheduler,
        warmup_ratio=config.warmup_ratio,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_steps=config.accumulate_steps,
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_metric=config.early_stopping_metric,
        reliability_low_gate=config.reliability_low_gate,
        reliability_high_gate=config.reliability_high_gate,
        save_name=config.result_xlsx,
    )

    curve_path = module_path + "_curve.jpeg"
    if plt_result is not None:
        plt_result.savefig(curve_path, format="jpeg", dpi=150)
        try:
            plt_result.close("all")
        except AttributeError:
            pass

    elapsed = time.time() - start
    del model, loss_record, val_loss_record
    print(f"Training finished in {elapsed:.1f}s")
    print(f"Saved model stem: {module_path}")
    return module_path
