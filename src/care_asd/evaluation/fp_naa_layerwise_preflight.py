"""Bounded normal-only mechanism preflight for FP-NAA v8."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as functional
from torch import Tensor

from care_asd.data.fp_naa_augmentation_cache import _AugmentationPlan, _make_plans, _prepare
from care_asd.fp_naa_config import FPLayerwiseConfig, FPNAAConfig, load_fp_naa_config
from care_asd.models.beats_frontend import OfficialBEATsFrontend, fixed_duration_waveform
from care_asd.models.layerwise_noise_aware import (
    LayerwiseNoiseAwareEncoder,
    finite_adapter_update,
)


@dataclass(frozen=True)
class LayerwisePreflightResult:
    output_directory: Path
    summary_path: Path
    diagnostics_path: Path
    gate_path: Path
    gate_passed: bool


@dataclass(frozen=True)
class _Prepared:
    plan: _AugmentationPlan
    waveforms: np.ndarray
    positions: Mapping[str, int]


@dataclass(frozen=True)
class _Arrays:
    file_ids: tuple[str, ...]
    families: tuple[str, ...]
    noisy: np.ndarray
    reference: np.ndarray
    fault_noisy: np.ndarray
    teacher_clean: np.ndarray
    teacher_fault: np.ndarray

    def __len__(self) -> int:
        return len(self.file_ids)


def run_layerwise_mechanism_preflight(
    *,
    base_cache_directory: str | Path,
    audio_root: str | Path,
    cache_directory: str | Path,
    output_directory: str | Path,
    checkpoint_directory: str | Path,
    config_path: str | Path,
    beats_source_directory: str | Path,
    checkpoint_path: str | Path,
    workers: int = 12,
    device: str = "cuda",
) -> LayerwisePreflightResult:
    """Train the V8 branch point and two capacity-matched continuations without anomaly labels."""
    if not 0 <= workers <= 16:
        raise ValueError("workers must be in [0, 16]")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The layerwise mechanism preflight requires CUDA")
    base_cache = Path(base_cache_directory).resolve()
    audio_root_path = Path(audio_root).resolve()
    cache = Path(cache_directory).resolve()
    output = Path(output_directory).resolve()
    checkpoints = Path(checkpoint_directory).resolve()
    config_source = Path(config_path).resolve()
    beats_source = Path(beats_source_directory).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    config = load_fp_naa_config(config_source)
    layerwise = config.layerwise
    if layerwise is None:
        raise ValueError("FP-NAA config does not define a layerwise preflight")
    if _sha256(checkpoint) != config.provenance.checkpoint_sha256:
        raise ValueError("BEATs checkpoint SHA-256 does not match the V8 config")
    index_path = base_cache / "index.parquet"
    metadata_path = base_cache / "cache.json"
    if not index_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Completed base BEATs cache not found: {base_cache}")

    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    gate_path = output / "gate.json"
    summary_path = output / "summary.csv"
    diagnostics_path = output / "diagnostics.csv"
    if gate_path.is_file() and summary_path.is_file() and diagnostics_path.is_file():
        gate = _read_json(gate_path)
        return LayerwisePreflightResult(
            output, summary_path, diagnostics_path, gate_path, bool(gate["passed"])
        )

    frame = pd.read_parquet(index_path)
    train = frame.loc[frame["dataset_split"] == "dev_train"].copy()
    if train.empty or train["file_id"].duplicated().any() or set(train["condition"]) != {"normal"}:
        raise ValueError("V8 preflight requires unique normal development-training rows")
    train = train.sort_values("file_id", kind="stable").reset_index(drop=True)
    features = cache / "features"
    features.mkdir(parents=True, exist_ok=True)
    plans = _make_plans(train, audio_root_path, features, config)
    training_plans, validation_plans = _select_plans(plans, layerwise)
    selected = training_plans + validation_plans
    roles = {
        plan.file_id: ("train" if plan in training_plans else "validation") for plan in selected
    }
    contract = {
        "schema_version": 1,
        "kind": "fp_naa_v8_layerwise_mechanism_preflight",
        "config_sha256": _sha256(config_source),
        "base_cache_metadata_sha256": _sha256(metadata_path),
        "base_cache_index_sha256": _sha256(index_path),
        "checkpoint_sha256": config.provenance.checkpoint_sha256,
        "beats_commit": config.provenance.beats_commit,
        "selection": [
            {"file_id": plan.file_id, "role": roles[plan.file_id], "heldout": plan.heldout}
            for plan in selected
        ],
    }
    contract_sha = _canonical_sha(contract)
    cache_contract = cache / "contract.json"
    if cache_contract.is_file():
        if _canonical_sha(_read_json(cache_contract)) != contract_sha:
            raise ValueError("V8 preflight cache contract mismatch")
    else:
        _atomic_json(cache_contract, contract)
    _atomic_json(output / "contract.json", contract)

    pending = [plan for plan in selected if not plan.feature_path.is_file()]
    completed = len(selected) - len(pending)
    _write_progress(output, stage="cache", completed=completed, total=len(selected), epoch=0)
    frontend = OfficialBEATsFrontend(
        source_directory=beats_source,
        checkpoint_path=checkpoint,
        device=device,
        frequency_patches=config.frontend.frequency_patches,
        mixed_precision=False,
    )
    runtime_probe = _run_actual_beats_probe(
        frontend,
        selected[0],
        config=config,
        layerwise=layerwise,
        device=device,
    )
    _atomic_json(output / "runtime_probe.json", runtime_probe)
    for batch in _chunks(pending, config.frontend.inference_batch_size):
        if workers == 0:
            prepared = [_prepare_item(plan, config) for plan in batch]
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as executor:
                prepared = list(executor.map(lambda plan: _prepare_item(plan, config), batch))
        offsets: list[int] = []
        cursor = 0
        for item in prepared:
            offsets.append(cursor)
            cursor += len(item.waveforms)
        waveforms = np.concatenate([item.waveforms for item in prepared], axis=0)
        taps = frontend.extract_encoder_taps(waveforms, taps=(0, 12))
        for item, offset in zip(prepared, offsets, strict=True):
            _write_item(item, offset=offset, tap0=taps[0], final=taps[12])
        completed += len(batch)
        _write_progress(output, stage="cache", completed=completed, total=len(selected), epoch=0)

    training = _load_arrays(training_plans, heldout=False, workers=workers)
    validation = _load_arrays(validation_plans, heldout=False, workers=workers)
    heldout_plans = [plan for plan in validation_plans if plan.heldout]
    heldout = _load_arrays(heldout_plans, heldout=True, workers=workers)
    print(
        f"V8 cache ready. train={len(training)} validation={len(validation)} heldout={len(heldout)}",
        flush=True,
    )

    _seed_all(layerwise.preflight_seed)
    model = LayerwiseNoiseAwareEncoder(
        beats_model=frontend._model,
        frequency_patches=config.frontend.frequency_patches,
        embedding_dim=config.frontend.embedding_dim,
        hidden_dim=config.adapter.hidden_dim,
        attention_heads=config.adapter.attention_heads,
        dropout=config.adapter.dropout,
        insertion_layers=tuple(layerwise.insertion_layers),
    ).to(device)
    initial_state = model.clone_adapter_state_dict()
    _train_normal(
        model,
        training,
        layerwise=layerwise,
        epochs=layerwise.common_epochs,
        seed=layerwise.preflight_seed,
        device=device,
        output=output,
        stage="common",
    )
    common_state = model.clone_adapter_state_dict()
    common_updated, common_update_norm = finite_adapter_update(initial_state, common_state)
    if not common_updated:
        raise RuntimeError("V8 common MSE branch produced no adapter update")
    _save_checkpoint(checkpoints / "common_branch.pt", common_state, contract_sha, "common")
    common_training_anchor = _predict(model, training, device=device)
    common_validation_anchor = _predict(model, validation, device=device)
    common_heldout_anchor = _predict(model, heldout, device=device)

    _train_normal(
        model,
        training,
        layerwise=layerwise,
        epochs=layerwise.branch_epochs,
        seed=layerwise.preflight_seed + 1,
        device=device,
        output=output,
        stage="l1_mse",
    )
    l1_state = model.clone_adapter_state_dict()
    l1_updated, l1_update_norm = finite_adapter_update(common_state, l1_state)
    if not l1_updated:
        raise RuntimeError("V8 L1 continuation produced no adapter update")
    _save_checkpoint(checkpoints / "l1_layerwise_mse.pt", l1_state, contract_sha, "l1")
    l1_in = _diagnose(
        model,
        validation,
        common_validation_anchor,
        candidate="l1_layerwise_mse",
        fault_set="in_support",
        device=device,
    )
    l1_held = _diagnose(
        model,
        heldout,
        common_heldout_anchor,
        candidate="l1_layerwise_mse",
        fault_set="heldout",
        device=device,
    )

    model.load_adapter_state_dict(common_state)
    _train_fault_transport(
        model,
        training,
        common_training_anchor,
        layerwise=layerwise,
        epochs=layerwise.branch_epochs,
        seed=layerwise.preflight_seed + 1,
        device=device,
        output=output,
    )
    l2_state = model.clone_adapter_state_dict()
    l2_updated, l2_update_norm = finite_adapter_update(common_state, l2_state)
    if not l2_updated:
        raise RuntimeError("V8 L2 continuation produced no adapter update")
    _save_checkpoint(checkpoints / "l2_layerwise_fault_transport.pt", l2_state, contract_sha, "l2")
    l2_in = _diagnose(
        model,
        validation,
        common_validation_anchor,
        candidate="l2_layerwise_fault_transport",
        fault_set="in_support",
        device=device,
    )
    l2_held = _diagnose(
        model,
        heldout,
        common_heldout_anchor,
        candidate="l2_layerwise_fault_transport",
        fault_set="heldout",
        device=device,
    )

    diagnostics = pd.concat((l1_in, l1_held, l2_in, l2_held), ignore_index=True)
    summary = _summarize(diagnostics)
    gate = _make_gate(
        summary,
        layerwise=layerwise,
        update_norms={
            "common": common_update_norm,
            "l1": l1_update_norm,
            "l2": l2_update_norm,
        },
        trainable_parameters=model.trainable_parameter_count(),
        runtime_probe=runtime_probe,
    )
    _atomic_csv(diagnostics_path, diagnostics)
    _atomic_csv(summary_path, summary)
    _atomic_json(gate_path, gate)
    _write_progress(
        output,
        stage="complete",
        completed=layerwise.common_epochs + 2 * layerwise.branch_epochs,
        total=layerwise.common_epochs + 2 * layerwise.branch_epochs,
        epoch=layerwise.common_epochs + 2 * layerwise.branch_epochs,
    )
    print(f"FP-NAA V8 mechanism preflight complete. gate={gate['passed']}", flush=True)
    return LayerwisePreflightResult(
        output, summary_path, diagnostics_path, gate_path, bool(gate["passed"])
    )


def _select_plans(
    plans: list[_AugmentationPlan], layerwise: FPLayerwiseConfig
) -> tuple[list[_AugmentationPlan], list[_AugmentationPlan]]:
    ranked = sorted(
        plans,
        key=lambda plan: hashlib.sha256(
            f"{layerwise.preflight_seed}:{plan.file_id}".encode()
        ).hexdigest(),
    )
    heldout = [plan for plan in ranked if plan.heldout]
    ordinary = [plan for plan in ranked if not plan.heldout]
    heldout_count = layerwise.preflight_heldout_clips
    ordinary_validation_count = layerwise.preflight_validation_clips - heldout_count
    if len(heldout) < heldout_count or len(ordinary) < ordinary_validation_count:
        raise ValueError("Insufficient deterministic clips for the V8 validation split")
    validation = heldout[:heldout_count] + ordinary[:ordinary_validation_count]
    validation_ids = {plan.file_id for plan in validation}
    training = [plan for plan in ordinary if plan.file_id not in validation_ids][
        : layerwise.preflight_train_clips
    ]
    if len(training) != layerwise.preflight_train_clips:
        raise ValueError("Insufficient deterministic clips for the V8 training split")
    return training, validation


def _run_actual_beats_probe(
    frontend: OfficialBEATsFrontend,
    plan: _AugmentationPlan,
    *,
    config: FPNAAConfig,
    layerwise: FPLayerwiseConfig,
    device: str,
) -> dict[str, object]:
    """Verify the custom layer loop against the pinned real BEATs implementation."""
    prepared = _prepare_item(plan, config)
    taps = frontend.extract_encoder_taps(prepared.waveforms, taps=(0, 12))
    noisy_index = prepared.positions["noisy_clean"]
    reference_index = prepared.positions["reference"]
    _seed_all(layerwise.preflight_seed)
    model = LayerwiseNoiseAwareEncoder(
        beats_model=frontend._model,
        frequency_patches=config.frontend.frequency_patches,
        embedding_dim=config.frontend.embedding_dim,
        hidden_dim=config.adapter.hidden_dim,
        attention_heads=config.adapter.attention_heads,
        dropout=config.adapter.dropout,
        insertion_layers=tuple(layerwise.insertion_layers),
    ).to(device)
    model.eval()
    target = _to_device(taps[0][noisy_index : noisy_index + 1], device)
    reference = _to_device(taps[0][reference_index : reference_index + 1], device)
    expected = _to_device(taps[12][noisy_index : noisy_index + 1], device)
    with torch.no_grad():
        predicted = model(target, reference)
    relative_error = float(
        (
            (predicted - expected).reshape(1, -1).norm(dim=1)
            / expected.reshape(1, -1).norm(dim=1).clamp_min(1.0e-8)
        ).item()
    )
    if not math.isfinite(relative_error) or relative_error > 1.0e-5:
        raise RuntimeError(
            "V8 layerwise zero-adapter path does not reproduce pinned BEATs: "
            f"relative_error={relative_error:.8g}"
        )
    before = model.clone_adapter_state_dict()
    model.train()
    optimizer = torch.optim.AdamW(model.adapters.parameters(), lr=layerwise.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    output = model(target, reference)
    loss = functional.mse_loss(output, expected + 0.01)
    _optimizer_step(model, optimizer, loss, layerwise.gradient_clip_norm)
    updated, update_norm = finite_adapter_update(before, model.adapter_state_dict())
    if not updated or not math.isfinite(update_norm):
        raise RuntimeError("V8 actual-BEATs optimizer probe produced no finite adapter update")
    trainable_parameters = model.trainable_parameter_count()
    del optimizer, model, target, reference, expected, predicted, output, loss
    torch.cuda.empty_cache()
    print(
        "V8 actual-BEATs runtime probe passed. "
        f"frozen_path_relative_error={relative_error:.3e} update_norm={update_norm:.3e}",
        flush=True,
    )
    return {
        "schema_version": 1,
        "status": "passed",
        "frozen_path_relative_error": relative_error,
        "frozen_path_relative_error_maximum": 1.0e-5,
        "optimizer_update_norm": update_norm,
        "trainable_parameters": trainable_parameters,
    }


def _prepare_item(plan: _AugmentationPlan, config: FPNAAConfig) -> _Prepared:
    prepared = _prepare(plan, config)
    target, sample_rate = sf.read(plan.target_audio, dtype="float32", always_2d=True)
    if sample_rate != config.frontend.sample_rate or target.shape[1] < 2:
        raise ValueError("V8 source audio violates the frozen stereo frontend contract")
    teacher_clean = fixed_duration_waveform(
        target[:, 0],
        sample_rate=sample_rate,
        duration_seconds=config.frontend.duration_seconds,
    )
    named = dict(zip(prepared.names, prepared.waveforms, strict=True))
    names = ["teacher_clean", "noisy_clean", "reference", "fault_teacher", "fault_noisy"]
    waveforms = [teacher_clean] + [named[name] for name in names[1:]]
    if plan.heldout:
        names.extend(
            [
                "heldout_noisy_clean",
                "heldout_reference",
                "heldout_fault_teacher",
                "heldout_fault_noisy",
            ]
        )
        waveforms.extend(named[name] for name in names[5:])
    return _Prepared(
        plan=plan,
        waveforms=np.stack(waveforms),
        positions={name: index for index, name in enumerate(names)},
    )


def _write_item(item: _Prepared, *, offset: int, tap0: np.ndarray, final: np.ndarray) -> None:
    position = item.positions

    def grid(source: np.ndarray, name: str) -> np.ndarray:
        value = np.asarray(source[offset + position[name]], dtype=np.float16)
        if value.ndim != 3 or not np.isfinite(value).all():
            raise RuntimeError(f"Invalid V8 cached grid: {item.plan.file_id}/{name}")
        return value

    payload: dict[str, np.ndarray] = {
        "tap0_noisy_clean": grid(tap0, "noisy_clean"),
        "tap0_reference": grid(tap0, "reference"),
        "tap0_fault_noisy": grid(tap0, "fault_noisy"),
        "teacher_clean": grid(final, "teacher_clean"),
        "teacher_fault": grid(final, "fault_teacher"),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "file_id": item.plan.file_id,
                    "fault_family": item.plan.family,
                    "heldout": item.plan.heldout,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    }
    if item.plan.heldout:
        payload.update(
            {
                "heldout_tap0_noisy_clean": grid(tap0, "heldout_noisy_clean"),
                "heldout_tap0_reference": grid(tap0, "heldout_reference"),
                "heldout_tap0_fault_noisy": grid(tap0, "heldout_fault_noisy"),
                "heldout_teacher_fault": grid(final, "heldout_fault_teacher"),
            }
        )
    temporary = item.plan.feature_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        savez_compressed = cast(Callable[..., None], np.savez_compressed)
        savez_compressed(handle, **payload)
    os.replace(temporary, item.plan.feature_path)


def _load_arrays(plans: list[_AugmentationPlan], *, heldout: bool, workers: int) -> _Arrays:
    def load(plan: _AugmentationPlan) -> tuple[str, str, tuple[np.ndarray, ...]]:
        prefix = "heldout_" if heldout else ""
        with np.load(plan.feature_path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            names = (
                f"{prefix}tap0_noisy_clean",
                f"{prefix}tap0_reference",
                f"{prefix}tap0_fault_noisy",
                "teacher_clean",
                f"{prefix}teacher_fault",
            )
            values = tuple(np.asarray(payload[name], dtype=np.float16) for name in names)
        if any(value.ndim != 3 or not np.isfinite(value).all() for value in values):
            raise ValueError(f"Invalid V8 cache payload: {plan.feature_path}")
        family = "friction_burst" if heldout else str(metadata["fault_family"])
        return plan.file_id, family, values

    if workers == 0:
        loaded = [load(plan) for plan in plans]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(plans)))) as executor:
            loaded = list(executor.map(load, plans))
    if not loaded:
        raise ValueError("V8 array selection is empty")
    columns = list(zip(*(item[2] for item in loaded), strict=True))
    arrays = [np.stack(column) for column in columns]
    return _Arrays(
        file_ids=tuple(item[0] for item in loaded),
        families=tuple(item[1] for item in loaded),
        noisy=arrays[0],
        reference=arrays[1],
        fault_noisy=arrays[2],
        teacher_clean=arrays[3],
        teacher_fault=arrays[4],
    )


def _train_normal(
    model: LayerwiseNoiseAwareEncoder,
    arrays: _Arrays,
    *,
    layerwise: FPLayerwiseConfig,
    epochs: int,
    seed: int,
    device: str,
    output: Path,
    stage: str,
) -> None:
    optimizer = torch.optim.AdamW(
        model.adapters.parameters(),
        lr=layerwise.learning_rate,
        weight_decay=layerwise.weight_decay,
    )
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(epochs):
        losses: list[float] = []
        for indices in _epoch_batches(len(arrays), layerwise.batch_size, rng):
            noisy, reference, teacher = _normal_batch(arrays, indices, device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(noisy, reference)
            loss = functional.mse_loss(predicted, teacher)
            _optimizer_step(model, optimizer, loss, layerwise.gradient_clip_norm)
            losses.append(float(loss.detach().cpu()))
        _write_progress(output, stage=stage, completed=epoch + 1, total=epochs, epoch=epoch + 1)
        print(f"V8 {stage} epoch={epoch + 1}/{epochs} loss={np.mean(losses):.7f}", flush=True)


def _train_fault_transport(
    model: LayerwiseNoiseAwareEncoder,
    arrays: _Arrays,
    common_anchor: np.ndarray,
    *,
    layerwise: FPLayerwiseConfig,
    epochs: int,
    seed: int,
    device: str,
    output: Path,
) -> None:
    optimizer = torch.optim.AdamW(
        model.adapters.parameters(),
        lr=layerwise.learning_rate,
        weight_decay=layerwise.weight_decay,
    )
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(epochs):
        losses: list[float] = []
        for indices in _epoch_batches(len(arrays), layerwise.batch_size, rng):
            noisy, reference, teacher_clean = _normal_batch(arrays, indices, device)
            fault_noisy = _to_device(arrays.fault_noisy[indices], device)
            teacher_fault = _to_device(arrays.teacher_fault[indices], device)
            anchor = _to_device(common_anchor[indices], device)
            targets = torch.cat((noisy, fault_noisy), dim=0)
            references = torch.cat((reference, reference), dim=0)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(targets, references)
            clean_output, fault_output = predicted.chunk(2, dim=0)
            normal_loss = functional.mse_loss(clean_output, teacher_clean)
            teacher_delta = teacher_fault - teacher_clean
            student_delta = fault_output - clean_output
            relative_error = _relative_norm(student_delta - teacher_delta, teacher_delta)
            excess = torch.relu(relative_error - layerwise.tangent_relative_error_limit)
            tail_count = max(1, math.ceil(layerwise.tangent_tail_fraction * len(excess)))
            tangent = (
                layerwise.tangent_mean_weight * excess.mean()
                + layerwise.tangent_tail_weight * torch.topk(excess, tail_count).values.mean()
            )
            anchor_error = _relative_norm(clean_output - anchor, anchor)
            anchor_penalty = torch.relu(
                anchor_error - layerwise.function_anchor_relative_limit
            ).mean()
            loss = normal_loss + tangent + layerwise.function_anchor_weight * anchor_penalty
            _optimizer_step(model, optimizer, loss, layerwise.gradient_clip_norm)
            losses.append(float(loss.detach().cpu()))
        _write_progress(
            output, stage="l2_fault_transport", completed=epoch + 1, total=epochs, epoch=epoch + 1
        )
        print(
            f"V8 l2_fault_transport epoch={epoch + 1}/{epochs} loss={np.mean(losses):.7f}",
            flush=True,
        )


def _optimizer_step(
    model: LayerwiseNoiseAwareEncoder,
    optimizer: torch.optim.Optimizer,
    loss: Tensor,
    clip_norm: float,
) -> None:
    if not torch.isfinite(loss):
        raise RuntimeError("V8 optimizer loss is non-finite")
    cast(Callable[[], None], loss.backward)()
    norm = torch.nn.utils.clip_grad_norm_(
        model.adapters.parameters(), clip_norm, error_if_nonfinite=True
    )
    if not torch.isfinite(norm):
        raise RuntimeError("V8 optimizer gradient norm is non-finite")
    optimizer.step()


def _predict(model: LayerwiseNoiseAwareEncoder, arrays: _Arrays, *, device: str) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(arrays), 8):
            stop = min(start + 8, len(arrays))
            output = model(
                _to_device(arrays.noisy[start:stop], device),
                _to_device(arrays.reference[start:stop], device),
            )
            outputs.append(output.float().cpu().numpy().astype(np.float16))
    predicted = np.concatenate(outputs, axis=0)
    if not np.isfinite(predicted).all():
        raise RuntimeError("V8 prediction contains non-finite values")
    return predicted


def _diagnose(
    model: LayerwiseNoiseAwareEncoder,
    arrays: _Arrays,
    common_anchor: np.ndarray,
    *,
    candidate: str,
    fault_set: str,
    device: str,
) -> pd.DataFrame:
    clean_outputs: list[np.ndarray] = []
    fault_outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(arrays), 8):
            stop = min(start + 8, len(arrays))
            noisy = _to_device(arrays.noisy[start:stop], device)
            fault = _to_device(arrays.fault_noisy[start:stop], device)
            reference = _to_device(arrays.reference[start:stop], device)
            predicted = model(
                torch.cat((noisy, fault), dim=0),
                torch.cat((reference, reference), dim=0),
            )
            clean, faulty = predicted.chunk(2, dim=0)
            clean_outputs.append(clean.float().cpu().numpy())
            fault_outputs.append(faulty.float().cpu().numpy())
    clean = np.concatenate(clean_outputs, axis=0).astype(np.float64)
    faulty = np.concatenate(fault_outputs, axis=0).astype(np.float64)
    teacher_clean = arrays.teacher_clean.astype(np.float64)
    teacher_fault = arrays.teacher_fault.astype(np.float64)
    teacher_delta = (teacher_fault - teacher_clean).reshape(len(arrays), -1)
    student_delta = (faulty - clean).reshape(len(arrays), -1)
    teacher_norm = np.linalg.norm(teacher_delta, axis=1)
    student_norm = np.linalg.norm(student_delta, axis=1)
    eps = 1.0e-12
    ratio = student_norm / np.maximum(teacher_norm, eps)
    retention = np.exp(-np.abs(np.log(np.maximum(ratio, eps))))
    direction = np.sum(teacher_delta * student_delta, axis=1) / np.maximum(
        teacher_norm * student_norm, eps
    )
    transport = np.linalg.norm(student_delta - teacher_delta, axis=1) / np.maximum(
        teacher_norm, eps
    )
    anchor = common_anchor.astype(np.float64).reshape(len(arrays), -1)
    drift = np.linalg.norm(clean.reshape(len(arrays), -1) - anchor, axis=1) / np.maximum(
        np.linalg.norm(anchor, axis=1), eps
    )
    return pd.DataFrame(
        {
            "file_id": arrays.file_ids,
            "candidate": candidate,
            "fault_set": fault_set,
            "fault_family": arrays.families,
            "retention": retention,
            "direction_cosine": direction,
            "transport_relative_error": transport,
            "normal_function_relative_drift": drift,
        }
    )


def _summarize(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (candidate, fault_set), group in diagnostics.groupby(["candidate", "fault_set"], sort=True):
        rows.append(
            {
                "candidate": candidate,
                "fault_set": fault_set,
                "clips": len(group),
                "retention_median": float(group["retention"].median()),
                "retention_q05": float(group["retention"].quantile(0.05)),
                "direction_median": float(group["direction_cosine"].median()),
                "transport_error_median": float(group["transport_relative_error"].median()),
                "normal_function_drift_median": float(
                    group["normal_function_relative_drift"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _make_gate(
    summary: pd.DataFrame,
    *,
    layerwise: FPLayerwiseConfig,
    update_norms: Mapping[str, float],
    trainable_parameters: int,
    runtime_probe: Mapping[str, object],
) -> dict[str, object]:
    def row(candidate: str, fault_set: str) -> pd.Series:
        selected = summary.loc[
            (summary["candidate"] == candidate) & (summary["fault_set"] == fault_set)
        ]
        if len(selected) != 1:
            raise RuntimeError(f"Missing V8 summary row: {candidate}/{fault_set}")
        return selected.iloc[0]

    l1 = row("l1_layerwise_mse", "in_support")
    l2 = row("l2_layerwise_fault_transport", "in_support")
    heldout = row("l2_layerwise_fault_transport", "heldout")
    checks = {
        "actual_beats_runtime_probe": runtime_probe.get("status") == "passed",
        "finite_real_updates": all(
            math.isfinite(value) and value > 0.0 for value in update_norms.values()
        ),
        "in_support_retention_median": float(l2.retention_median)
        >= layerwise.retention_median_minimum,
        "in_support_retention_q05": float(l2.retention_q05) >= layerwise.retention_q05_minimum,
        "median_gain_over_l1": float(l2.retention_median - l1.retention_median)
        >= layerwise.retention_median_gain_minimum,
        "q05_gain_over_l1": float(l2.retention_q05 - l1.retention_q05)
        >= layerwise.retention_q05_gain_minimum,
        "heldout_retention_median": float(heldout.retention_median)
        >= layerwise.heldout_retention_median_minimum,
        "heldout_retention_q05": float(heldout.retention_q05)
        >= layerwise.heldout_retention_q05_minimum,
        "normal_function_anchor": float(l2.normal_function_drift_median)
        <= layerwise.function_anchor_relative_limit,
    }
    return {
        "schema_version": 1,
        "gate": "V8_M_layerwise_mechanism_preflight",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "l1_in_support_retention_median": float(l1.retention_median),
            "l1_in_support_retention_q05": float(l1.retention_q05),
            "l2_in_support_retention_median": float(l2.retention_median),
            "l2_in_support_retention_q05": float(l2.retention_q05),
            "l2_minus_l1_retention_median": float(l2.retention_median - l1.retention_median),
            "l2_minus_l1_retention_q05": float(l2.retention_q05 - l1.retention_q05),
            "l2_heldout_retention_median": float(heldout.retention_median),
            "l2_heldout_retention_q05": float(heldout.retention_q05),
            "l2_normal_function_drift_median": float(l2.normal_function_drift_median),
        },
        "criteria": {
            "retention_median_minimum": layerwise.retention_median_minimum,
            "retention_q05_minimum": layerwise.retention_q05_minimum,
            "retention_median_gain_minimum": layerwise.retention_median_gain_minimum,
            "retention_q05_gain_minimum": layerwise.retention_q05_gain_minimum,
            "heldout_retention_median_minimum": layerwise.heldout_retention_median_minimum,
            "heldout_retention_q05_minimum": layerwise.heldout_retention_q05_minimum,
            "normal_function_drift_maximum": layerwise.function_anchor_relative_limit,
        },
        "optimizer_update_norms": dict(update_norms),
        "trainable_parameters": trainable_parameters,
        "runtime_probe": dict(runtime_probe),
        "authorization": (
            "A pass authorizes V8 three-seed G2 implementation and execution only. It is not a "
            "development performance result and does not authorize LOMO."
        ),
    }


def _normal_batch(
    arrays: _Arrays, indices: np.ndarray, device: str
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        _to_device(arrays.noisy[indices], device),
        _to_device(arrays.reference[indices], device),
        _to_device(arrays.teacher_clean[indices], device),
    )


def _relative_norm(numerator: Tensor, denominator: Tensor) -> Tensor:
    batch = len(numerator)
    return cast(
        Tensor,
        numerator.reshape(batch, -1).norm(dim=1)
        / denominator.reshape(batch, -1).norm(dim=1).clamp_min(1.0e-8),
    )


def _to_device(values: np.ndarray, device: str) -> Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device, non_blocking=True)


def _epoch_batches(size: int, batch_size: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    permutation = rng.permutation(size)
    for start in range(0, size, batch_size):
        yield permutation[start : start + batch_size]


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _save_checkpoint(
    path: Path, state: Mapping[str, Tensor], contract_sha: str, stage: str
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "stage": stage,
            "contract_sha256": contract_sha,
            "adapter_state": dict(state),
        },
        temporary,
    )
    os.replace(temporary, path)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_progress(output: Path, *, stage: str, completed: int, total: int, epoch: int) -> None:
    text = (
        f"stage={stage}\ncompleted={completed}\ntotal={total}\nepoch={epoch}\n"
        f"updated_utc={datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
    )
    _atomic_text(output / "progress.env", text)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _canonical_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
