"""Bounded pre-encoder tangent-repair mechanism preflight for FP-NAA v9."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch import Tensor

from care_asd.data.fp_naa_augmentation_cache import _AugmentationPlan, _make_plans
from care_asd.evaluation.fp_naa_layerwise_preflight import (
    _atomic_csv,
    _atomic_json,
    _canonical_sha,
    _chunks,
    _epoch_batches,
    _prepare_item,
    _read_json,
    _seed_all,
    _sha256,
    _to_device,
    _write_progress,
)
from care_asd.fp_naa_config import FPNAAConfig, FPTapRepairConfig, load_fp_naa_config
from care_asd.models.beats_frontend import OfficialBEATsFrontend
from care_asd.models.fp_naa_adapter import BandwiseReferenceAdapter
from care_asd.models.layerwise_noise_aware import finite_adapter_update


@dataclass(frozen=True)
class TapRepairPreflightResult:
    output_directory: Path
    summary_path: Path
    diagnostics_path: Path
    gate_path: Path
    gate_passed: bool


@dataclass(frozen=True)
class _TapArrays:
    file_ids: tuple[str, ...]
    families: tuple[str, ...]
    noisy: np.ndarray
    reference: np.ndarray
    fault_noisy: np.ndarray
    teacher_clean: np.ndarray
    teacher_fault: np.ndarray

    def __len__(self) -> int:
        return len(self.file_ids)


def run_tap_repair_preflight(
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
) -> TapRepairPreflightResult:
    """Train capacity-matched tap-0 MSE and ACTT branches without anomaly labels."""
    if not 0 <= workers <= 16:
        raise ValueError("workers must be in [0, 16]")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("The tap-repair mechanism preflight requires CUDA")
    base_cache = Path(base_cache_directory).resolve()
    audio_root_path = Path(audio_root).resolve()
    cache = Path(cache_directory).resolve()
    output = Path(output_directory).resolve()
    checkpoints = Path(checkpoint_directory).resolve()
    config_source = Path(config_path).resolve()
    beats_source = Path(beats_source_directory).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    config = load_fp_naa_config(config_source)
    repair = config.tap_repair
    if repair is None or repair.tap != 0:
        raise ValueError("FP-NAA v9 config must define a fixed tap-0 repair preflight")
    if _sha256(checkpoint) != config.provenance.checkpoint_sha256:
        raise ValueError("BEATs checkpoint SHA-256 does not match the V9 config")
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
        return TapRepairPreflightResult(
            output, summary_path, diagnostics_path, gate_path, bool(gate["passed"])
        )

    frame = pd.read_parquet(index_path)
    train = frame.loc[frame["dataset_split"] == "dev_train"].copy()
    if train.empty or train["file_id"].duplicated().any() or set(train["condition"]) != {"normal"}:
        raise ValueError("V9 preflight requires unique normal development-training rows")
    train = train.sort_values("file_id", kind="stable").reset_index(drop=True)
    features = cache / "features"
    features.mkdir(parents=True, exist_ok=True)
    plans = _make_plans(train, audio_root_path, features, config)
    training_plans, validation_plans = _select_plans(plans, repair)
    selected = training_plans + validation_plans
    training_ids = {plan.file_id for plan in training_plans}
    contract = {
        "schema_version": 1,
        "kind": "fp_naa_v9_preencoder_tangent_repair_preflight",
        "config_sha256": _sha256(config_source),
        "base_cache_metadata_sha256": _sha256(metadata_path),
        "base_cache_index_sha256": _sha256(index_path),
        "checkpoint_sha256": config.provenance.checkpoint_sha256,
        "beats_commit": config.provenance.beats_commit,
        "fixed_tap": 0,
        "selection": [
            {
                "file_id": plan.file_id,
                "role": "train" if plan.file_id in training_ids else "validation",
                "heldout": plan.heldout,
            }
            for plan in selected
        ],
    }
    contract_sha = _canonical_sha(contract)
    cache_contract = cache / "contract.json"
    if cache_contract.is_file():
        if _canonical_sha(_read_json(cache_contract)) != contract_sha:
            raise ValueError("V9 tap-repair cache contract mismatch")
    else:
        _atomic_json(cache_contract, contract)
    _atomic_json(output / "contract.json", contract)

    pending = [plan for plan in selected if not plan.feature_path.is_file()]
    completed = len(selected) - len(pending)
    _write_progress(output, stage="cache", completed=completed, total=len(selected), epoch=0)
    if pending:
        frontend = OfficialBEATsFrontend(
            source_directory=beats_source,
            checkpoint_path=checkpoint,
            device=device,
            frequency_patches=config.frontend.frequency_patches,
            mixed_precision=False,
        )
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
            tap0 = frontend.extract_encoder_taps(waveforms, taps=(0,))[0]
            for item, offset in zip(prepared, offsets, strict=True):
                _write_tap_item(item, offset=offset, tap0=tap0)
            completed += len(batch)
            _write_progress(
                output, stage="cache", completed=completed, total=len(selected), epoch=0
            )
        del frontend
        torch.cuda.empty_cache()

    training = _load_tap_arrays(training_plans, heldout=False, workers=workers)
    validation = _load_tap_arrays(validation_plans, heldout=False, workers=workers)
    heldout_plans = [plan for plan in validation_plans if plan.heldout]
    heldout = _load_tap_arrays(heldout_plans, heldout=True, workers=workers)
    print(
        f"V9 tap-0 cache ready. train={len(training)} validation={len(validation)} "
        f"heldout={len(heldout)}",
        flush=True,
    )

    runtime_probe = _run_gpu_probe(training, config=config, repair=repair, device=device)
    _atomic_json(output / "runtime_probe.json", runtime_probe)
    _seed_all(repair.preflight_seed)
    model = _new_adapter(config, device=device)
    initial_state = _clone_state(model)
    _train_normal(
        model,
        training,
        repair=repair,
        epochs=repair.common_epochs,
        seed=repair.preflight_seed,
        device=device,
        output=output,
        stage="common",
    )
    common_state = _clone_state(model)
    common_updated, common_update_norm = finite_adapter_update(initial_state, common_state)
    if not common_updated:
        raise RuntimeError("V9 common MSE branch produced no adapter update")
    _save_checkpoint(checkpoints / "common_branch.pt", common_state, contract_sha, "common")
    common_training_anchor = _predict(model, training, device=device)
    common_validation_anchor = _predict(model, validation, device=device)
    common_heldout_anchor = _predict(model, heldout, device=device)

    _train_normal(
        model,
        training,
        repair=repair,
        epochs=repair.branch_epochs,
        seed=repair.preflight_seed + 1,
        device=device,
        output=output,
        stage="p1_mse",
    )
    p1_state = _clone_state(model)
    p1_updated, p1_update_norm = finite_adapter_update(common_state, p1_state)
    if not p1_updated:
        raise RuntimeError("V9 P1 MSE continuation produced no adapter update")
    _save_checkpoint(checkpoints / "p1_tap0_mse.pt", p1_state, contract_sha, "p1")
    p1_in = _diagnose(
        model,
        validation,
        common_validation_anchor,
        candidate="p1_tap0_mse",
        fault_set="in_support",
        device=device,
    )
    p1_held = _diagnose(
        model,
        heldout,
        common_heldout_anchor,
        candidate="p1_tap0_mse",
        fault_set="heldout",
        device=device,
    )

    model.load_state_dict(common_state, strict=True)
    _train_actt(
        model,
        training,
        common_training_anchor,
        repair=repair,
        epochs=repair.branch_epochs,
        seed=repair.preflight_seed + 1,
        device=device,
        output=output,
    )
    p2_state = _clone_state(model)
    p2_updated, p2_update_norm = finite_adapter_update(common_state, p2_state)
    if not p2_updated:
        raise RuntimeError("V9 P2 ACTT continuation produced no adapter update")
    _save_checkpoint(checkpoints / "p2_tap0_actt.pt", p2_state, contract_sha, "p2")
    p2_in = _diagnose(
        model,
        validation,
        common_validation_anchor,
        candidate="p2_tap0_actt",
        fault_set="in_support",
        device=device,
    )
    p2_held = _diagnose(
        model,
        heldout,
        common_heldout_anchor,
        candidate="p2_tap0_actt",
        fault_set="heldout",
        device=device,
    )

    diagnostics = pd.concat((p1_in, p1_held, p2_in, p2_held), ignore_index=True)
    summary = _summarize(diagnostics)
    gate = _make_gate(
        summary,
        repair=repair,
        runtime_probe=runtime_probe,
        update_norms={"common": common_update_norm, "p1": p1_update_norm, "p2": p2_update_norm},
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters()),
    )
    _atomic_csv(diagnostics_path, diagnostics)
    _atomic_csv(summary_path, summary)
    _atomic_json(gate_path, gate)
    total_epochs = repair.common_epochs + 2 * repair.branch_epochs
    _write_progress(
        output, stage="complete", completed=total_epochs, total=total_epochs, epoch=total_epochs
    )
    print(f"FP-NAA V9 tap-repair preflight complete. gate={gate['passed']}", flush=True)
    return TapRepairPreflightResult(
        output, summary_path, diagnostics_path, gate_path, bool(gate["passed"])
    )


def _write_tap_item(item: Any, *, offset: int, tap0: np.ndarray) -> None:
    position = item.positions

    def grid(name: str) -> np.ndarray:
        value = np.asarray(tap0[offset + position[name]], dtype=np.float16)
        if value.ndim != 3 or not np.isfinite(value).all():
            raise RuntimeError(f"Invalid V9 cached grid: {item.plan.file_id}/{name}")
        return value

    payload: dict[str, np.ndarray] = {
        "tap0_noisy_clean": grid("noisy_clean"),
        "tap0_reference": grid("reference"),
        "tap0_fault_noisy": grid("fault_noisy"),
        "tap0_teacher_clean": grid("teacher_clean"),
        "tap0_teacher_fault": grid("fault_teacher"),
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
                "heldout_tap0_noisy_clean": grid("heldout_noisy_clean"),
                "heldout_tap0_reference": grid("heldout_reference"),
                "heldout_tap0_fault_noisy": grid("heldout_fault_noisy"),
                "heldout_tap0_teacher_fault": grid("heldout_fault_teacher"),
            }
        )
    temporary = item.plan.feature_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        savez_compressed = cast(Callable[..., None], np.savez_compressed)
        savez_compressed(handle, **payload)
    os.replace(temporary, item.plan.feature_path)


def _load_tap_arrays(plans: list[_AugmentationPlan], *, heldout: bool, workers: int) -> _TapArrays:
    def load(plan: _AugmentationPlan) -> tuple[str, str, tuple[np.ndarray, ...]]:
        prefix = "heldout_" if heldout else ""
        with np.load(plan.feature_path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            names = (
                f"{prefix}tap0_noisy_clean",
                f"{prefix}tap0_reference",
                f"{prefix}tap0_fault_noisy",
                "tap0_teacher_clean",
                f"{prefix}tap0_teacher_fault",
            )
            values = tuple(np.asarray(payload[name], dtype=np.float16) for name in names)
        if any(value.ndim != 3 or not np.isfinite(value).all() for value in values):
            raise ValueError(f"Invalid V9 cache payload: {plan.feature_path}")
        family = "friction_burst" if heldout else str(metadata["fault_family"])
        return plan.file_id, family, values

    if workers == 0:
        loaded = [load(plan) for plan in plans]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(plans)))) as executor:
            loaded = list(executor.map(load, plans))
    if not loaded:
        raise ValueError("V9 array selection is empty")
    columns = list(zip(*(item[2] for item in loaded), strict=True))
    arrays = [np.stack(column) for column in columns]
    return _TapArrays(
        file_ids=tuple(item[0] for item in loaded),
        families=tuple(item[1] for item in loaded),
        noisy=arrays[0],
        reference=arrays[1],
        fault_noisy=arrays[2],
        teacher_clean=arrays[3],
        teacher_fault=arrays[4],
    )


def _select_plans(
    plans: list[_AugmentationPlan], repair: FPTapRepairConfig
) -> tuple[list[_AugmentationPlan], list[_AugmentationPlan]]:
    ranked = sorted(
        plans,
        key=lambda plan: hashlib.sha256(
            f"{repair.preflight_seed}:{plan.file_id}".encode()
        ).hexdigest(),
    )
    heldout = [plan for plan in ranked if plan.heldout]
    ordinary = [plan for plan in ranked if not plan.heldout]
    heldout_count = repair.preflight_heldout_clips
    ordinary_validation_count = repair.preflight_validation_clips - heldout_count
    if len(heldout) < heldout_count or len(ordinary) < ordinary_validation_count:
        raise ValueError("Insufficient deterministic clips for the V9 validation split")
    validation = heldout[:heldout_count] + ordinary[:ordinary_validation_count]
    validation_ids = {plan.file_id for plan in validation}
    training = [plan for plan in ordinary if plan.file_id not in validation_ids][
        : repair.preflight_train_clips
    ]
    if len(training) != repair.preflight_train_clips:
        raise ValueError("Insufficient deterministic clips for the V9 training split")
    return training, validation


def _new_adapter(config: FPNAAConfig, *, device: str) -> BandwiseReferenceAdapter:
    model = BandwiseReferenceAdapter(
        embedding_dim=config.frontend.embedding_dim,
        hidden_dim=config.adapter.hidden_dim,
        attention_heads=config.adapter.attention_heads,
        dropout=config.adapter.dropout,
        conditioning_mode="target_conditioned",
    )
    model.to(device)
    return model


def _clone_state(model: BandwiseReferenceAdapter) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _run_gpu_probe(
    arrays: _TapArrays,
    *,
    config: FPNAAConfig,
    repair: FPTapRepairConfig,
    device: str,
) -> dict[str, object]:
    _seed_all(repair.preflight_seed)
    model = _new_adapter(config, device=device).eval()
    target = _to_device(arrays.noisy[:1], device)
    reference = _to_device(arrays.reference[:1], device)
    teacher = _to_device(arrays.teacher_clean[:1], device)
    with torch.no_grad():
        predicted = model(target, reference)
    relative_error = float(
        (
            (predicted - target).reshape(1, -1).norm(dim=1)
            / target.reshape(1, -1).norm(dim=1).clamp_min(1.0e-8)
        ).item()
    )
    if not math.isfinite(relative_error) or relative_error > 1.0e-7:
        raise RuntimeError(f"V9 zero-adapter identity probe failed: {relative_error:.8g}")
    before = _clone_state(model)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=repair.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    loss = functional.mse_loss(model(target, reference), teacher + 0.01)
    _optimizer_step(model, optimizer, loss, repair.gradient_clip_norm)
    after = _clone_state(model)
    updated, update_norm = finite_adapter_update(before, after)
    if not updated or not math.isfinite(update_norm):
        raise RuntimeError("V9 GPU optimizer probe produced no finite adapter update")
    trainable = sum(parameter.numel() for parameter in model.parameters())
    del model, optimizer, target, reference, teacher, predicted, loss
    torch.cuda.empty_cache()
    print(
        f"V9 tap-0 GPU probe passed. identity_error={relative_error:.3e} "
        f"update_norm={update_norm:.3e}",
        flush=True,
    )
    return {
        "schema_version": 1,
        "status": "passed",
        "identity_relative_error": relative_error,
        "identity_relative_error_maximum": 1.0e-7,
        "optimizer_update_norm": update_norm,
        "trainable_parameters": trainable,
    }


def _train_normal(
    model: BandwiseReferenceAdapter,
    arrays: _TapArrays,
    *,
    repair: FPTapRepairConfig,
    epochs: int,
    seed: int,
    device: str,
    output: Path,
    stage: str,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=repair.learning_rate, weight_decay=repair.weight_decay
    )
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(epochs):
        losses: list[float] = []
        for indices in _epoch_batches(len(arrays), repair.batch_size, rng):
            noisy, reference, teacher = _normal_batch(arrays, indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss = functional.mse_loss(model(noisy, reference), teacher)
            _optimizer_step(model, optimizer, loss, repair.gradient_clip_norm)
            losses.append(float(loss.detach().cpu()))
        _write_progress(output, stage=stage, completed=epoch + 1, total=epochs, epoch=epoch + 1)
        print(f"V9 {stage} epoch={epoch + 1}/{epochs} loss={np.mean(losses):.7f}", flush=True)


def _train_actt(
    model: BandwiseReferenceAdapter,
    arrays: _TapArrays,
    common_anchor: np.ndarray,
    *,
    repair: FPTapRepairConfig,
    epochs: int,
    seed: int,
    device: str,
    output: Path,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=repair.learning_rate, weight_decay=repair.weight_decay
    )
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(epochs):
        losses: list[float] = []
        for indices in _epoch_batches(len(arrays), repair.batch_size, rng):
            noisy, reference, teacher_clean = _normal_batch(arrays, indices, device)
            fault_noisy = _to_device(arrays.fault_noisy[indices], device)
            teacher_fault = _to_device(arrays.teacher_fault[indices], device)
            anchor = _to_device(common_anchor[indices], device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(
                torch.cat((noisy, fault_noisy), dim=0),
                torch.cat((reference, reference), dim=0),
            )
            clean_output, fault_output = predicted.chunk(2, dim=0)
            normal_loss = functional.mse_loss(clean_output, teacher_clean)
            teacher_delta = teacher_fault - teacher_clean
            student_delta = fault_output - clean_output
            relative_error = _relative_norm(student_delta - teacher_delta, teacher_delta)
            excess = torch.relu(relative_error - repair.tangent_relative_error_limit)
            tail_count = max(1, math.ceil(repair.tangent_tail_fraction * len(excess)))
            tangent = (
                repair.tangent_mean_weight * excess.mean()
                + repair.tangent_tail_weight * torch.topk(excess, tail_count).values.mean()
            )
            anchor_error = _relative_norm(clean_output - anchor, anchor)
            anchor_penalty = torch.relu(anchor_error - repair.function_anchor_relative_limit).mean()
            loss = normal_loss + tangent + repair.function_anchor_weight * anchor_penalty
            _optimizer_step(model, optimizer, loss, repair.gradient_clip_norm)
            losses.append(float(loss.detach().cpu()))
        _write_progress(output, stage="p2_actt", completed=epoch + 1, total=epochs, epoch=epoch + 1)
        print(f"V9 p2_actt epoch={epoch + 1}/{epochs} loss={np.mean(losses):.7f}", flush=True)


def _optimizer_step(
    model: BandwiseReferenceAdapter,
    optimizer: torch.optim.Optimizer,
    loss: Tensor,
    clip_norm: float,
) -> None:
    if not torch.isfinite(loss):
        raise RuntimeError("V9 optimizer loss is non-finite")
    cast(Callable[[], None], loss.backward)()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise RuntimeError("V9 optimizer gradient norm is non-finite")
    optimizer.step()


def _normal_batch(
    arrays: _TapArrays, indices: np.ndarray, device: str
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        _to_device(arrays.noisy[indices], device),
        _to_device(arrays.reference[indices], device),
        _to_device(arrays.teacher_clean[indices], device),
    )


def _relative_norm(numerator: Tensor, denominator: Tensor) -> Tensor:
    top = numerator.reshape(len(numerator), -1).norm(dim=1)
    bottom = denominator.reshape(len(denominator), -1).norm(dim=1).clamp_min(1.0e-8)
    return cast(Tensor, top / bottom)


def _predict(
    model: BandwiseReferenceAdapter,
    arrays: _TapArrays,
    *,
    device: str,
    fault: bool = False,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    source = arrays.fault_noisy if fault else arrays.noisy
    with torch.inference_mode():
        for start in range(0, len(arrays), 16):
            end = min(start + 16, len(arrays))
            output = model(
                _to_device(source[start:end], device),
                _to_device(arrays.reference[start:end], device),
            )
            outputs.append(output.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _diagnose(
    model: BandwiseReferenceAdapter,
    arrays: _TapArrays,
    common_anchor: np.ndarray,
    *,
    candidate: str,
    fault_set: str,
    device: str,
) -> pd.DataFrame:
    clean = _predict(model, arrays, device=device, fault=False).astype(np.float64)
    fault = _predict(model, arrays, device=device, fault=True).astype(np.float64)
    teacher_clean = arrays.teacher_clean.astype(np.float64)
    teacher_fault = arrays.teacher_fault.astype(np.float64)
    raw_clean = arrays.noisy.astype(np.float64)
    raw_fault = arrays.fault_noisy.astype(np.float64)
    student_delta = (fault - clean).reshape(len(arrays), -1)
    teacher_delta = (teacher_fault - teacher_clean).reshape(len(arrays), -1)
    raw_delta = (raw_fault - raw_clean).reshape(len(arrays), -1)
    eps = 1.0e-12
    teacher_norm = np.linalg.norm(teacher_delta, axis=1)
    student_norm = np.linalg.norm(student_delta, axis=1)
    raw_norm = np.linalg.norm(raw_delta, axis=1)
    ratio = student_norm / np.maximum(teacher_norm, eps)
    raw_ratio = raw_norm / np.maximum(teacher_norm, eps)
    retention = np.exp(-np.abs(np.log(np.maximum(ratio, eps))))
    raw_retention = np.exp(-np.abs(np.log(np.maximum(raw_ratio, eps))))
    direction = np.sum(student_delta * teacher_delta, axis=1) / np.maximum(
        student_norm * teacher_norm, eps
    )
    raw_direction = np.sum(raw_delta * teacher_delta, axis=1) / np.maximum(
        raw_norm * teacher_norm, eps
    )
    transport_error = np.linalg.norm(student_delta - teacher_delta, axis=1) / np.maximum(
        teacher_norm, eps
    )
    raw_transport_error = np.linalg.norm(raw_delta - teacher_delta, axis=1) / np.maximum(
        teacher_norm, eps
    )
    anchor = common_anchor.astype(np.float64).reshape(len(arrays), -1)
    normal = clean.reshape(len(arrays), -1)
    normal_drift = np.linalg.norm(normal - anchor, axis=1) / np.maximum(
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
            "transport_relative_error": transport_error,
            "raw_retention": raw_retention,
            "raw_direction_cosine": raw_direction,
            "raw_transport_relative_error": raw_transport_error,
            "teacher_delta_norm": teacher_norm,
            "student_delta_norm": student_norm,
            "raw_delta_norm": raw_norm,
            "normal_function_relative_drift": normal_drift,
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
                "raw_retention_median": float(group["raw_retention"].median()),
                "raw_retention_q05": float(group["raw_retention"].quantile(0.05)),
                "normal_function_drift_median": float(
                    group["normal_function_relative_drift"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _make_gate(
    summary: pd.DataFrame,
    *,
    repair: FPTapRepairConfig,
    runtime_probe: dict[str, object],
    update_norms: dict[str, float],
    trainable_parameters: int,
) -> dict[str, object]:
    def row(candidate: str, fault_set: str) -> pd.Series:
        selected = summary.loc[
            (summary["candidate"] == candidate) & (summary["fault_set"] == fault_set)
        ]
        if len(selected) != 1:
            raise ValueError(f"Missing V9 summary row: {candidate}/{fault_set}")
        return selected.iloc[0]

    p1 = row("p1_tap0_mse", "in_support")
    p2 = row("p2_tap0_actt", "in_support")
    heldout = row("p2_tap0_actt", "heldout")
    finite_updates = all(math.isfinite(value) and value > 0.0 for value in update_norms.values())
    checks = {
        "gpu_runtime_probe": runtime_probe.get("status") == "passed",
        "finite_real_updates": finite_updates,
        "in_support_retention_median": float(p2.retention_median)
        >= repair.retention_median_minimum,
        "in_support_retention_q05": float(p2.retention_q05) >= repair.retention_q05_minimum,
        "median_gain_over_p1": float(p2.retention_median - p1.retention_median)
        >= repair.retention_median_gain_minimum,
        "q05_gain_over_p1": float(p2.retention_q05 - p1.retention_q05)
        >= repair.retention_q05_gain_minimum,
        "heldout_retention_median": float(heldout.retention_median)
        >= repair.heldout_retention_median_minimum,
        "heldout_retention_q05": float(heldout.retention_q05)
        >= repair.heldout_retention_q05_minimum,
        "normal_function_anchor": float(p2.normal_function_drift_median)
        <= repair.function_anchor_relative_limit,
    }
    metrics = {
        "raw_in_support_retention_median": float(p2.raw_retention_median),
        "raw_in_support_retention_q05": float(p2.raw_retention_q05),
        "p1_in_support_retention_median": float(p1.retention_median),
        "p1_in_support_retention_q05": float(p1.retention_q05),
        "p2_in_support_retention_median": float(p2.retention_median),
        "p2_in_support_retention_q05": float(p2.retention_q05),
        "p2_minus_p1_retention_median": float(p2.retention_median - p1.retention_median),
        "p2_minus_p1_retention_q05": float(p2.retention_q05 - p1.retention_q05),
        "p2_heldout_retention_median": float(heldout.retention_median),
        "p2_heldout_retention_q05": float(heldout.retention_q05),
        "p2_normal_function_drift_median": float(p2.normal_function_drift_median),
    }
    return {
        "schema_version": 1,
        "gate": "V9_M_preencoder_tangent_repair_preflight",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "criteria": {
            "retention_median_minimum": repair.retention_median_minimum,
            "retention_q05_minimum": repair.retention_q05_minimum,
            "retention_median_gain_minimum": repair.retention_median_gain_minimum,
            "retention_q05_gain_minimum": repair.retention_q05_gain_minimum,
            "heldout_retention_median_minimum": repair.heldout_retention_median_minimum,
            "heldout_retention_q05_minimum": repair.heldout_retention_q05_minimum,
            "normal_function_drift_maximum": repair.function_anchor_relative_limit,
        },
        "optimizer_update_norms": update_norms,
        "runtime_probe": runtime_probe,
        "trainable_parameters": trainable_parameters,
        "authorization": (
            "A pass authorizes V9 three-seed G2 implementation and execution only. "
            "It is not a development performance result and does not authorize LOMO."
        ),
    }


def _save_checkpoint(path: Path, state: dict[str, Tensor], contract_sha: str, branch: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "schema_version": 1,
            "kind": "fp_naa_v9_tap0_adapter",
            "branch": branch,
            "contract_sha256": contract_sha,
            "state_dict": state,
        },
        temporary,
    )
    os.replace(temporary, path)
