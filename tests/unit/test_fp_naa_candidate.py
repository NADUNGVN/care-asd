from __future__ import annotations

# ruff: noqa: E402 -- optional Torch must be checked before importing Torch-backed modules.
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")

from care_asd.evaluation.fp_naa_candidate import (
    _exclude_machine,
    _load_or_train_model,
    _TrainingArrays,
)
from care_asd.fp_naa_config import FPNAAConfig


def _tiny_config() -> FPNAAConfig:
    payload = yaml.safe_load(Path("configs/experiment/fp_naa_v1.yaml").read_text())
    payload["frontend"].update(
        {"embedding_dim": 8, "frequency_patches": 2, "inference_batch_size": 2}
    )
    payload["adapter"].update({"hidden_dim": 8, "attention_heads": 2, "dropout": 0.0})
    payload["training"].update(
        {
            "epochs": 1,
            "batch_size": 2,
            "warmup_epochs": 0,
            "workers": 1,
            "mixed_precision": False,
            "screening_seeds": [7],
            "confirmatory_seeds": [7],
        }
    )
    return FPNAAConfig.model_validate(payload)


@pytest.mark.parametrize("candidate", ["c1_mse", "c2_fault_preserving"])
def test_candidate_training_writes_resumable_checkpoint(
    tmp_path: Path,
    candidate: str,
) -> None:
    rng = np.random.default_rng(7)
    shape = (4, 2, 2, 8)
    teacher_clean = rng.normal(size=shape).astype(np.float32)
    reference = rng.normal(scale=0.2, size=shape).astype(np.float32)
    noisy_clean = (teacher_clean + 0.1 * reference).astype(np.float32)
    delta = rng.normal(scale=0.05, size=shape).astype(np.float32)
    arrays = _TrainingArrays(
        frame=pd.DataFrame(
            {
                "file_id": [f"clip-{index}" for index in range(4)],
                "fault_family": ["periodic_resonance"] * 4,
                "heldout": [False] * 4,
                "machine_type": ["fan", "fan", "gearbox", "gearbox"],
            }
        ),
        noisy_clean=noisy_clean,
        reference=reference,
        teacher_clean=teacher_clean,
        fault_noisy=(noisy_clean + delta).astype(np.float32),
        teacher_fault=(teacher_clean + delta).astype(np.float32),
    )
    checkpoint = tmp_path / f"{candidate}.pt"
    progress = tmp_path / "progress"
    progress.mkdir()
    model = _load_or_train_model(
        checkpoint=checkpoint,
        history_path=tmp_path / f"{candidate}.csv",
        arrays=arrays,
        candidate=candidate,  # type: ignore[arg-type]
        seed=7,
        config=_tiny_config(),
        device=torch.device("cpu"),
        progress_output=progress,
        run_number=1,
        total_runs=2,
    )
    assert checkpoint.is_file()
    assert len(pd.read_csv(tmp_path / f"{candidate}.csv")) == 1
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["candidate"] == candidate
    assert json.loads(json.dumps(payload["config"]))["training"]["epochs"] == 1
    output = model(torch.from_numpy(noisy_clean[:1]), torch.from_numpy(reference[:1]))
    assert output.shape == torch.Size((1, 2, 2, 8))

    fold = _exclude_machine(arrays, "fan")
    assert len(fold.frame) == 2
    assert set(fold.frame["machine_type"]) == {"gearbox"}
    assert fold.noisy_clean.shape[0] == 2
