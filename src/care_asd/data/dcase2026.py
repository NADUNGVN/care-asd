"""Safe acquisition and audit utilities for DCASE 2026 Task 2 data."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast
from urllib.request import Request, urlopen

import pandas as pd
import soundfile as sf

DatasetSplit = Literal["dev", "additional", "evaluation"]
_SPLIT_RECORDS: dict[DatasetSplit, str] = {
    "dev": "19336329",
    "additional": "20151556",
    "evaluation": "20437238",
}
_SPLIT_ALIASES = {"dev": "dev", "additional": "additional", "evaluation": "evaluation"}


@dataclass(frozen=True)
class DownloadedArchive:
    """Verified archive materialized under the dataset root."""

    filename: str
    path: str
    checksum: str
    bytes: int
    downloaded: bool


@dataclass(frozen=True)
class ExtractionSummary:
    """Idempotent extraction result."""

    archive: str
    extracted_files: int
    reused_files: int


@dataclass(frozen=True)
class DatasetAudit:
    """Manifest-level channel and metadata audit."""

    manifest_path: str
    clips: int
    stereo_clips: int
    invalid_channel_clips: int
    sample_rates: tuple[int, ...]
    conditions: tuple[str, ...]
    domains: tuple[str, ...]


def normalize_split(value: str) -> DatasetSplit:
    """Validate a user-visible split name."""
    if value not in _SPLIT_ALIASES:
        raise ValueError("split must be one of: dev, additional, evaluation")
    return cast(DatasetSplit, _SPLIT_ALIASES[value])


def download_dcase2026_split(
    data_root: str | Path,
    split: DatasetSplit,
    *,
    accept_evaluation_policy: bool = False,
    timeout_seconds: int = 60,
) -> list[DownloadedArchive]:
    """Download and checksum-verify every archive in an official Zenodo record."""
    if split == "evaluation" and not accept_evaluation_policy:
        raise PermissionError("evaluation download requires explicit policy acceptance")
    root = Path(data_root)
    archive_dir = root / "raw" / "dcase2026" / split / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    record_id = _SPLIT_RECORDS[split]
    metadata = _fetch_json(f"https://zenodo.org/api/records/{record_id}", timeout_seconds)
    files = metadata.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Zenodo record {record_id} did not contain downloadable files")

    downloaded: list[DownloadedArchive] = []
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Zenodo record contains an invalid file entry")
        filename = entry.get("key")
        checksum = entry.get("checksum")
        links = entry.get("links")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(checksum, str)
            or not isinstance(links, dict)
            or not isinstance(links.get("self"), str)
        ):
            raise ValueError("Zenodo file entry is missing a safe name, checksum, or URL")
        target = archive_dir / filename
        was_downloaded = not target.exists()
        if was_downloaded:
            _download_file(links["self"], target, timeout_seconds)
        _verify_checksum(target, checksum)
        downloaded.append(
            DownloadedArchive(
                filename=filename,
                path=str(target),
                checksum=checksum,
                bytes=target.stat().st_size,
                downloaded=was_downloaded,
            )
        )

    sidecar = root / "checksums" / f"dcase2026_{split}_downloads.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    _write_json_idempotent(
        sidecar,
        {
            "record_id": record_id,
            "files": [
                {key: value for key, value in asdict(item).items() if key != "downloaded"}
                for item in downloaded
            ],
        },
    )
    return downloaded


def extract_dcase2026_split(data_root: str | Path, split: DatasetSplit) -> list[ExtractionSummary]:
    """Safely extract all ZIP archives for one split without overwriting files."""
    root = Path(data_root)
    archive_dir = root / "raw" / "dcase2026" / split / "archives"
    if not archive_dir.is_dir():
        raise FileNotFoundError(f"No archive directory for split={split}: {archive_dir}")
    archives = sorted(archive_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"No ZIP archives found for split={split}: {archive_dir}")
    extracted_root = root / "raw" / "dcase2026" / split / "extracted"
    return [_extract_zip_idempotent(archive, extracted_root) for archive in archives]


def build_dcase2026_manifest(
    data_root: str | Path,
    split: DatasetSplit,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Write a deterministic parquet manifest from extracted WAV metadata."""
    root = Path(data_root)
    extracted_root = root / "raw" / "dcase2026" / split / "extracted"
    wav_paths = sorted(extracted_root.rglob("*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found for split={split}: {extracted_root}")

    records = [_manifest_record(path, extracted_root, split) for path in wav_paths]
    frame = pd.DataFrame(records).sort_values("file_id", kind="stable").reset_index(drop=True)
    destination = (
        Path(output_path) if output_path is not None else dcase2026_manifest_path(root, split)
    )
    if destination.exists():
        raise FileExistsError(f"Manifest already exists (refusing to overwrite): {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return destination


def audit_dcase2026_manifest(manifest_path: str | Path) -> DatasetAudit:
    """Validate immutable manifest constraints required by CARE-ASD."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    frame = pd.read_parquet(path)
    required = {
        "file_id",
        "relative_path",
        "channels",
        "sample_rate",
        "condition",
        "domain",
        "dataset_split",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Manifest contains no clips")
    invalid_channels = int((frame["channels"] != 2).sum())
    if invalid_channels:
        raise ValueError(f"DCASE 2026 requires stereo audio; invalid clips={invalid_channels}")
    if not frame["file_id"].is_unique:
        raise ValueError("Manifest file_id values must be unique")
    relative_paths = frame["relative_path"].astype(str)
    if any(
        candidate in {"", ".", ".."}
        or Path(candidate).is_absolute()
        or ".." in Path(candidate).parts
        for candidate in relative_paths
    ):
        raise ValueError("Manifest contains an unsafe relative_path")
    return DatasetAudit(
        manifest_path=str(path),
        clips=len(frame),
        stereo_clips=int((frame["channels"] == 2).sum()),
        invalid_channel_clips=invalid_channels,
        sample_rates=tuple(sorted(int(value) for value in frame["sample_rate"].unique())),
        conditions=tuple(sorted(str(value) for value in frame["condition"].unique())),
        domains=tuple(sorted(str(value) for value in frame["domain"].unique())),
    )


def _fetch_json(url: str, timeout_seconds: int) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "CARE-ASD/0.1"})
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Zenodo API response must be an object")
    return payload


def _download_file(url: str, destination: Path, timeout_seconds: int) -> None:
    request = Request(url, headers={"User-Agent": "CARE-ASD/0.1"})
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".part",
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                shutil.copyfileobj(response, temporary, length=1024 * 1024)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(destination)


def _verify_checksum(path: Path, checksum: str) -> None:
    try:
        algorithm, expected = checksum.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError(f"Invalid Zenodo checksum for {path.name}: {checksum}") from exc
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Checksum mismatch for {path}")


def _extract_zip_idempotent(archive: Path, extracted_root: Path) -> ExtractionSummary:
    extracted_root.mkdir(parents=True, exist_ok=True)
    root_resolved = extracted_root.resolve()
    extracted_files = 0
    reused_files = 0
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (extracted_root / member.filename).resolve()
            if target != root_resolved and root_resolved not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != member.file_size:
                    raise FileExistsError(f"Existing extraction differs: {target}")
                reused_files += 1
                continue
            with source.open(member) as source_file, target.open("xb") as target_file:
                shutil.copyfileobj(source_file, target_file, length=1024 * 1024)
            extracted_files += 1
    return ExtractionSummary(
        archive=str(archive),
        extracted_files=extracted_files,
        reused_files=reused_files,
    )


def _manifest_record(path: Path, extracted_root: Path, split: DatasetSplit) -> dict[str, Any]:
    info = sf.info(path)
    relative = path.relative_to(extracted_root)
    tokens = path.stem.split("_")
    lower_tokens = {token.lower() for token in tokens}
    condition = (
        "anomaly"
        if "anomaly" in lower_tokens
        else "normal"
        if "normal" in lower_tokens
        else "unknown"
    )
    domain = (
        "source"
        if "source" in lower_tokens
        else "target"
        if "target" in lower_tokens
        else "unknown"
    )
    machine_type = relative.parts[0] if relative.parts else "unknown"
    section_index = next((index for index, token in enumerate(tokens) if token == "section"), None)
    section = (
        f"section_{tokens[section_index + 1]}"
        if section_index is not None and section_index + 1 < len(tokens)
        else "unknown"
    )
    dataset_split = {
        "dev": "dev_test" if "test" in relative.parts else "dev_train",
        "additional": "add_train",
        "evaluation": "eval_test",
    }[split]
    return {
        "file_id": relative.as_posix(),
        "relative_path": relative.as_posix(),
        "machine_type": machine_type,
        "section": section,
        "condition": condition,
        "domain": domain,
        "dataset_split": dataset_split,
        "sample_rate": int(info.samplerate),
        "frames": int(info.frames),
        "duration_seconds": float(info.duration),
        "channels": int(info.channels),
        "subtype": info.subtype,
    }


def dcase2026_manifest_path(data_root: str | Path, split: DatasetSplit) -> Path:
    """Return the canonical immutable manifest location for a DCASE split."""
    root = Path(data_root)
    suffix = {"dev": "dev", "additional": "additional", "evaluation": "evaluation"}[split]
    return root / "manifests" / f"dcase2026_{suffix}.parquet"


def _write_json_idempotent(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Checksum sidecar differs (refusing to overwrite): {path}")
        return
    path.write_text(text, encoding="utf-8")
