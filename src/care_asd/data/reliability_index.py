"""Small immutable per-clip CARE reliability index derived from the Phase 5 cache."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from care_asd.data.neural_cache import load_cached_feature


@dataclass(frozen=True)
class ReliabilityIndex:
    directory: Path
    values_path: Path
    clips: int


def build_reliability_index(
    *, neural_cache_directory: str | Path, output_directory: str | Path, workers: int = 1
) -> ReliabilityIndex:
    """Average cached path confidence per clip without reading labels for values."""
    cache = Path(neural_cache_directory)
    output = Path(output_directory)
    index_path = cache / "index.parquet"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite reliability index: {output}")
    if not index_path.is_file() or not (cache / "cache.json").is_file():
        raise FileNotFoundError("Neural cache requires index.parquet and cache.json")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    frame = pd.read_parquet(index_path)
    if "file_id" not in frame or "cache_file" not in frame or frame["file_id"].duplicated().any():
        raise ValueError("Neural cache must provide unique file_id and cache_file columns")

    def read(row: tuple[str, str]) -> tuple[str, float]:
        file_id, cache_file = row
        values = load_cached_feature(cache / cache_file, ("path_confidence",))
        return file_id, float(np.clip(np.mean(values), 0.0, 1.0))

    items = list(zip(frame["file_id"].astype(str), frame["cache_file"].astype(str), strict=True))
    if workers == 1:
        values = [read(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            values = list(executor.map(read, items))
    output.mkdir(parents=True)
    result = pd.DataFrame(values, columns=["file_id", "reliability"]).sort_values(
        "file_id", kind="stable"
    )
    values_path = output / "reliability.parquet"
    result.to_parquet(values_path, index=False)
    metadata = json.loads((cache / "cache.json").read_text(encoding="utf-8"))
    (output / "index.json").write_text(
        json.dumps(
            {
                "clips": len(result),
                "source_neural_cache": str(cache),
                "source_frontend": metadata.get("frontend"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ReliabilityIndex(output, values_path, len(result))
