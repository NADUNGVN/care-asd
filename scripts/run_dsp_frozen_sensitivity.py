"""Generate the post-hoc DSP sensitivity package from frozen CARE-ASD scores."""

from __future__ import annotations

import argparse
from pathlib import Path

from care_asd.evaluation.dsp_frozen_sensitivity import run_frozen_dsp_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/research/dsp_frozen_sensitivity_v1.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional new repository-relative output directory; existing paths are never overwritten.",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output = run_frozen_dsp_sensitivity(
        args.config,
        repo_root,
        output_dir_override=args.output_dir,
    )
    print(f"Frozen DSP sensitivity analysis complete: {output}")


if __name__ == "__main__":
    main()
