#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${CARE_ASD_REPO_DIR:-$HOME/Dung_TDTU/CARE_ASD}"
CONDA_ENV_NAME="${CARE_ASD_CONDA_ENV:-care-asd-fp-naa}"
CONDA_EXE="${CARE_ASD_CONDA_EXE:-$HOME/miniconda3/bin/conda}"
ENVIRONMENT_FILE="environments/fp-naa-cu118.yml"
REQUIREMENTS_FILE="requirements/fp-naa-cu118.lock.txt"
cd "$REPO_DIR"

if [ ! -x "$CONDA_EXE" ]; then
    printf 'Conda executable not found: %s\n' "$CONDA_EXE" >&2
    exit 1
fi
if pgrep -af '[r]un_fp_naa_' >/dev/null; then
    printf 'Refusing Conda setup while an FP-NAA job is running.\n' >&2
    exit 1
fi

if "$CONDA_EXE" run -n "$CONDA_ENV_NAME" python --version >/dev/null 2>&1; then
    printf 'Updating Conda environment: %s\n' "$CONDA_ENV_NAME"
    env -u LD_LIBRARY_PATH -u LD_PRELOAD "$CONDA_EXE" env update -n "$CONDA_ENV_NAME" -f "$ENVIRONMENT_FILE" --prune
else
    printf 'Creating Conda environment: %s\n' "$CONDA_ENV_NAME"
    env -u LD_LIBRARY_PATH -u LD_PRELOAD "$CONDA_EXE" env create -n "$CONDA_ENV_NAME" -f "$ENVIRONMENT_FILE"
fi

printf 'Installing the frozen FP-NAA Python/CUDA package set.\n'
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV_NAME" python -m pip install --requirement "$REQUIREMENTS_FILE"
env -u LD_LIBRARY_PATH -u LD_PRELOAD "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV_NAME" python -m pip install --no-deps --editable "$REPO_DIR"
bash scripts/server/fp_naa_conda_run.sh python -c "import importlib.metadata as metadata, torch, torchaudio; bad=sorted({dist.metadata['Name'] for dist in metadata.distributions() if 'cu13' in (dist.metadata['Name'] or '').lower()}); assert not bad, bad; assert torch.__version__ == '2.6.0+cu118', torch.__version__; assert torchaudio.__version__ == '2.6.0+cu118', torchaudio.__version__; assert torch.version.cuda == '11.8', torch.version.cuda; assert torch.cuda.is_available(); assert torch.backends.cudnn.is_available(); layer=torch.nn.Conv2d(1, 4, 3, padding=1).cuda(); sample=torch.randn(2, 1, 32, 32, device='cuda'); output=layer(sample); torch.cuda.synchronize(); assert output.shape == (2, 4, 32, 32); print(f'env=$CONDA_ENV_NAME torch={torch.__version__} torchaudio={torchaudio.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()} gpu={torch.cuda.get_device_name(0)}')"
printf 'FP-NAA Conda environment is ready: %s\n' "$CONDA_ENV_NAME"

