# FP-NAA Conda and server architecture

Status: active contract for branch `research/fp-naa`.

## Design goals

- one named environment: `care-asd-fp-naa`;
- no FP-NAA shell wrapper, implicit activation, `nohup`, `pgrep`, or `.venv` dependency;
- one public CLI with stable JSON success and error envelopes;
- exactly one detached experiment job at a time;
- atomic PID/state tracking that survives SSH disconnection;
- gate-driven continuation with no automatic retry after a process failure;
- immutable result reports committed to `research/fp-naa`.

## Component boundaries

```text
conda run -n care-asd-fp-naa
        |
        v
care-asd fp-naa job <command>          public, validated JSON contract
        |
        v
care_asd.server.fp_naa_jobs            process/state/gate controller
        |
        +--> detached current Python   start_new_session + file-backed stdout
        |
        +--> existing fp-naa commands  cache/train/evaluate implementations
        |
        +--> reports + Git             durable evidence and remote hand-off
```

The controller does not invoke a shell. External commands are argument arrays executed with
`shell=False`; the detached child uses `sys.executable`, which is the Python interpreter belonging
to the selected Conda environment.

## Environment lifecycle

Conda owns Python 3.11 and the environment boundary. Pip installs the exact transitive set in
`requirements/fp-naa-cu118.lock.txt` inside that boundary. This avoids mixing Conda CUDA libraries
with pip CUDA libraries while retaining normal Conda inspection/export/removal commands.

Setup consists of explicit commands rather than a setup script. A runtime that has ever failed a
Torch/CUDA check is rebuilt instead of repaired in place: `conda env update --prune` cannot be
relied on to remove CUDA packages previously installed by pip.

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD conda env remove -n care-asd-fp-naa -y
```

An `EnvironmentLocationNotFound` response only means there was no old environment to remove.

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD conda env create -f environments/fp-naa-cu118.yml
```

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD conda run -n care-asd-fp-naa python -m pip install -r requirements/fp-naa-cu118.lock.txt
```

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD conda run -n care-asd-fp-naa python -m pip install --no-deps -e .
```

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD conda run -n care-asd-fp-naa care-asd fp-naa runtime-check
```

The last command rejects the wrong Conda env, any CUDA 13 package, a Torch/torchaudio version
mismatch, unavailable CUDA, or a failed real cuDNN convolution.

Every detached subprocess also receives `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Adapter training uses
hard deterministic algorithms, disables cuDNN benchmarking and the nondeterministic flash/memory-
efficient attention kernels, and records the resulting backend state in checkpoints and run
metadata. A determinism warning is therefore a failed experiment contract, not an ignorable log.

## BEATs extraction numerical contract

Frozen BEATs inference and adapter optimization have separate precision controls. BEATs token
extraction runs in FP32 (`frontend.inference_mixed_precision: false`); adapter training may still
use AMP (`training.mixed_precision: true`). Every audio batch, FP32 frontend result, and float16
cache payload must be finite before an atomic feature write. Completed caches are reusable only
when their manifest, full config, frontend contract, checkpoint, cache schema, and precision mode
match exactly.

The cache reader overlaps one bounded batch of CPU/NFS loading with the current GPU extraction.
The registered server configuration uses batch size 16 and at most 12 loader workers, leaving the
reserved CPU capacity available for other SERVER-02 work. MBW and power may still oscillate at
batch boundaries; correctness and throughput are determined from progress and finite-value checks,
not from a flat utilization trace.

The failed C0 run `server02_fp_naa_c0_20260817T042420Z` produced 8,400 cache files under the old
`beats_iter3_stereo_10s_v1` namespace before baseline validation found an all-NaN token payload.
That run is a numerical-infrastructure failure, not a C0 scientific result. Its report and cache
remain immutable evidence. All new runs use `beats_iter3_stereo_10s_fp32infer_v2`; downstream
counterfactual and reference-safety caches also use new namespaces so no invalid v1 artifact can
enter an experiment.

## Public command contract

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD conda run -n care-asd-fp-naa care-asd fp-naa job start --stage c0 --workers 12
```

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD conda run -n care-asd-fp-naa care-asd fp-naa job status
```

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD conda run -n care-asd-fp-naa care-asd fp-naa job list --limit 10
```

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD conda run -n care-asd-fp-naa care-asd fp-naa job continue --workers 12
```

`start` is explicit and is the only public mutation that creates a process. `status` and `list` are
read-only snapshots. `continue` returns the current status when a job is active, starts the next
stage only when all earlier gates pass, and refuses to retry a failed process automatically.

## Durable state contract

The untracked runtime registry is under `outputs/fp_naa/job_control/` and
`outputs/fp_naa/jobs/<run_id>/`. Each `state.json` has schema version 1 and records:

- run ID and frozen stage name;
- `STARTING`, `RUNNING`, `DONE`, or `FAILED`;
- current step and numeric task status;
- worker count and operating-system PID;
- source Git SHA and UTC timestamps;
- log/report paths, gate outcome, and report-push status.

Writes use atomic replacement. A job is complete only when state is terminal; scientific
progression additionally requires the stage-specific gate to pass. GPU utilization is diagnostic,
never a completion signal.

## Frozen stage order

```text
c0 -> screening -> lomo -> confirmatory -> confirmatory-lomo -> reference-safety
```

The controller validates required gates and artifact hashes before launching dependent stages.
Explicit `start --stage ...` is required after a failed process so a transient failure cannot cause
an infinite restart loop.

After the valid v1 screening failure, new screening and downstream jobs use the versioned
`configs/experiment/fp_naa_v2.yaml` tail-safe objective. The immutable v1 reports remain valid
negative-ablation artifacts; the shared BEATs and augmentation caches are reusable because v2 does
not change their frontend or augmentation contracts.

The public `fp-naa runtime-check` also executes CUDA smoke probes for the tail-safe objective and
primary-safe gradient projection, so server validation remains one short command.
