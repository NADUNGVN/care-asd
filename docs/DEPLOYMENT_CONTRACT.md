# Deployment contract

## Boundary

Research code remains Python 3.11+. Jetson Xavier devices run JetPack 5.1.5
with TensorRT through the C++ runner. An engine is always built on its target
device; never copy an engine from AGX Xavier to Xavier NX or the reverse.

## Bundle v1

`care-asd bundle-validate BUNDLE_DIR` validates a no-overwrite bundle before
engine build or benchmark. A bundle contains:

| File | Contract |
|---|---|
| `bundle.json` | schema version, config hash, input tensor and hashes |
| `model.onnx` | exported encoder graph |
| `scorer.json` | normal-only Mahalanobis mean and precision matrix |
| `calibration.json` | fixed normal-only threshold and abstain margin |

All artifact names are bundle-local and every artifact is SHA-256 checked.
Altered artifacts, missing files, invalid scorer dimensions, or overwritten
bundles fail validation.

## Safe CARE invariant

The Python front-end returns the unmodified `near_stft` plus a bounded
`residual_stft` auxiliary view. Downstream export must preserve the near view;
the residual cannot replace it. The runtime bundle is created only after this
front-end, feature layout, scorer, and calibration policy have been frozen.

## TensorRT runner

`deployment/tensorrt_runner` is a C++17 utility for a fixed-shape TensorRT
engine. It performs real GPU inference from a float32 feature tensor, writes
the encoder output, and reports warm-up and p50/p95/p99 model latency. It is
deliberately model-only: CPU safe-CARE preprocessing and score/calibration
parity are measured separately until their C++ implementation is frozen.

The runner rejects engines with zero/multiple inputs or outputs, dynamic
dimensions, malformed input length, and CUDA/TensorRT failures. Its output
must be compared against the Python/ONNX reference before reporting device
metrics.
