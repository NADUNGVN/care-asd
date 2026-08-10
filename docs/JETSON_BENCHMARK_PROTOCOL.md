# Jetson benchmark protocol

## Devices and software

| Role | Device | Profiles |
|---|---|---|
| E2 primary student | Jetson Xavier NX Developer Kit | TensorRT FP16/INT8, 10 W and 15 W |
| E3 expert/comparison | Jetson AGX Xavier Developer Kit | same student at 15 W; pre-frozen expert at 30 W |
| E1 comparison | Raspberry Pi 5 + Hailo-8 | CPU/ONNX now; Hailo only after runtime smoke test |

Record exact module SKU/RAM, carrier board, JetPack release, TensorRT version,
cooling, `nvpmodel`, clock policy, ambient temperature, engine hash, bundle
config hash, and git commit. Xavier runs JetPack 5.1.5; do not upgrade it to a
JetPack release unsupported by Xavier.

## Procedure

1. Validate the deployment bundle and build the TensorRT engine on the target.
2. Replay identical precomputed float32 feature windows on every device.
3. Run 100 warm-up windows, then 1,000 timed windows, for three independent
   repetitions under each profile.
4. Capture model latency p50/p95/p99, end-to-end CPU-preprocess/model/postprocess
   latency, real-time factor, peak RSS, steady-state temperature, and throttle
   state.
5. Measure system energy with a documented external meter. Subtract an equal
   duration idle trace, then divide by processed 10-second clips. Without a
   meter, energy is `not_measured`, never inferred from a power mode.
6. Run numerical parity against Python reference features/scores and report
   maximum absolute and relative differences before any accuracy claim.

## Reporting rules

- Same student model at 15 W is the only cross-device platform comparison.
- AGX 30 W expert and any routing threshold are selected on development data
  before evaluation freeze.
- INT8 calibration uses normal training data only and reports accuracy delta
  against FP16.
- Hailo numbers are forbidden until HailoRT, `/dev/hailo0`, and a project smoke
  test are recorded in the hardware inventory.
