# Hardware profile

**Status:** Not yet finalized. Phase 10 must not hard-code device-specific
backends until this document is complete.

## Device inventory (to fill)

| Device | Exact model | RAM | OS / JetPack | Accelerator | Power mode | Cooling | Runtime |
|--------|-------------|-----|--------------|-------------|------------|---------|---------|
| Pi 4 | | | | CPU | | | |
| Pi 5 | | | | CPU | | | |
| Pi 5 + Hailo | | | | Hailo-8 26 TOPS | | | |
| Jetson Nano | | | | CUDA/TensorRT | | | |
| Jetson NX | | | | | | | |
| Jetson AGX | | | | | | | |

## Microphone / capture (to fill)

| Item | Value |
|------|-------|
| Microphone model | |
| Number of mics | |
| USB / audio interface | |
| Sync (HW / SW) | |
| Sample rate | |
| Bit depth | |
| Expected spacing | |
| Self-collected data planned? | |

## Interim policy

Until this profile is filled:

- All latency and energy claims use **CPU reference** measurements only.
- Export paths may scaffold ONNX, but must not assume Xavier vs Orin vs Hailo operators.
