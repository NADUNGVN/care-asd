# CARE-ASD TensorRT runner

Build on a Xavier device after installing JetPack TensorRT packages:

```bash
cmake -S deployment/tensorrt_runner -B build/tensorrt_runner
cmake --build build/tensorrt_runner --config Release
```

The runner executes a fixed-shape engine with a raw float32 input tensor:

```bash
care_asd_trt_runner \
  --engine model.plan \
  --input features.f32 \
  --input-elements 32768 \
  --output embedding.f32 \
  --warmup 100 \
  --iterations 1000 \
  --report model_latency.json
```

Validate the source deployment bundle before building `model.plan`:

```bash
care-asd bundle-validate path/to/bundle
```

The engine must expose exactly one fixed-shape float32 input and one float32
output. The JSON report is limited to model latency; follow
[`../../docs/JETSON_BENCHMARK_PROTOCOL.md`](../../docs/JETSON_BENCHMARK_PROTOCOL.md)
for end-to-end and energy reporting.
