# Hardware profile (CARE-ASD)

**Canonical inventory** for train servers and edge deploy hardware is **shared** across all Teacher_Vu research tracks:

→ **[`../../docs/SHARED_INFRASTRUCTURE.md`](../../docs/SHARED_INFRASTRUCTURE.md)**  
  - Section 2 — Train servers (`SERVER-01`…`04`)  
  - Section 3 — Hardware deploy (**E1** Pi5 + Hailo-8)  
  - Section 4 — Project mapping row for `CARE_ASD/`

Paper-specific path overrides: [`INFRA_OVERRIDE.md`](INFRA_OVERRIDE.md).

## Status

| Topic | Status |
|-------|--------|
| Shared template | `infra-v1` |
| Train inventory | **4/4 filled** |
| Edge E1 (Pi5 + Hailo-8) | **Board + PCIe Hailo-8 confirmed** |
| Hailo runtime on E1 | **Not installed** (`/dev/hailo*` missing; no hailort) |
| Edge E2 (Xavier NX Developer Kit) | **Selected; physical inventory pending** |
| Edge E3 (AGX Xavier Developer Kit) | **Selected; physical inventory pending** |
| Mic / stereo capture on edge | **TBD** |

## Interim policy (CARE-ASD)

- Do **not** hard-code Hailo backend until §3.2.1 in shared doc is green.
- Prefer **CPU / ONNX Runtime** on E1 for early edge scaffolding.
- Latency/energy claims must label backend and commit; idle thermal baseline ~44°C, no throttle at audit.
- Note PCIe link **x1 downgraded** (capable x4) when interpreting Hailo throughput later.
- Jetson export/benchmark starts after E2/E3 inventory and external-meter setup;
  Hailo remains a supplemental comparison after its software smoke test.
- Xavier E2/E3 use JetPack 5.1.5 + TensorRT C++ runner; inventory exact module
  SKU/RAM, carrier, cooling and power meter before a benchmark is valid.

## CARE-ASD-only deploy notes

| Item | Value |
|------|-------|
| Deploy device | **E1** — Raspberry Pi 5 Model B Rev 1.1 + Hailo-8 |
| Primary student device | **E2** — Jetson Xavier NX Developer Kit (10/15 W) |
| Expert/comparison device | **E3** — Jetson AGX Xavier Developer Kit (15/30 W) |
| Channel map (DCASE / CARE) | ch0=near, ch1=far |
| Near/far mic spacing for demos | TBD |
| Selective routing | student on E1; expert on train GPU servers (optional) |
