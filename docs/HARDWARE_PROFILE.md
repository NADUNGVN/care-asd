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
| Mic / stereo capture on edge | **TBD** |

## Interim policy (CARE-ASD)

- Do **not** hard-code Hailo backend until §3.2.1 in shared doc is green.
- Prefer **CPU / ONNX Runtime** on E1 for early edge scaffolding.
- Latency/energy claims must label backend and commit; idle thermal baseline ~44°C, no throttle at audit.
- Note PCIe link **x1 downgraded** (capable x4) when interpreting Hailo throughput later.
- Phase 10 full export/benchmark after Hailo SW install + optional mic inventory.

## CARE-ASD-only deploy notes

| Item | Value |
|------|-------|
| Deploy device | **E1** — Raspberry Pi 5 Model B Rev 1.1 + Hailo-8 |
| Channel map (DCASE / CARE) | ch0=near, ch1=far |
| Near/far mic spacing for demos | TBD |
| Selective routing | student on E1; expert on train GPU servers (optional) |
