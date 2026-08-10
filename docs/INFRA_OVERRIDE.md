# CARE-ASD — infrastructure notes

Shared inventory (servers + edge hardware) lives at the workspace level:

→ **[`../../docs/SHARED_INFRASTRUCTURE.md`](../../docs/SHARED_INFRASTRUCTURE.md)**

Do not duplicate host/GPU/board tables here.

## CARE-ASD-specific mapping

| Item | Value |
|------|--------|
| `project_id` | `care-asd` |
| Dataset on shared disk | _(e.g. under NFS `/home/ubuntu/.../datasets/dcase2026/`)_ — TBD |
| Outputs on shared disk | _(e.g. under NFS `.../experiments/care-asd/`)_ — TBD |
| Primary train servers | SERVER-01…04 (see shared §2.1.1) |
| Deploy device IDs | **E1** (Pi5 + Hailo-8), **E2** (Xavier NX), **E3** (AGX Xavier) |
| Hailo on E1 | Hardware PCIe **present**; software stack **not installed** (2026-08-06 audit) |
| Jetson E2/E3 | Selected for TensorRT study; inventory and external-meter setup pending |
| Local machine role | Code, unit tests, docs only — **no heavy train** |

## Related CARE-ASD docs

| File | Role |
|------|------|
| [`HARDWARE_PROFILE.md`](HARDWARE_PROFILE.md) | Status + interim no-Hailo-until-ready policy |
| [`DATASET.md`](DATASET.md) | DCASE protocol (scientific), not server paths |
