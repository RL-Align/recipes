# Validation record — 2026-11-25

Environment: CPU-only Linux, Python 3.12. No ROCm GPU or Docker daemon was
available for the checks below. This is not a device acceptance report.

| Check | Result |
| --- | --- |
| Recipe allocation, math workload, Hydra/trainer config, placement, raw-bit negative control and eight-replica sync barrier/failure cases | 22 passed |
| Paired verl eight-GPU and multi-node CPU regressions, against the assembled source | 30 passed |
| Paired RL-Kernel eight-GPU launcher/DDP CPU regressions | 10 passed |
| Fresh source fetch → checksummed patch application → exact expected Git tree | Passed for both repositories |
| Ruff 0.12.2 lint and format | Passed |
| Python syntax, including patched source under Python 3.10 grammar | Passed |
| gfx950 build-script shell syntax and local Markdown links | Passed |
| Derived image build and binary rebuild | **Not executed here** |
| New 8-GPU / 32-GPU model runs | **Not executed here** |
| DAPO/GSM8K 200-update acceptance or performance | **Not executed here** |

The six topology/profile combinations compose the actual pinned verl Hydra
configuration and pass its trainer configuration contract. The 32-GPU tests
require all actor ranks, unique physical placement and correctly pinned rollout
instances. The synchronization negative control fails the last of eight
replicas and checks that no replica resumes admission.

CPU tests do not validate RCCL transport, native GPU arithmetic, host-driver
compatibility, long-context parity, math rewards, throughput or 200 successful
updates. AMD execution must produce its own complete retained evidence using
the [operator guide](amd-quickstart.zh-CN.md).
