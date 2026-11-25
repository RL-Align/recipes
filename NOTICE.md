# Third-party provenance

The project scaffolding, orchestration scripts and documentation are MIT under
the root LICENSE.

`models/deepseek-v4/rocm/patches/eight-gpu-*.patch` reproduce the complete paired
review patches published with
[RL-Kernel #717](https://github.com/RL-Align/RL-Kernel/pull/717#issuecomment-7059960424)
at artifact commit `70d56040357a62c65ea0edbec55ff8a1d260841d` in
`a1021223029-lgtm/RL-Kernel`. Their original per-file Apache-2.0 notices remain.
The supplemental `recipe-*.patch` changes target the same Apache-2.0 source
files in RL-Kernel and verl and are distributed under Apache-2.0. The applicable
license text is at https://www.apache.org/licenses/LICENSE-2.0.
`scripts/rebuild_gfx950.sh` adapts the pinned RL-Kernel MI35x build instructions
and retains Apache-2.0 attribution.

Those patches do not relicense RL-Kernel, verl, vLLM, Megatron, image contents,
model weights or datasets. Refer to each upstream project's license and model /
dataset terms. Images and assets remain external references.
