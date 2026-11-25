#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build commands adapted from RL-Kernel/docs/usage/dsv4-mi35x-validation.md
# at 71c7cb2d033806632cef1a3678939bf73dc581dd. No external source patch/upgrade.
set -euo pipefail
export TMPDIR=/tmp PYTORCH_ROCM_ARCH=gfx950
export ROCM_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel
export ROCM_HOME="$ROCM_PATH" HIP_PATH="$ROCM_PATH"
export PATH="/opt/dsv4-venv/bin:$ROCM_PATH/bin:$PATH"
test "$(git -C /opt/dsv4/vllm rev-parse HEAD)" = c07d58dc82d75f7d1a947b3264b460384ec3e18d
grep -q b5c00f43283086bf0751f3bf3a179f1e921bbe59 /opt/dsv4/artifacts/te-source-sha.txt

cmake -S /opt/dsv4/vllm -B /tmp/recipe-vllm-gfx950 -G Ninja \
  -DVLLM_TARGET_DEVICE=rocm \
  -DVLLM_PYTHON_EXECUTABLE=/opt/dsv4-venv/bin/python \
  -DROCM_PATH="$ROCM_PATH" -DCMAKE_HIP_ARCHITECTURES=gfx950 \
  -DCMAKE_INSTALL_PREFIX=/opt/dsv4/vllm -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DFETCHCONTENT_BASE_DIR=/opt/dsv4/vllm/.deps -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
  -DCMAKE_JOB_POOLS=compile=16 -DCMAKE_JOB_POOL_COMPILE=compile
cmake --build /tmp/recipe-vllm-gfx950 \
  --target _C _C_stable_libtorch _moe_C_stable_libtorch _rocm_C -j 16
for component in _C _C_stable_libtorch _moe_C_stable_libtorch _rocm_C; do
  cmake --install /tmp/recipe-vllm-gfx950 --component "$component"
done

export NVTE_ROCM_ARCH=gfx950 NVTE_USE_ROCM=1
export HIP_DEVICE_LIB_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/llvm/amdgcn/bitcode
export MAX_JOBS=16 CMAKE_BUILD_PARALLEL_LEVEL=16
cmake -S /opt/dsv4/TransformerEngine/transformer_engine/common \
  -B /tmp/recipe-te-gfx950 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/dsv4/TransformerEngine \
  -DPython_EXECUTABLE=/opt/dsv4-venv/bin/python \
  -DUSE_ROCM=ON -DUSE_FUSED_ATTN_AOTRITON=OFF \
  -DUSE_FUSED_ATTN_CK=ON -DUSE_HIPKITTENS_GEMM=ON \
  -DNVTE_BUILD_LEGACY_STATIC_NORM=ON -DNVTE_BUILD_LEGACY_STATIC_FUSED_SOFTMAX=ON \
  -DCK_FUSED_ATTN_FLOAT_TO_BFLOAT16_DEFAULT=3 \
  -DTE_ROCM_PURELIB=/opt/dsv4-venv/lib/python3.12/site-packages \
  -DCMAKE_JOB_POOLS=compile=16 -DCMAKE_JOB_POOL_COMPILE=compile
cmake --build /tmp/recipe-te-gfx950 -j 16
cmake --install /tmp/recipe-te-gfx950
