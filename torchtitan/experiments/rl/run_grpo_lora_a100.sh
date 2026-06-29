#!/usr/bin/env bash
# Launch TorchTitan RL/GRPO + LoRA on a single Polaris node.
# Usage: bash run_grpo_lora_a100.sh
#   MODE=flex|grpo|grpo-lora|grpo-lora-flex (default: grpo-lora-flex)
#   STEPS=N  SMOKE=1  CONDA_ENV_PATH=...  HF_ASSETS_PATH=...
set -euo pipefail

TORCHTITAN_ROOT="${TORCHTITAN_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/lus/grand/projects/Intel/songhappy/envs/monarch-cu128}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
MODE="${MODE:-grpo-lora-flex}"
SMOKE="${SMOKE:-0}"
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/lus/grand/projects/Intel/models/Qwen3-0.6B}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VLLM_TORCH_COMPILE_LEVEL="${VLLM_TORCH_COMPILE_LEVEL:-0}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
# flashinfer JIT needs libcudart.so for linking; pip-installed nvidia-cuda-runtime has it
CUDA_RT_LIB="$CONDA_ENV_PATH/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
export LD_LIBRARY_PATH="${CUDA_RT_LIB}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_RT_LIB}:${LIBRARY_PATH:-}"

source "$CONDA_SH"
conda activate "$CONDA_ENV_PATH"
cd "$TORCHTITAN_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Pre-build flashinfer JIT cache. Polaris HPC SDK cuda/12.9 has nvcc but
# libcudart is in pip nvidia-cuda-runtime and curand.h is in math_libs.
# Create a merged CUDA_HOME with symlinks so flashinfer finds everything.
FLASHINFER_CUDA="$HOME/.local/flashinfer_cuda"
rm -rf "$HOME/.cache/flashinfer" "$FLASHINFER_CUDA"
if [[ ! -f "$HOME/.cache/flashinfer/0.6.12/80/cached_ops/sampling/sampling.so" ]]; then
    echo "Pre-building flashinfer JIT cache..."
    mkdir -p "$FLASHINFER_CUDA/lib64" "$FLASHINFER_CUDA/include"
    NV_PKGS="$CONDA_ENV_PATH/lib/python3.12/site-packages/nvidia"
    HPC_CUDA="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9"
    MATH_LIBS="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux"
    # bin: symlink entire directory (nvcc needs cudafe++, cicc, ptxas, etc.)
    ln -sfn "$HPC_CUDA/bin" "$FLASHINFER_CUDA/bin"
    # includes: cuda headers + curand from math_libs
    ln -sf "$HPC_CUDA/include/"* "$FLASHINFER_CUDA/include/" 2>/dev/null || true
    ln -sf "$MATH_LIBS/include/curand"* "$FLASHINFER_CUDA/include/" 2>/dev/null || true
    # libs: libcudart from pip, libcuda stub from HPC SDK
    ln -sf "$NV_PKGS/cuda_runtime/lib/libcudart.so.12" "$FLASHINFER_CUDA/lib64/libcudart.so"
    ln -sf "$HPC_CUDA/lib64/stubs/libcuda.so" "$FLASHINFER_CUDA/lib64/libcuda.so" 2>/dev/null || true
    CUDA_HOME="$FLASHINFER_CUDA" \
        python -c "from flashinfer.sampling import gen_sampling_module; gen_sampling_module().build_and_load(); print('flashinfer cache OK')"
fi

case "$MODE" in
    grpo)           CONFIG_NAME="rl_grpo_qwen3_0_6b_varlen";      EXPECTED_GPUS=6 ;;
    grpo-lora)      CONFIG_NAME="rl_grpo_qwen3_0_6b_varlen_lora"; EXPECTED_GPUS=6 ;;
    flex)           CONFIG_NAME="rl_grpo_qwen3_0_6b_flex";         EXPECTED_GPUS=4 ;;
    grpo-lora-flex) CONFIG_NAME="rl_grpo_qwen3_0_6b_flex_lora";   EXPECTED_GPUS=4 ;;
    *) echo "ERROR: unknown MODE=$MODE" >&2; exit 1 ;;
esac

echo "=== TorchTitan RL: $MODE ($CONFIG_NAME, ${EXPECTED_GPUS} GPUs) ==="

if [[ "$SMOKE" == "1" ]]; then
    python -c "
from torchtitan.experiments.rl.examples.alphabet_sort.config_registry import $CONFIG_NAME
cfg = ${CONFIG_NAME}()
print('OK: $CONFIG_NAME')
print('  trainer TP =', cfg.trainer.parallelism.tensor_parallel_degree)
print('  generator TP =', cfg.generator.parallelism.tensor_parallel_degree)
"
    exit 0
fi

GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( GPU_COUNT < EXPECTED_GPUS )); then
    echo "ERROR: need $EXPECTED_GPUS GPUs, only $GPU_COUNT visible" >&2; exit 1
fi

EXTRA_ARGS=()
[[ -n "${STEPS:-}" ]] && EXTRA_ARGS+=(--num_steps "$STEPS")

set -x
exec python -m torchtitan.experiments.rl.train \
    --module alphabet_sort \
    --config "$CONFIG_NAME" \
    --hf_assets_path "$HF_ASSETS_PATH" \
    --metrics.no-enable-wandb \
    --metrics.enable-tensorboard \
    "${EXTRA_ARGS[@]}"
