#!/bin/bash
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -N grpo_lora_1n
#PBS -l filesystems=flare:home
#PBS -A Intel-Aurora
#PBS -j oe
#
# Single-node GRPO+LoRA on Intel XPU (4 tiles: 2 trainer + 2 generator).
#
# Interactive:
#   qsub -I -l select=1 -l walltime=01:00:00 -A <account> -q <queue>
#   bash torchtitan/experiments/rl/run_grpo_lora_sn.sh
# Batch:
#   qsub torchtitan/experiments/rl/run_grpo_lora_sn.sh
#
# Overridable: CONFIG NUM_STEPS HF_ASSETS_PATH ZE_AFFINITY_MASK DUMP_FOLDER.
# Extra args are appended to the training command verbatim, e.g.
#   NUM_STEPS=200 bash run_grpo_lora_sn.sh \
#       --trainer.parallelism.data_parallel_shard_degree=2
#
# The former run_grpo_lora_xpu_1node_dp8.sh diagnostic (dp_shard=8 on one node,
# 12 tiles) is reachable from here -- that question is settled, but to re-run it:
#   ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7,8,9,10,11 CONFIG=rl_grpo_lora_qwen3_0_6b_dp8 \
#       DUMP_FOLDER=outputs/rl_lora_1n_dp8 bash run_grpo_lora_sn.sh

# No `set -e`: an accidental `source` of this file should not kill the shell.

source ~/env-3.sh
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate monarch

export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1,2,3}
export CCL_OFI_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64/libfabric.so.1
export FI_PROVIDER=cxi
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_OFI_PROVIDER=cxi
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:${LD_LIBRARY_PATH}
export FI_PROVIDER_PATH=/opt/cray/libfabric/1.22.0/lib64/libfabric
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-hsn0}
export TORCHINDUCTOR_CACHE_DIR=~/.cache/torchinductor_xpu
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# vLLM's Triton kernels JIT with icpx, which borrows libstdc++ from the gcc on
# PATH; Aurora's default gcc 7.5 lacks <filesystem>. Point icpx at gcc 13.4.
GCC13_ROOT=/opt/aurora/26.26.0/spack/unified/1.1.1/install/linux-x86_64/gcc-13.4.0-hgnyg4p
if [ -d "$GCC13_ROOT" ]; then
    export CCC_OVERRIDE_OPTIONS="+--gcc-toolchain=$GCC13_ROOT"
    export PATH="$GCC13_ROOT/bin:$PATH"
fi

# No LD_PRELOAD interposer here: the patched RMSNorm backward kernel is needed
# only by FULL-parameter GRPO, which trains the qk_norm weights. LoRA freezes
# them, so its backward never asks for that weight gradient. See run_grpo_sn.sh
# for the full-parameter single-node arm.

HF_ASSETS_PATH=${HF_ASSETS_PATH:-/flare/Aurora_deployment/intel/models/Qwen3-0.6B}
CONFIG=${CONFIG:-rl_grpo_lora_qwen3_0_6b}
NUM_STEPS=${NUM_STEPS:-10}
DUMP_FOLDER=${DUMP_FOLDER:-outputs/rl_lora_1n}

# A stale checkpoint from a different mesh shape fails to load, so start clean.
rm -rf ~/git/torchtitan/"$DUMP_FOLDER"/checkpoint/ 2>/dev/null
rm -rf ~/.cache/torchinductor_xpu/triton 2>/dev/null
cd ~/git/torchtitan || { echo "ERROR: ~/git/torchtitan not found"; exit 1; }

# Fail early with a clear message if the model or XPU devices aren't visible,
# rather than deep inside the training stack.
if [ ! -f "$HF_ASSETS_PATH/config.json" ]; then
    echo "ERROR: no model at $HF_ASSETS_PATH (config.json missing)."
    echo "       Set HF_ASSETS_PATH or download Qwen3-0.6B there first."
    exit 1
fi
if ! python3 -c "import torch; assert torch.xpu.is_available() and torch.xpu.device_count() > 0" 2>/dev/null; then
    echo "ERROR: no XPU devices visible. Run this on a compute node (not the UAN)"
    echo "       with oneAPI sourced and ZE_AFFINITY_MASK set."
    exit 1
fi

echo "=== Single-node GRPO+LoRA: ${CONFIG}, tiles ${ZE_AFFINITY_MASK} ==="

python3 -m torchtitan.experiments.rl.train \
    --module alphabet_sort --config "$CONFIG" \
    --hf_assets_path="$HF_ASSETS_PATH" \
    --async_loop.num_training_steps="$NUM_STEPS" \
    --dump_folder="$DUMP_FOLDER" \
    --generator.gpu_memory_limit=0.90 \
    "$@" \
    2>&1 | tee torchtitan/experiments/rl/train_lora_1n.log
exit "${PIPESTATUS[0]}"
