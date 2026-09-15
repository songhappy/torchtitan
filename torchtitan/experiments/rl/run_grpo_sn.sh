#!/bin/bash
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -N grpo_full_1n
#PBS -l filesystems=flare:home
#PBS -A Intel-Aurora
#
# Single-node full-parameter GRPO on Intel XPU (4 tiles: 2 trainer + 2 generator).
#
# Interactive:
#   qsub -I -l select=1 -l walltime=01:00:00 -A <account> -q <queue>
#   bash torchtitan/experiments/rl/run_grpo_sn.sh
# Batch:
#   qsub torchtitan/experiments/rl/run_grpo_sn.sh
#
# Overridable: CONFIG NUM_STEPS HF_ASSETS_PATH ZE_AFFINITY_MASK INTERPOSER
# DUMP_FOLDER. Extra args are appended to the training command verbatim, e.g.
#   NUM_STEPS=200 bash run_grpo_sn.sh --trainer.parallelism.data_parallel_shard_degree=2

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

# REQUIRED FOR FULL GRPO, not for LoRA. Full GRPO trains the qk_norm weights, so
# its backward asks aten::_fused_rms_norm_backward for a weight gradient; above
# xe_core_count * 1024 rows the stock XPU kernel dies with "tensor does not have
# a device". The interposer supplies the patched kernel via LD_PRELOAD without
# touching the conda env. Build it with build_rmsnorm_interposer.sh.
# Exported so every process train.py spawns inherits it -- the crash is inside a
# trainer rank's backward, not in this shell.
INTERPOSER=${INTERPOSER:-/home/songhappy/git/torchtitan/torchtitan/experiments/rl/rmsnorm_interposer/libinterpose_layernorm.so}
if [ ! -f "$INTERPOSER" ]; then
    echo "FATAL: interposer not found at $INTERPOSER"
    echo "       Run: bash torchtitan/experiments/rl/build_rmsnorm_interposer.sh"
    exit 1
fi
export LD_PRELOAD=$INTERPOSER
echo "LD_PRELOAD = $LD_PRELOAD"

HF_ASSETS_PATH=${HF_ASSETS_PATH:-/flare/Aurora_deployment/intel/models/Qwen3-0.6B}
CONFIG=${CONFIG:-rl_grpo_full_qwen3_0_6b_flex}
NUM_STEPS=${NUM_STEPS:-10}
DUMP_FOLDER=${DUMP_FOLDER:-outputs/rl_full_1n}

# A stale checkpoint from a different mesh shape fails to load, so start clean.
rm -rf ~/git/torchtitan/"$DUMP_FOLDER"/checkpoint/ 2>/dev/null
rm -rf ~/.cache/torchinductor_xpu/triton 2>/dev/null
cd ~/git/torchtitan || { echo "ERROR: ~/git/torchtitan not found"; exit 1; }

if [ ! -f "$HF_ASSETS_PATH/config.json" ]; then
    echo "ERROR: no model at $HF_ASSETS_PATH (config.json missing)."
    exit 1
fi
if ! python3 -c "import torch; assert torch.xpu.is_available() and torch.xpu.device_count() > 0" 2>/dev/null; then
    echo "ERROR: no XPU devices visible. Run on a compute node with oneAPI sourced."
    exit 1
fi

# Confirm the preload actually took effect before spending the allocation: a
# silently ineffective LD_PRELOAD would just reproduce the old crash mid-run.
python3 -c "
import torch
with open('/proc/self/maps') as f:
    mapped = 'libinterpose_layernorm.so' in f.read()
assert mapped, 'LD_PRELOAD did not take effect'
x = torch.randn(2, 2048, 16, 128, device='xpu', dtype=torch.bfloat16, requires_grad=True)
torch.nn.RMSNorm(128, eps=1e-6, dtype=torch.bfloat16).to('xpu')(x).sum().backward()
torch.xpu.synchronize()
print('interposer active; 65536-row RMSNorm weight-grad backward: OK')
"
if [ $? -ne 0 ]; then
    echo "FATAL: interposer precheck failed, not starting the pipeline."
    exit 1
fi

python3 -m torchtitan.experiments.rl.train \
    --module alphabet_sort --config "$CONFIG" \
    --hf_assets_path="$HF_ASSETS_PATH" \
    --async_loop.num_training_steps="$NUM_STEPS" \
    --dump_folder="$DUMP_FOLDER" \
    "$@" \
    2>&1 | tee torchtitan/experiments/rl/train_full_1n.log
exit "${PIPESTATUS[0]}"
